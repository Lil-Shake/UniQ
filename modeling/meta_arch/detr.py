import logging
import math
from multiprocessing import Condition
from turtle import back
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks, pairwise_iou
from detectron2.utils.logger import log_first_n
from detectron2.data import MetadataCatalog
from fvcore.nn import giou_loss, smooth_l1_loss
from ..transformer import build_detr, build_criterion, build_transformer, build_matcher, build_position_encoding
from ..transformer.segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from ..transformer.util.utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, NestedTensor, convert_coco_poly_to_mask
from ..transformer.util import box_ops
from ..backbone import MaskedBackbone, Joiner, Backbone, DeformableDETRMaskedBackbone, DeformableDETRJoiner
from detectron2.layers import batched_nms

__all__ = ["Detr"]


@META_ARCH_REGISTRY.register()
class Detr(nn.Module):
    """
    Implement Detr
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_relation_classes = cfg.MODEL.DETR.NUM_RELATION_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        self.use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.use_gt_label = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        self.later_nms_thres = cfg.MODEL.DETR.LATER_NMS_THRESHOLD
        self.use_freq_bias = cfg.MODEL.DETR.USE_FREQ_BIAS
        self.test_index = cfg.MODEL.DETR.TEST_INDEX
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        num_relation_queries = cfg.MODEL.DETR.NUM_RELATION_QUERIES
        create_bg_pairs = cfg.MODEL.DETR.CREATE_BG_PAIRS
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        obj_dec_layers = cfg.MODEL.DETR.OBJECT_DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        no_rel_weight = cfg.MODEL.DETR.NO_REL_WEIGHT
        cost_class = cfg.MODEL.DETR.COST_CLASS
        nms_weight = cfg.MODEL.DETR.NMS_WEIGHT
        cost_selection = cfg.MODEL.DETR.COST_SELECTION
        beta = cfg.MODEL.DETR.BETA
        self.matcher_topk = cfg.MODEL.DETR.MATCHER_TOPK
        self.nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        # TODO Group DETR
        self.num_group = cfg.MODEL.DETR.GROUP_DETR

        d2_backbone = MaskedBackbone(cfg)
        position_embedding = build_position_encoding(cfg.MODEL.DETR.POSITION_EMBEDDING, hidden_dim)
        backbone = Joiner(d2_backbone, position_embedding)
        backbone.num_channels = d2_backbone.num_channels
        # TODO Group DETR
        transformer = build_transformer(cfg.MODEL.DETR.TRANSFORMER, d_model=hidden_dim, dropout=dropout, nhead=nheads, dim_feedforward=dim_feedforward, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, normalize_before=pre_norm, return_intermediate_dec=deep_supervision, num_object_decoder_layers=obj_dec_layers, num_classes=self.num_classes, num_relation_classes=self.num_relation_classes, beta=beta, num_group=self.num_group)
        # TODO Group DETR
        self.detr = build_detr(cfg.MODEL.DETR.NAME, backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision, use_gt_box=self.use_gt_box, use_gt_label = self.use_gt_label, num_relation_queries=num_relation_queries, num_group=self.num_group)
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm
        if cfg.MODEL.DETR.FROZEN_WEIGHTS != '':
            self.load_detr_from_pretrained(cfg.MODEL.DETR.FROZEN_WEIGHTS)
        self.detr.to(self.device)

        # building criterion
        # TODO Group DETR
        matcher = build_matcher(cfg.MODEL.DETR.MATCHER, cost_class=cost_class, cost_bbox=l1_weight, cost_giou=giou_weight, num_group=self.num_group, topk=self.matcher_topk)
        weight_dict = {"loss_ce": cost_class, "loss_bbox": l1_weight, 'loss_ce_subject': cost_class, 'loss_ce_object': cost_class, 'loss_bbox_subject': l1_weight, 'loss_bbox_object': l1_weight, 'loss_giou_subject': giou_weight, 'loss_giou_object': giou_weight, 'loss_relation': 1, 'loss_bbox_relation': l1_weight, 'loss_giou_relation':giou_weight, 'loss_nms':nms_weight, 'loss_selection_subject': cost_selection, 'loss_selection_object': cost_selection}
        weight_dict["loss_giou"] = giou_weight
        # weight_dict = {"loss_ce": 1.0, "loss_bbox": 1.0, 'loss_ce_subject': 1.0, 'loss_ce_object': 1.0, 'loss_bbox_subject': 1.0, 'loss_bbox_object': 1.0, 'loss_giou_subject': 1.0, 'loss_giou_object': 1.0, 'loss_relation': 1.0, 'loss_bbox_relation': 1.0, 'loss_giou_relation':1.0, 'loss_nms':nms_weight, 'loss_selection_subject': cost_selection, 'loss_selection_object': cost_selection}
        # weight_dict["loss_giou"] = 1.0
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(max(dec_layers, obj_dec_layers) - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        print (weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        statistics = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).statistics
        self.criterion = build_criterion(cfg.MODEL.DETR.CRITERION, self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses, use_gt_box=self.use_gt_box, use_gt_label=self.use_gt_label, num_relation_classes=self.num_relation_classes, intersection_iou_threshold=cfg.MODEL.DETR.INTERSECTION_IOU_THRESHOLD, intersection_iou_lambda=cfg.MODEL.DETR.INTERSECTION_IOU_LAMBDA, intersection_loss=cfg.MODEL.DETR.INTERSECTION_LOSS, rel_eos_coef=no_rel_weight, statistics=statistics, reweight_relations=cfg.MODEL.DETR.REWEIGHT_RELATIONS, reweight_rel_eos_coef=cfg.MODEL.DETR.REWEIGHT_REL_EOS_COEF, neg_rel_fraction=cfg.MODEL.DETR.NEGATIVE_RELATION_FRACTION, max_rel_pairs=cfg.MODEL.DETR.MAX_RELATION_PAIRS, use_reweight_log=cfg.MODEL.DETR.REWEIGHT_USE_LOG, focal_alpha=cfg.MODEL.DETR.FOCAL_ALPHA, create_bg_pairs=create_bg_pairs, oversample_param=cfg.MODEL.DETR.OVERSAMPLE_PARAM, undersample_param=cfg.MODEL.DETR.UNDERSAMPLE_PARAM)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)
        self._freeze_layers(layers=cfg.MODEL.DETR.FREEZE_LAYERS)
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print ("Number of Parameters:", pytorch_total_params)
        logging.getLogger('detectron2').log(logging.WARN, "Number of Parameters: {}".format(pytorch_total_params))

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def load_detr_from_pretrained(self, path):
        print("Loading DETR checkpoint from pretrained: ", path)
        pretrained_detr = torch.load(path)['model']
        pretrained_detr_without_class_head = {k: v for k, v in pretrained_detr.items() if 'class_embed' not in k}
        self.detr.load_state_dict(pretrained_detr_without_class_head, strict=False)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)
        
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets, self.num_group)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

@META_ARCH_REGISTRY.register()
class UniqDetr(Detr):
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets((gt_instances, gt_relations))
            
            # TODO Group DETR
            loss_dict = self.criterion(output, targets, self.num_group)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            results = self.inference(output, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r,
                                          "rel_pair_idxs": results_per_image._rel_pair_idxs,
                                          "pred_rel_scores": results_per_image._pred_rel_scores,
                                          "pred_rel_labels": results_per_image._pred_rel_labels,
                                          "pred_class_prob": results_per_image._pred_rel_class_prob,
                                         })
            return processed_results

    def inference(self, output, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        
        if self.test_index == -1:
            logits_r = F.softmax(output['relation_logits'], -1)
            bs, nq, num_rel_class = logits_r.shape

            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores_s, labels_s = F.softmax(output['relation_subject_logits'], -1)[:, :, :-1].max(-1)
            scores_o, labels_o = F.softmax(output['relation_object_logits'], -1)[:, :, :-1].max(-1)
            # if self.matcher_topk == 1:
            scores_r, labels_r = logits_r[:, :, :-1].max(-1)

            box_s = output['relation_subject_boxes']
            box_o = output['relation_object_boxes']
        else:
            logits_r = F.softmax(output['aux_outputs_r'][self.test_index]['pred_logits'], -1)
             # For each box we assign the best class or the second best if the best on is `no_object`.
            scores_s, labels_s = F.softmax(output['aux_outputs_r_sub'][self.test_index]['pred_logits'], -1)[:, :, :-1].max(-1)
            scores_o, labels_o = F.softmax(output['aux_outputs_r_obj'][self.test_index]['pred_logits'], -1)[:, :, :-1].max(-1)
            scores_r, labels_r = logits_r[:, :, :-1].max(-1)
            
            box_s = output['aux_outputs_r_sub'][self.test_index]['pred_boxes']
            box_o = output['aux_outputs_r_obj'][self.test_index]['pred_boxes']

        for i, (scores_per_image_s, labels_per_image_s, box_per_image_s, scores_per_image_o, labels_per_image_o, box_per_image_o, scores_per_image_r, labels_per_image_r, logits_per_image_r, image_size) in enumerate(zip(
            scores_s, labels_s, box_s, scores_o, labels_o, box_o, scores_r, labels_r, logits_r, image_sizes     
        )):
            
            image_boxes = Boxes(box_cxcywh_to_xyxy(torch.cat([box_per_image_s, box_per_image_o])))
            image_scores = torch.cat([scores_per_image_s, scores_per_image_o])
            image_pred_classes = torch.cat([labels_per_image_s, labels_per_image_o]) # [2*nq]                
            keep = batched_nms(image_boxes.tensor, image_scores, image_pred_classes, self.nms_thresh)
            keep_classes = image_pred_classes[keep]
            ious = pairwise_iou(image_boxes, image_boxes[keep])
            iou_assignments = torch.zeros_like(image_pred_classes) # [2*nq]
            for class_id in torch.unique(keep_classes):
                curr_indices = torch.where(image_pred_classes == class_id)[0]
                curr_keep_indices = torch.where(keep_classes == class_id)[0]
                curr_ious = ious[curr_indices][:, curr_keep_indices]
                curr_iou_assignment = curr_keep_indices[curr_ious.argmax(-1)]
                iou_assignments[curr_indices] = curr_iou_assignment
            
            result = Instances(image_size)
            result.pred_boxes = image_boxes[keep]
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])            
            result.scores = image_scores[keep]
            result.pred_classes = image_pred_classes[keep]
            
            rel_pair_idx = torch.stack(torch.split(iou_assignments, labels_per_image_s.size(0)), 1) # [nq, 2]
            triple_scores = scores_per_image_r * scores_per_image_s * scores_per_image_o
            _, sorting_idx = torch.sort(triple_scores, descending=True)
            
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = logits_per_image_r[sorting_idx]
            rel_labels = labels_per_image_r[sorting_idx]
            rel_scores = scores_per_image_r[sorting_idx]
            
            triplets = torch.cat((rel_pair_idx, rel_labels.unsqueeze(-1)), -1) #[nq, 3]
            
            unique_triplets = {}
            keep_triplet = torch.zeros_like(rel_labels)
            for idx, triplet in enumerate(triplets):
                # print(f'id:{idx}-{triplet}
                # TODO delete self connected edges
                if ("{}-{}-{}".format(triplet[0], triplet[1], triplet[2]) not in unique_triplets) and (triplet[0] != triplet[1]): #  
                    unique_triplets[ "{}-{}-{}".format(triplet[0], triplet[1], triplet[2])] = 1
                    keep_triplet[idx] = 1     
                     
            rel_pair_idx = rel_pair_idx[keep_triplet == 1] # (nq, 2)
            rel_class_prob = rel_class_prob[keep_triplet == 1] # (nq, #rel_class)
            rel_scores = rel_scores[keep_triplet == 1]
            rel_labels = rel_labels[keep_triplet == 1]
            
            # TODO Topk links 
            if self.matcher_topk > 1:
                rel_pair_idx = rel_pair_idx.unsqueeze(1).repeat(1, self.matcher_topk, 1).view(-1, 2) # (nq, 2) -> (nq*3, 2)
                rel_scores, rel_labels = torch.topk(rel_class_prob[:, :-1], self.matcher_topk, dim=-1) # (nq, #rel_class) -> (nq, matcher_topk)
                rel_scores = rel_scores.view(-1)
                rel_labels = rel_labels.view(-1)
                # rel_labels = rel_labels[keep_triplet == 1] # (nq, #rel_class)
                rel_class_prob = rel_class_prob.unsqueeze(1).repeat(1, self.matcher_topk, 1).view(-1, num_rel_class)
            
            result._rel_pair_idxs = rel_pair_idx # (#rel, 2)
            result._pred_rel_class_prob = rel_class_prob # (#rel, #rel_class)
            result._pred_rel_scores = rel_scores
            result._pred_rel_labels = rel_labels # (#rel, )
            results.append(result)
        return results

    def boxes_union(self, boxes1, boxes2):
        """
        Compute the union region of two set of boxes
        Arguments:
        box1: (Boxes) bounding boxes, sized [N,4].
        box2: (Boxes) bounding boxes, sized [N,4].
        Returns:
        (Boxes) union, sized [N,4].
        """
        assert len(boxes1) == len(boxes2)

        union_box = torch.cat((
            torch.min(boxes1.tensor[:,:2], boxes2.tensor[:,:2]),
            torch.max(boxes1.tensor[:,2:], boxes2.tensor[:,2:])
            ),dim=1)
        return Boxes(union_box)

    def get_center_coords(self, boxes):
        x0, y0, x1, y1 = boxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2]
        return torch.stack(b, dim=-1)

    def center_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            torch.abs(x1 - x0), torch.abs(y1 - y0)]
        return torch.stack(b, dim=-1)

    def prepare_targets(self, targets, box_threshold=1e-5):
        new_targets = []
        for image_idx, (targets_per_image, relations_per_image) in enumerate(zip(targets[0], targets[1])):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            subject_boxes = targets_per_image.gt_boxes[relations_per_image[:, 0]]
            object_boxes = targets_per_image.gt_boxes[relations_per_image[:, 1]]

            gt_boxes = self.boxes_union(subject_boxes, object_boxes)
            gt_boxes = gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            gt_subject_classes = targets_per_image.gt_classes[relations_per_image[:, 0]]
            gt_subject_boxes = subject_boxes.tensor / image_size_xyxy
            gt_subject_boxes = box_xyxy_to_cxcywh(gt_subject_boxes)

            gt_object_classes = targets_per_image.gt_classes[relations_per_image[:, 1]]
            gt_object_boxes = object_boxes.tensor / image_size_xyxy
            gt_object_boxes = box_xyxy_to_cxcywh(gt_object_boxes)

            gt_combined_classes = targets_per_image.gt_classes
            gt_combined_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_combined_boxes = box_xyxy_to_cxcywh(gt_combined_boxes)

            subject_boxes_center = self.get_center_coords(subject_boxes.tensor)
            object_boxes_center = self.get_center_coords(object_boxes.tensor)
            center_boxes = torch.cat([subject_boxes_center, object_boxes_center], -1)
            center_boxes = center_boxes / image_size_xyxy
            center_boxes = self.center_xyxy_to_cxcywh(center_boxes)

            # Remove degenerate boxes
            center_boxes_xyxy = Boxes(box_cxcywh_to_xyxy(center_boxes))
            center_boxes_xyxy.scale(scale_x=targets_per_image.image_size[1], scale_y=targets_per_image.image_size[0])
            center_masks = center_boxes_xyxy.nonempty(threshold=box_threshold)
            center_boxes[~center_masks] = gt_subject_boxes[~center_masks]

            gt_classes = relations_per_image[:, 2]
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'subject_boxes': gt_subject_boxes, 'object_boxes': gt_object_boxes, 'combined_boxes': gt_combined_boxes, 'subject_labels': gt_subject_classes, 'object_labels': gt_object_classes, 'combined_labels': gt_combined_classes, 'image_relations': relations_per_image, 'relation_boxes':center_boxes, 'relation_labels':relations_per_image[:, 2]})
        return new_targets

