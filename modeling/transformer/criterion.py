from collections import defaultdict
from email.policy import default
from re import I
import torch
import torch.nn.functional as F
from torch import nn

from .util import box_ops
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from scipy.optimize import linear_sum_assignment

import copy
import numpy as np
from detectron2.utils.registry import Registry
from .segmentation import sigmoid_focal_loss, dice_loss, cross_entropy_loss
from torchvision.ops.boxes import box_area
from collections import defaultdict
import copy
from scipy.optimize import linear_sum_assignment
from detectron2.structures.boxes import pairwise_iou, Boxes

CRITERION_REGISTRY = Registry("CRITERION_REGISTRY")

@CRITERION_REGISTRY.register()
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices) # 每个GT匹配到的Queries的索引
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 对于每一个batch，得到Quries对应的GT类别
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device) # 将目标类别全部设置为空
        target_classes[idx] = target_classes_o # 设置有匹配的Queries为它对应的GT

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) 
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, num_group):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * num_group
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

@CRITERION_REGISTRY.register()
class IterativeRelationCriterionBase(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_rel_classes = kwargs['num_relation_classes']
        self.statistics = kwargs['statistics']
        self.reweight_rel = kwargs['reweight_relations']
        self.use_reweight_log = kwargs['use_reweight_log']

        empty_weight_obj = torch.ones(self.num_classes + 1)
        empty_weight_obj[-1] = self.eos_coef
        empty_rel_weight = torch.ones(self.num_rel_classes + 1)
        empty_rel_weight[-1] = kwargs['rel_eos_coef']
        if self.reweight_rel:
            if self.use_reweight_log:
                empty_rel_weight = (self.statistics['fg_rel_count'].sum() / (self.statistics['fg_rel_count'] + 1e-5)).log()
            else:
                empty_rel_weight = (self.statistics['fg_rel_count'].sum() / (self.statistics['fg_rel_count'] + 1e-5))
            empty_rel_weight[-1] = kwargs['reweight_rel_eos_coef']
        print('*'*20)
        print(f'empty_rel_weight:{empty_rel_weight.shape}')
        self.register_buffer('empty_weight_obj', empty_weight_obj)
        self.register_buffer('empty_rel_weight', empty_rel_weight)
        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, loss_weight=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO use_focal_loss
        # use_focal_loss = False
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_obj)
        # if loss_weight is None:
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v[1]) for v in indices], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_src_permutation_idx_rel(self, indices):
        # permute predictions following indices
        batch_idx = []
        src_idx = []
        for i, src in enumerate(indices):
            if len(src) > 0:
                batch_idx.append(torch.full_like(src[0], i))
                src_idx.append(src[0])
        if len(batch_idx) > 0:
            batch_idx = torch.cat(batch_idx)
            src_idx = torch.cat(src_idx)
        else:
            batch_idx = torch.tensor([]).long()
            src_idx = torch.tensor([]).long()
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx_rel(self, indices):
        # permute predictions following indices
        batch_idx = []
        tgt_idx = []
        for i, tgt in enumerate(indices):
            if len(src) > 0:
                batch_idx.append(torch.full_like(tgt[1], i))
                tgt_idx.append(tgt[1])
        if len(batch_idx) > 0:
            batch_idx = torch.cat(batch_idx)
            tgt_idx = torch.cat(tgt_idx)
        else:
            batch_idx = torch.tensor([]).long()
            tgt_idx = torch.tensor([]).long()
        return batch_idx, tgt_idx
  
    def get_relation_loss(self, outputs, targets, indices, num_relation_boxes):
        # TODO use_focal_loss
        # use_focal_loss = False
        src_logits = outputs['relation_logits']
        idx = self._get_src_permutation_idx_rel(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_rel_weight)
        losses = {'loss_relation': loss_ce}

        if len(idx[0]) > 0:
            src_boxes = outputs['relation_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

            losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou_relation'] = loss_giou.sum() / num_relation_boxes
        else:
            losses['loss_bbox_relation'] = (outputs['relation_boxes'] * 0.0).sum()
            losses['loss_giou_relation'] = (outputs['relation_boxes'] * 0.0).sum()
        
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_relation_losses(self, relation_outputs, entity_targets, relation_targets, combined_indices, num_group, **kwargs):
        losses = {}
        # TODO Group DETR
        # num_subject_boxes = sum(len(t[1]) for t in combined_indices['subject']) * num_group
        num_subject_boxes = sum(len(t[1]) for t in combined_indices['subject']) 
        num_subject_boxes = torch.as_tensor([num_subject_boxes], dtype=torch.float, device=next(iter(relation_outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_subject_boxes)
        num_subject_boxes = torch.clamp(num_subject_boxes / get_world_size(), min=1).item()
        relation_subject_outputs = {'pred_boxes': relation_outputs['relation_subject_boxes'], 'pred_logits': relation_outputs['relation_subject_logits']}
        for loss in self.losses:
            if loss == 'labels':
                subject_losses = self.get_loss(loss, relation_subject_outputs, entity_targets, combined_indices['subject'], num_subject_boxes, **kwargs)
            else:
                subject_losses = self.get_loss(loss, relation_subject_outputs, entity_targets, combined_indices['subject'], num_subject_boxes)
            subject_losses = {k + f'_subject': v for k, v in subject_losses.items()}
            losses.update(subject_losses)

        # TODO Group DETR
        # num_object_boxes = sum(len(t[1]) for t in combined_indices['object']) * num_group
        num_object_boxes = sum(len(t[1]) for t in combined_indices['object'])
        num_object_boxes = torch.as_tensor([num_object_boxes], dtype=torch.float, device=next(iter(relation_outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_object_boxes)
        num_object_boxes = torch.clamp(num_object_boxes / get_world_size(), min=1).item()
        relation_object_outputs = {'pred_boxes': relation_outputs['relation_object_boxes'], 'pred_logits': relation_outputs['relation_object_logits']}
        for loss in self.losses:
            if loss == 'labels':
                object_losses = self.get_loss(loss, relation_object_outputs, entity_targets, combined_indices['object'], num_object_boxes, **kwargs)
            else:
                object_losses = self.get_loss(loss, relation_object_outputs, entity_targets, combined_indices['object'], num_object_boxes)
            object_losses = {k + f'_object': v for k, v in object_losses.items()}
            losses.update(object_losses)
        
        # TODO Group DETR
        # num_relation_boxes = sum(len(t[1]) for t in combined_indices['relation']) * num_group
        num_relation_boxes = sum(len(t[1]) for t in combined_indices['relation'])
        num_relation_boxes = torch.as_tensor([num_relation_boxes], dtype=torch.float, device=next(iter(relation_outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_relation_boxes)
        num_relation_boxes = torch.clamp(num_relation_boxes / get_world_size(), min=1).item()
        losses.update(self.get_relation_loss(relation_outputs, relation_targets, combined_indices['relation'], num_relation_boxes))
        return losses
    # TODO Group DETR
    def forward(self, outputs, targets, num_group):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Detection Branch
        device = next(iter(outputs.values())).device
        losses = {}

        # Relation  Branch
        relation_outputs_without_aux = {k: v for k, v in outputs.items() if 'aux_outputs' not in k and 'relation' in k}
        combined_indices = self.matcher.forward_relation(relation_outputs_without_aux, targets)
        
        # Losses
        entity_targets = [{'boxes': x['combined_boxes'], 'labels': x['combined_labels']} for x in targets]
        relation_targets = [{'boxes': x['relation_boxes'], 'labels': x['relation_labels']} for x in targets]
        # TODO Group DETR
        losses.update(self.get_relation_losses(relation_outputs_without_aux, entity_targets, relation_targets, combined_indices, num_group))
        if 'aux_outputs_r' in outputs:
            for i, (aux_outputs_r, aux_outputs_r_sub, aux_outputs_r_obj) in enumerate(zip(outputs['aux_outputs_r'], outputs['aux_outputs_r_sub'], outputs['aux_outputs_r_obj'])):
                relation_aux_outputs = {'relation_logits': aux_outputs_r['pred_logits'], 'relation_boxes': aux_outputs_r['pred_boxes'],
                                        'relation_subject_logits': aux_outputs_r_sub['pred_logits'], 'relation_subject_boxes': aux_outputs_r_sub['pred_boxes'],
                                        'relation_object_logits': aux_outputs_r_obj['pred_logits'], 'relation_object_boxes': aux_outputs_r_obj['pred_boxes']}
                aux_combined_indices = self.matcher.forward_relation(relation_aux_outputs, targets)
                kwargs = {'log': False}
                l_dict = self.get_relation_losses(relation_aux_outputs, entity_targets, relation_targets, aux_combined_indices, num_group, **kwargs)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


@CRITERION_REGISTRY.register()
class IterativeRelationCriterion(IterativeRelationCriterionBase):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses, use_gt_box=use_gt_box, use_gt_label=use_gt_label, **kwargs)
        del self.empty_rel_weight
        self.oversample_param = kwargs['oversample_param']
        self.undersample_param = kwargs['undersample_param']

        if self.reweight_rel:
            empty_rel_weight = self.statistics['fg_rel_count'] / self.statistics['fg_rel_count'].sum() 
            empty_rel_weight = torch.maximum((self.oversample_param / (empty_rel_weight + 1e-5)).sqrt(), torch.ones_like(empty_rel_weight))
            empty_rel_weight = torch.pow(empty_rel_weight, self.undersample_param)
            empty_rel_weight[-1] = kwargs['reweight_rel_eos_coef']
        else:
            empty_rel_weight = torch.ones(self.num_rel_classes + 1)
            empty_rel_weight[-1] = kwargs['reweight_rel_eos_coef']
        self.register_buffer('empty_rel_weight', empty_rel_weight)
        print ("SCALED", self.empty_rel_weight)

    def get_relation_loss(self, outputs, targets, indices, num_relation_boxes):
        src_logits = outputs['relation_logits']
        idx = self._get_src_permutation_idx_rel(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_rel_weight)
        losses = {'loss_relation': loss_ce}

        if len(idx[0]) > 0:
            src_boxes = outputs['relation_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

            losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou_relation'] = loss_giou.sum() / num_relation_boxes
        else:
            losses['loss_bbox_relation'] = (outputs['relation_boxes'] * 0.0).sum()
            losses['loss_giou_relation'] = (outputs['relation_boxes'] * 0.0).sum()
        
        return losses
        
    def forward(self, outputs, targets, num_group):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Detection Branch
        device = next(iter(outputs.values())).device
        losses = {}
        
        # Relation  Branch
        relation_outputs_without_aux = {k: v for k, v in outputs.items() if 'aux_outputs' not in k and 'relation' in k}
        combined_indices = self.matcher.forward_relation(outputs, targets)
        
        # Losses
        entity_targets = [{'boxes': x['combined_boxes'], 'labels': x['combined_labels']} for x in targets]
        relation_targets = [{'boxes': x['relation_boxes'], 'labels': x['relation_labels']} for x in targets]
        # TODO Group DETR
        losses.update(self.get_relation_losses(relation_outputs_without_aux, entity_targets, relation_targets, combined_indices, num_group))
        if 'aux_outputs_r' in outputs:
            for i, (aux_outputs_r, aux_outputs_r_sub, aux_outputs_r_obj) in enumerate(zip(outputs['aux_outputs_r'], outputs['aux_outputs_r_sub'], outputs['aux_outputs_r_obj'])):
                relation_aux_outputs = {'relation_logits': aux_outputs_r['pred_logits'], 'relation_boxes': aux_outputs_r['pred_boxes'],
                                        'relation_subject_logits': aux_outputs_r_sub['pred_logits'], 'relation_subject_boxes': aux_outputs_r_sub['pred_boxes'],
                                        'relation_object_logits': aux_outputs_r_obj['pred_logits'], 'relation_object_boxes': aux_outputs_r_obj['pred_boxes']}
                kwargs = {'log': False}
                # TODO Group DETR
                l_dict = self.get_relation_losses(relation_aux_outputs, entity_targets, relation_targets, combined_indices, num_group, **kwargs)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
    
@CRITERION_REGISTRY.register()
class FuseTripletCriterion(IterativeRelationCriterionBase):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses, use_gt_box=use_gt_box, use_gt_label=use_gt_label, **kwargs)
        del self.empty_rel_weight
        self.oversample_param = kwargs['oversample_param']
        self.undersample_param = kwargs['undersample_param']

        if self.reweight_rel:
            empty_rel_weight = self.statistics['fg_rel_count'] / self.statistics['fg_rel_count'].sum() 
            empty_rel_weight = torch.maximum((self.oversample_param / (empty_rel_weight + 1e-5)).sqrt(), torch.ones_like(empty_rel_weight))
            empty_rel_weight = torch.pow(empty_rel_weight, self.undersample_param)
            empty_rel_weight[-1] = kwargs['reweight_rel_eos_coef']
        else:
            empty_rel_weight = torch.ones(self.num_rel_classes + 1)
            empty_rel_weight[-1] = kwargs['reweight_rel_eos_coef']
        self.register_buffer('empty_rel_weight', empty_rel_weight)
        print ("SCALED", self.empty_rel_weight)

    def get_relation_loss(self, outputs, targets, indices, num_relation_boxes):
        src_logits = outputs['relation_logits']
        idx = self._get_src_permutation_idx_rel(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_rel_weight)
        losses = {'loss_relation': loss_ce}

        if len(idx[0]) > 0:
            src_boxes = outputs['relation_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

            losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou_relation'] = loss_giou.sum() / num_relation_boxes
        else:
            losses['loss_bbox_relation'] = (outputs['relation_boxes'] * 0.0).sum()
            losses['loss_giou_relation'] = (outputs['relation_boxes'] * 0.0).sum()
        
        return losses

    def del_tensor(arr, idx):
        arr1 = arr[0:idx]
        arr2 = arr[idx+1:]
        return torch.cat([arr1, arr2], dim=0)
        
    def forward(self, outputs, targets, num_group):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Detection Branch
        device = next(iter(outputs.values())).device
        losses = {}
        
        # Relation  Branch
        relation_outputs_without_aux = {k: v for k, v in outputs.items() if 'aux_outputs' not in k and 'relation' in k}
        aux_layers = len(outputs['aux_outputs_r'])
        combined_indices = self.matcher.forward_relation(outputs, targets) # 3 group

        bs, num_queries = outputs["relation_logits"].shape[:2]
        relation_assign = torch.full((bs, num_queries), -1)
        for bs, relation_idx  in enumerate(combined_indices['relation']):
            rel_nq_list, rel_tgt_list = relation_idx
            for i, (nq_id, rel_tgt_id) in enumerate(zip(rel_nq_list, rel_tgt_list)):
                relation_assign[bs][nq_id] = rel_tgt_id
                
        # print(f'relation_assign top: {relation_assign}')
        # Losses
        entity_targets = [{'boxes': x['combined_boxes'], 'labels': x['combined_labels']} for x in targets]
        relation_targets = [{'boxes': x['relation_boxes'], 'labels': x['relation_labels']} for x in targets]
        # TODO Group DETR
        losses.update(self.get_relation_losses(relation_outputs_without_aux, entity_targets, relation_targets, combined_indices, num_group))
        if 'aux_outputs_r' in outputs:
            # aux_layers = len(outputs['aux_outputs_r'])
            stage_combined_indices = combined_indices
            for i, (aux_outputs_r, aux_outputs_r_sub, aux_outputs_r_obj) in enumerate(zip(reversed(outputs['aux_outputs_r']), reversed(outputs['aux_outputs_r_sub']), reversed((outputs['aux_outputs_r_obj'])))):
                aux_id = aux_layers - i - 1
                relation_aux_outputs = {'relation_logits': aux_outputs_r['pred_logits'], 'relation_boxes': aux_outputs_r['pred_boxes'],
                                       'relation_subject_logits': aux_outputs_r_sub['pred_logits'], 'relation_subject_boxes': aux_outputs_r_sub['pred_boxes'],
                                       'relation_object_logits': aux_outputs_r_obj['pred_logits'], 'relation_object_boxes': aux_outputs_r_obj['pred_boxes']}
                kwargs = {'log': False}
                # TODO Group DETR
                aux_combined_indices = self.matcher.forward_aux_relation(outputs, targets, aux_id)
                        
                # for bs, (relation_idx, subject_idx, object_idx)  in enumerate(zip(aux_combined_indices['relation'], aux_combined_indices['subject'], aux_combined_indices['object'])):
                for bs, (relation_idx, subject_idx, object_idx)  in enumerate(zip(aux_combined_indices['relation'], aux_combined_indices['subject'], aux_combined_indices['object'])):
                    rel_nq_list, rel_tgt_list = relation_idx
                    _, sub_tgt_list = subject_idx
                    _, obj_tgt_list = object_idx
                    for idx, (nq_id, rel_tgt_id, sub_tgt_id, obj_tgt_id) in enumerate(zip(rel_nq_list, rel_tgt_list, sub_tgt_list, obj_tgt_list)):
                        if relation_assign[bs][nq_id] == -1:
                            relation_assign[bs][nq_id] = rel_tgt_id
                            rel_q = torch.cat([stage_combined_indices['relation'][bs][0],nq_id.unsqueeze(0)], dim=0)
                            sub_q = torch.cat([stage_combined_indices['subject'][bs][0],nq_id.unsqueeze(0)], dim=0)
                            obj_q = torch.cat([stage_combined_indices['object'][bs][0],nq_id.unsqueeze(0)], dim=0)
                            
                            rel_tgt = torch.cat([stage_combined_indices['relation'][bs][1],rel_tgt_id.unsqueeze(0)], dim=0)
                            sub_tgt = torch.cat([stage_combined_indices['subject'][bs][1],sub_tgt_id.unsqueeze(0)], dim=0)
                            obj_tgt = torch.cat([stage_combined_indices['object'][bs][1],obj_tgt_id.unsqueeze(0)], dim=0) 
                            
                            stage_combined_indices['relation'][bs] = (rel_q, rel_tgt)
                            stage_combined_indices['subject'][bs] = (sub_q, sub_tgt)
                            stage_combined_indices['object'][bs] = (obj_q, obj_tgt)
                l_dict = self.get_relation_losses(relation_aux_outputs, entity_targets, relation_targets, stage_combined_indices, num_group, **kwargs)
                l_dict = {k + f'_{aux_id}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

def build_criterion(name, num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=False, use_gt_label=False, **kwargs):
    return CRITERION_REGISTRY.get(name)(num_classes, matcher, weight_dict, eos_coef, losses, use_gt_box=use_gt_box, use_gt_label=use_gt_label, **kwargs)
