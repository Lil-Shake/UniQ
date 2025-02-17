import copy
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou

from detectron2.utils.registry import Registry
from torchvision.ops.boxes import box_area

MATCHER_REGISTRY = Registry("MATCHER_REGISTRY")

@MATCHER_REGISTRY.register()
class HungarianMatcher_GroupMatch(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_group=1, topk=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_group = num_group
        self.matcher_topk = topk
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    # implemention of topk match
    def top_score_match(self, match_cost, return_init_idx=False):
        indices_all = []
        for cost in [c[i] for i, c in enumerate(match_cost)]:
            cost_inplace = copy.deepcopy(cost)
            # print(f'cost_inplace:{cost_inplace.device}')
            topk = self.matcher_topk
            indice_multi = []
            if cost_inplace.shape[1] == 0:
                indices = linear_sum_assignment(cost_inplace)
                indices = (indices[0], indices[1])
                indice_multi.append(indices)
            else:
                for _ in range(topk): # [0,1,...,topk]
                    # selective matching:
                    # We observe the the macthing is only happend in the 
                    # small set of predictions that have top K cost value, 
                    # to this end, we optimize the matching pool by: instead 
                    # matching with all possible prediction, we use the 
                    # top K times of GT num predictions for matching
                    # print('----------------------debug------------------')
                    # print(cost_inplace.shape)
                    # print(cost_inplace)
                    min_cost = cost_inplace.min(-1)[0]
                    selected_range = 4096 # number of GT
                    selected_range = ( 
                        selected_range
                        if selected_range < cost_inplace.shape[0]
                        else cost_inplace.shape[0]
                    )
                    _, idx = min_cost.topk(selected_range, largest=False) 
                    indices = linear_sum_assignment(cost_inplace) 
                    indices = (indices[0], indices[1]) # 
                    # if one pred match with the gt, we exclude it 
                    cost_inplace[indices[0], :] = 1e10
                    indice_multi.append(indices)

            if self.training:
                # filtering that the prediction from one query is matched with the multiple GT
                init_pred_idx = np.concatenate([each[0] for each in indice_multi])
                pred_idx = init_pred_idx

                # check the matching relationship between the query id and GT id
                gt_idx = np.concatenate([each[1] for each in indice_multi])
                dup_match_dict = dict()
                for init_idx, (p_i, g_i) in enumerate(zip(pred_idx, gt_idx)):
                    if dup_match_dict.get(p_i) is not None:
                        if cost[p_i][dup_match_dict[p_i][1]] > cost[p_i][g_i]:
                            print('*'*20)
                            print(p_i)
                            # print(cost[p_i][dup_match_dict[p_i]], cost[p_i][g_i])
                            # print(p_i, dup_match_dict[p_i], g_i)
                            dup_match_dict[p_i] = (init_idx, g_i)
                    else:
                        dup_match_dict[p_i] = (init_idx, g_i)

                init_pred_idx_sort = []
                pred_idx = []
                gt_idx = []
                for p_i, (init_idx, g_i) in dup_match_dict.items():
                    pred_idx.append(p_i)
                    gt_idx.append(g_i)
                    init_pred_idx_sort.append(init_pred_idx[init_idx])

                if return_init_idx:
                   indices_all.append((np.array(init_pred_idx_sort), np.array(gt_idx)))
                else:
                    indices_all.append((np.array(pred_idx), np.array(gt_idx)))
            else:
                indices_all.append(
                    (
                        np.concatenate([each[0] for each in indice_multi]),
                        np.concatenate([each[1] for each in indice_multi]),
                    )
            )

        return indices_all

    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # print(f'out_prob device:{out_prob.device}')
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        

        sizes = [len(v["boxes"]) for v in targets]
        
        # TODO Group Match
        if self.num_group == 1:
            C_split = C.split(sizes, -1)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
            
        elif self.num_group > 1:
            indices = []
            g_num_queries = num_queries // self.num_group
            C_list = C.split(g_num_queries, dim=1)
            for g_i in range(self.num_group):
                C_g = C_list[g_i]
                indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
                if g_i == 0:
                    indices = indices_g
                else:
                    indices = [
                        (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                        for indice1, indice2 in zip(indices, indices_g)
                    ]
        else:
            NotImplementedError
                
        # if not return_cost_matrix:
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # else:
        #     return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split

@MATCHER_REGISTRY.register()
class UniqHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_group=1, topk=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_group = num_group
        self.matcher_topk = topk
        self.iou_threshold = float(0.7)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def top_score_match(self, match_cost, return_init_idx=False):
        indices_all = []
        for cost in [c[i] for i, c in enumerate(match_cost)]:
            cost_inplace = copy.deepcopy(cost)
            # print('*'*20)
            # print(cost_inplace.device)
            topk = self.matcher_topk
            indice_multi = []
            if cost_inplace.shape[1] == 0:
                indices = linear_sum_assignment(cost_inplace)
                indices = (indices[0], indices[1])
                indice_multi.append(indices)
            else:
                for _ in range(topk): # [0,1,...,topk]
                    # selective matching:
                    # We observe the the macthing is only happend in the 
                    # small set of predictions that have top K cost value, 
                    # to this end, we optimize the matching pool by: instead 
                    # matching with all possible prediction, we use the
                    # top K times of GT num predictions for matching
                    min_cost = cost_inplace.min(-1)[0]
                    selected_range = 4096 
                    selected_range = ( 
                        selected_range
                        if selected_range < cost_inplace.shape[0]
                        else cost_inplace.shape[0]
                    )
                    _, idx = min_cost.topk(selected_range, largest=False)
                    indices = linear_sum_assignment(cost_inplace[idx, :]) 
                    indices = (idx[indices[0]], indices[1]) # 
                    # if one pred match with the gt, we exclude it
                    cost_inplace[indices[0], :] = 1e10
                    indice_multi.append(indices)
            indices_all.append(
                (
                    np.concatenate([each[0] for each in indice_multi]),
                    np.concatenate([each[1] for each in indice_multi]),
                )
            )

        return indices_all
    
    @torch.no_grad()
    def forward(self, outputs, targets, return_cost_matrix=False, mask=None):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()        
        if mask is not None:
            C[:, ~mask] = np.float("inf")

        sizes = [len(v["boxes"]) for v in targets]
        C_split = C.split(sizes, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_split)]
        
        if not return_cost_matrix:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], C_split
    
    @torch.no_grad()
    def forward_relation(self, outputs, targets, return_cost_matrix=False):
        bs, num_queries = outputs["relation_logits"].shape[:2]
        
        out_prob = outputs["relation_logits"].flatten(0, 1).softmax(-1)
        out_sub_prob = outputs["relation_subject_logits"].flatten(0, 1).softmax(-1)
        out_obj_prob = outputs["relation_object_logits"].flatten(0, 1).softmax(-1)
            
        out_bbox = outputs["relation_boxes"].flatten(0, 1)
        out_sub_bbox = outputs["relation_subject_boxes"].flatten(0, 1)
        out_obj_bbox = outputs["relation_object_boxes"].flatten(0, 1)

        aux_out_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r']]
        aux_out_sub_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_obj']]
            
        aux_out_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r']]
        aux_out_sub_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_obj']]

        device = out_prob.device

        gt_labels = [v['combined_labels'] for v in targets]
        gt_boxes = [v['combined_boxes'] for v in targets]
        relations = [v["image_relations"] for v in targets]
        relation_boxes = [v['relation_boxes'] for v in targets]
        
        if len(relations) > 0:
            tgt_ids = torch.cat(relations)[:, 2]
            tgt_sub_labels = torch.cat([gt_label[relation[:, 0]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_obj_labels = torch.cat([gt_label[relation[:, 1]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_boxes = torch.cat(relation_boxes)
            tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
            tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
        else:
            tgt_ids = torch.tensor([]).long().to(device)
            tgt_sub_labels = torch.tensor([]).long().to(device)
            tgt_obj_labels = torch.tensor([]).long().to(device)
            tgt_boxes = torch.zeros((0,4)).to(device)
            tgt_sub_boxes = torch.zeros((0,4)).to(device)
            tgt_obj_boxes = torch.zeros((0,4)).to(device)
    
        cost_class = -out_prob[:, tgt_ids]
        cost_subject_class = -out_sub_prob[:, tgt_sub_labels]
        cost_object_class = -out_obj_prob[:, tgt_obj_labels]     
        
        cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
        cost_subject_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_object_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))
        cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))

        C = self.cost_bbox * (cost_bbox + cost_subject_bbox + cost_object_bbox) + self.cost_class * (cost_class + cost_subject_class + cost_object_class) + self.cost_giou * (cost_giou + cost_subject_giou + cost_object_giou)

        for aux_idx in range(len(outputs['aux_outputs_r'])):
            aux_cost_class = -aux_out_prob[aux_idx][:, tgt_ids]
            aux_cost_subject_class = -aux_out_sub_prob[aux_idx][:, tgt_sub_labels]
            aux_cost_object_class = -aux_out_obj_prob[aux_idx][:, tgt_obj_labels]

            aux_cost_bbox = torch.cdist(aux_out_bbox[aux_idx], tgt_boxes, p=1)
            aux_cost_subject_bbox = torch.cdist(aux_out_sub_bbox[aux_idx], tgt_sub_boxes, p=1)
            aux_cost_object_bbox = torch.cdist(aux_out_obj_bbox[aux_idx], tgt_obj_boxes, p=1)

            aux_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_boxes))
            aux_cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_sub_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_sub_boxes))
            aux_cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_obj_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_obj_boxes))
            aux_C = self.cost_bbox * (aux_cost_bbox + aux_cost_subject_bbox + aux_cost_object_bbox) + self.cost_class * (aux_cost_class + aux_cost_subject_class + aux_cost_object_class) + self.cost_giou * (aux_cost_giou + aux_cost_subject_giou + aux_cost_object_giou)

            C = C + aux_C
        
        C = C.view(bs, num_queries, -1).cpu()   
        sizes = [len(v["image_relations"]) for v in targets]
        # TODO Group Match
        indices = []
        g_num_queries = num_queries // self.num_group
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(self.num_group):
            C_g = C_list[g_i]
            if self.matcher_topk == 1:
                indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            elif self.matcher_topk > 1:
                match_cost = C_g.split(sizes, -1)
                indices_g = self.top_score_match(match_cost)
            else:
                NotImplementedError
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # Remaining GT objects matching
        combined_indices = {'subject' :[], 'object': [], 'relation': []}
        for image_idx, target in enumerate(targets):
            relation = target['image_relations']
            curr_relation_idx = indices[image_idx]
            
            combined_indices['relation'].append((curr_relation_idx[0], curr_relation_idx[1]))
            for branch_idx, branch_type in enumerate(['subject', 'object']):  
                combined_indices[branch_type].append((curr_relation_idx[0], relation[:, branch_idx][curr_relation_idx[1]].cpu()))
        # return combined_indices, sub_weight, obj_weight, rel_weight
        return combined_indices
    
    @torch.no_grad()
    def forward_aux_relation(self, outputs, targets, aux_id, return_cost_matrix=False):
        # TODO use_focal_loss
        # use_focal_loss = False
        # TODO aux_id
        # print(aux_id)
        bs, num_queries = outputs["relation_logits"].shape[:2]

        aux_out_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r']]
        aux_out_sub_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_prob = [output['pred_logits'].flatten(0, 1).softmax(-1) for output in outputs['aux_outputs_r_obj']]
            
        aux_out_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r']]
        aux_out_sub_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_sub']]
        aux_out_obj_bbox = [output['pred_boxes'].flatten(0, 1) for output in outputs['aux_outputs_r_obj']]

        device = aux_out_prob[aux_id].device

        gt_labels = [v['combined_labels'] for v in targets]
        gt_boxes = [v['combined_boxes'] for v in targets]
        relations = [v["image_relations"] for v in targets]
        relation_boxes = [v['relation_boxes'] for v in targets]
        
        if len(relations) > 0:
            tgt_ids = torch.cat(relations)[:, 2]
            tgt_sub_labels = torch.cat([gt_label[relation[:, 0]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_obj_labels = torch.cat([gt_label[relation[:, 1]] for gt_label, relation in zip(gt_labels, relations)])
            tgt_boxes = torch.cat(relation_boxes)
            tgt_sub_boxes = torch.cat([gt_box[relation[:, 0]] for gt_box, relation in zip(gt_boxes, relations)])
            tgt_obj_boxes = torch.cat([gt_box[relation[:, 1]] for gt_box, relation in zip(gt_boxes, relations)])
        else:
            tgt_ids = torch.tensor([]).long().to(device)
            tgt_sub_labels = torch.tensor([]).long().to(device)
            tgt_obj_labels = torch.tensor([]).long().to(device)
            tgt_boxes = torch.zeros((0,4)).to(device)
            tgt_sub_boxes = torch.zeros((0,4)).to(device)
            tgt_obj_boxes = torch.zeros((0,4)).to(device)
        
        # print(aux_id+1)
        C = None
        for aux_idx in range(aux_id+1):
            aux_cost_class = -aux_out_prob[aux_idx][:, tgt_ids]
            aux_cost_subject_class = -aux_out_sub_prob[aux_idx][:, tgt_sub_labels]
            aux_cost_object_class = -aux_out_obj_prob[aux_idx][:, tgt_obj_labels]

            aux_cost_bbox = torch.cdist(aux_out_bbox[aux_idx], tgt_boxes, p=1)
            aux_cost_subject_bbox = torch.cdist(aux_out_sub_bbox[aux_idx], tgt_sub_boxes, p=1)
            aux_cost_object_bbox = torch.cdist(aux_out_obj_bbox[aux_idx], tgt_obj_boxes, p=1)

            aux_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_boxes))
            aux_cost_subject_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_sub_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_sub_boxes))
            aux_cost_object_giou = -generalized_box_iou(box_cxcywh_to_xyxy(aux_out_obj_bbox[aux_idx]), box_cxcywh_to_xyxy(tgt_obj_boxes))
            aux_C = self.cost_bbox * (aux_cost_bbox + aux_cost_subject_bbox + aux_cost_object_bbox) + self.cost_class * (aux_cost_class + aux_cost_subject_class + aux_cost_object_class) + self.cost_giou * (aux_cost_giou + aux_cost_subject_giou + aux_cost_object_giou)
            if C is None:
                C = aux_C
            else:
                C = C + aux_C
        
        C = C.view(bs, num_queries, -1).cpu()   
        sizes = [len(v["image_relations"]) for v in targets]
        # TODO Group Match
        indices = []
        g_num_queries = num_queries // self.num_group
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(self.num_group):
            C_g = C_list[g_i]
            if self.matcher_topk == 1:
                indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            elif self.matcher_topk > 1:
                match_cost = C_g.split(sizes, -1)
                indices_g = self.top_score_match(match_cost)
            else:
                NotImplementedError
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
        combined_indices = {'subject' :[], 'object': [], 'relation': []}
        for image_idx, target in enumerate(targets):
            relation = target['image_relations']
            curr_relation_idx = indices[image_idx]
            
            combined_indices['relation'].append((curr_relation_idx[0], curr_relation_idx[1]))
            for branch_idx, branch_type in enumerate(['subject', 'object']):  
                combined_indices[branch_type].append((curr_relation_idx[0], relation[:, branch_idx][curr_relation_idx[1]].cpu()))
        # return combined_indices, sub_weight, obj_weight, rel_weight
        return combined_indices
    
# TODO Group Match
def build_matcher(name, cost_class, cost_bbox, cost_giou, num_group, topk=1):
    return MATCHER_REGISTRY.get(name)(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou, num_group=num_group, topk=topk)