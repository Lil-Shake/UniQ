import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from detectron2.utils.registry import Registry
from .detr import MLP
from .detr import gen_sineembed_for_position
from .util.misc import inverse_sigmoid
import math
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from .attention import MultiheadAttention

TRANSFORMER_REGISTRY = Registry("TRANSFORMER_REGISTRY")

@TRANSFORMER_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                # TODO Group Match
                                                kwargs['num_group'])
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
        # TODO Group Match
        self.num_group = kwargs['num_group']

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

@TRANSFORMER_REGISTRY.register()
class UniQTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()
        relation_selfattend = False
        Triplet_selfattend = True
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer_triplet_selfattd(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,
                                            # TODO Group Match
                                            kwargs['num_group'])
        
        layer_norm = nn.LayerNorm(d_model)
        relation_layer_norm = nn.LayerNorm(d_model)       

        self.decoder = UnifiedDecoder(decoder_layer, num_decoder_layers, layer_norm, relation_layer_norm,
                                                      return_intermediate=return_intermediate_dec, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)
 
        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']
        
        # TODO Group Match
        self.num_group = kwargs['num_group']
        self._reset_parameters()

    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, subject_embed, object_embed, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        subject_query_embed = subject_embed.unsqueeze(1).repeat(1, bs, 1)
        object_query_embed = object_embed.unsqueeze(1).repeat(1, bs, 1)
        relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)
    
        # Condition on subject
        tgt_sub = torch.zeros_like(subject_query_embed)
        tgt_obj = torch.zeros_like(object_query_embed)
        tgt_rel = torch.zeros_like(relation_query_embed)
        hs_subject, hs_object, hs_relation = self.decoder(tgt_sub, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, subject_pos=subject_query_embed, object_pos=object_query_embed, relation_pos=relation_query_embed)        
        relation_subject_class = self.object_embed(hs_subject)
        relation_subject_coords = self.object_bbox_coords(hs_subject).sigmoid()
        relation_object_class = self.object_embed(hs_object)
        relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.object_bbox_coords(hs_relation).sigmoid()
        output = {
                  'relation_coords': relation_coords.transpose(1, 2),
                  'relation_logits': relation_class.transpose(1, 2),
                  'relation_subject_logits': relation_subject_class.transpose(1, 2),
                  'relation_object_logits': relation_object_class.transpose(1, 2),
                  'relation_subject_coords': relation_subject_coords.transpose(1, 2),
                  'relation_object_coords': relation_object_coords.transpose(1, 2)
                  }

        return output
    
@TRANSFORMER_REGISTRY.register()
class STA_RelationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
           
        # if fused_triplet_ultra is False:
        decoder_layer = TransformerDecoderLayer_fuse(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,
                                            # TODO Group Match
                                            kwargs['num_group'])

        layer_norm = nn.LayerNorm(d_model)      

        self.decoder = STA_TripletDecoder(decoder_layer, num_decoder_layers, layer_norm, 
                                                      return_intermediate=return_intermediate_dec, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        
        self.d_model = d_model
        self.nhead = nhead
        self.subject_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)
        self.subject_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.relation_bbox_coords = MLP(d_model, d_model, 4, 3)
 
        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']
        
        # TODO Group Match
        self.num_group = kwargs['num_group']
        self._reset_parameters()
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)
    
        # Condition on subject
        tgt_rel = torch.zeros_like(relation_query_embed)
        hs_relation = self.decoder(tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, relation_pos=relation_query_embed)        
        relation_subject_class = self.subject_embed(hs_relation)
        relation_subject_coords = self.subject_bbox_coords(hs_relation).sigmoid()
        relation_object_class = self.object_embed(hs_relation)
        relation_object_coords = self.object_bbox_coords(hs_relation).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.relation_bbox_coords(hs_relation).sigmoid()
        output = {
                  'relation_coords': relation_coords.transpose(1, 2),
                  'relation_logits': relation_class.transpose(1, 2),
                  'relation_subject_logits': relation_subject_class.transpose(1, 2),
                  'relation_object_logits': relation_object_class.transpose(1, 2),
                  'relation_subject_coords': relation_subject_coords.transpose(1, 2),
                  'relation_object_coords': relation_object_coords.transpose(1, 2)
                  }

        return output
        
        
@TRANSFORMER_REGISTRY.register()
class TTS_RelationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer_fuse(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,
                                            # TODO Group Match
                                            kwargs['num_group'])
        layer_norm = nn.LayerNorm(d_model)
        relation_layer_norm = nn.LayerNorm(d_model)       
        
        self.decoder = TTS_TripletDecoder(decoder_layer, num_decoder_layers, layer_norm, relation_layer_norm,
                                                      return_intermediate=return_intermediate_dec, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead
        # self.subject_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)
        # self.subject_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)
        # self.relation_bbox_coords = MLP(d_model, d_model, 4, 3)
 
        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']
        
        # TODO Group Match
        self.num_group = kwargs['num_group']
        self._reset_parameters()
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, subject_embed, object_embed, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        subject_query_embed = subject_embed.unsqueeze(1).repeat(1, bs, 1)
        object_query_embed = object_embed.unsqueeze(1).repeat(1, bs, 1)
        relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)
    
        # Condition on subject
        tgt_sub = torch.zeros_like(subject_query_embed)
        tgt_obj = torch.zeros_like(object_query_embed)
        tgt_rel = torch.zeros_like(relation_query_embed)
        hs_subject, hs_object, hs_relation = self.decoder(tgt_sub, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, subject_pos=subject_query_embed, object_pos=object_query_embed, relation_pos=relation_query_embed)        
        
        relation_subject_class = self.object_embed(hs_subject)
        relation_subject_coords = self.object_bbox_coords(hs_subject).sigmoid()
        relation_object_class = self.object_embed(hs_object)
        relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.object_bbox_coords(hs_relation).sigmoid()
        output = {
                  'relation_coords': relation_coords.transpose(1, 2),
                  'relation_logits': relation_class.transpose(1, 2),
                  'relation_subject_logits': relation_subject_class.transpose(1, 2),
                  'relation_object_logits': relation_object_class.transpose(1, 2),
                  'relation_subject_coords': relation_subject_coords.transpose(1, 2),
                  'relation_object_coords': relation_object_coords.transpose(1, 2)
                  }

        return output

@TRANSFORMER_REGISTRY.register()
class STS_RelationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer_fuse(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,
                                            # TODO Group Match
                                            kwargs['num_group'])
        layer_norm = nn.LayerNorm(d_model)
        relation_layer_norm = nn.LayerNorm(d_model)       
        
        self.decoder = STS_TripletDecoder(decoder_layer, num_decoder_layers, layer_norm, relation_layer_norm,
                                                      return_intermediate=return_intermediate_dec, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.d_model = d_model
        self.nhead = nhead
        self.subject_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)
        self.subject_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)
        self.relation_bbox_coords = MLP(d_model, d_model, 4, 3)
 
        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']
        
        # TODO Group Match
        self.num_group = kwargs['num_group']
        self._reset_parameters()
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, subject_embed, object_embed, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        subject_query_embed = subject_embed.unsqueeze(1).repeat(1, bs, 1)
        object_query_embed = object_embed.unsqueeze(1).repeat(1, bs, 1)
        relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)
    
        # Condition on subject
        tgt_sub = torch.zeros_like(subject_query_embed)
        tgt_obj = torch.zeros_like(object_query_embed)
        tgt_rel = torch.zeros_like(relation_query_embed)
        hs_subject, hs_object, hs_relation = self.decoder(tgt_sub, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, subject_pos=subject_query_embed, object_pos=object_query_embed, relation_pos=relation_query_embed)        
        
        relation_subject_class = self.subject_embed(hs_subject)
        relation_subject_coords = self.subject_bbox_coords(hs_subject).sigmoid()
        relation_object_class = self.object_embed(hs_object)
        relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.relation_bbox_coords(hs_relation).sigmoid()
        output = {
                  'relation_coords': relation_coords.transpose(1, 2),
                  'relation_logits': relation_class.transpose(1, 2),
                  'relation_subject_logits': relation_subject_class.transpose(1, 2),
                  'relation_object_logits': relation_object_class.transpose(1, 2),
                  'relation_subject_coords': relation_subject_coords.transpose(1, 2),
                  'relation_object_coords': relation_object_coords.transpose(1, 2)
                  }

        return output

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer_fuse(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 # TODO Group DETR
                 num_group=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        # TODO Group DETR
        self.num_group=num_group
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # TODO Group Match
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        num_queries, bs, n_model = q.shape
        if self.training:
            q = torch.cat(q.split(num_queries // self.num_group, dim=0), dim=1) # [nq*ng, bs, h_d] -> [nq, bs*ng, h_d]
            k = torch.cat(k.split(num_queries // self.num_group, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.num_group, dim=0), dim=1)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0) # [nq, bs*ng, h_d] -> [nq*ng, bs, h_d]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
class TransformerDecoderLayer_triplet_selfattd(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 # TODO Group DETR
                 num_group=1):
        super().__init__()
        # Implementation of triplet self-attention
        self.self_attn_triplet = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        # TODO Group DETR
        self.num_group=num_group
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # TODO Group Match
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        num_queries, bs, n_model = q.shape # [nq*ng, bs*3, n_model]
        q_sub, q_obj, q_rel = q.split(bs // 3, dim=1) # [nq*ng, bs*3, n_model] -> [nq*ng, bs, n_model]
        v_sub, v_obj, v_rel = v.split(bs // 3, dim=1)

        q_sub = torch.cat(q_sub.split(1, dim=0), dim=1) # [nq*ng, bs, n_model] -> [1, bs*nq*ng, n_model]
        q_obj = torch.cat(q_obj.split(1, dim=0), dim=1)
        q_rel = torch.cat(q_rel.split(1, dim=0), dim=1)
        v_sub = torch.cat(v_sub.split(1, dim=0), dim=1)
        v_obj = torch.cat(v_obj.split(1, dim=0), dim=1)
        v_rel = torch.cat(v_rel.split(1, dim=0), dim=1)
        
        q = torch.cat([q_sub, q_obj, q_rel], dim=0) # [1, bs*nq*ng, n_model] -> [3, bs*nq*ng, n_model]
        v = torch.cat([v_sub, v_obj, v_rel], dim=0)
        k = q
        
        tgt2 = self.self_attn_triplet(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt2_sub, tgt2_obj, tgt2_rel = tgt2.split(1, dim=0) # [3, bs*nq*ng, n_model] -> [1, bs*nq*ng, n_model]

        tgt2_sub = torch.cat(tgt2_sub.split(bs // 3, dim=1), dim=0) # [1, bs*nq*ng, n_model] -> [nq*ng, bs, d_model]
        tgt2_obj = torch.cat(tgt2_obj.split(bs // 3, dim=1), dim=0)
        tgt2_rel = torch.cat(tgt2_rel.split(bs // 3, dim=1), dim=0)

        tgt2 = torch.cat([tgt2_sub, tgt2_obj, tgt2_rel], dim=1) # [nq*ng, bs, d_model] -> [nq*ng, bs*3, d_model]
        # print(f'tgt2:{tgt2.shape}')
        tgt = tgt + self.dropout0(tgt2)
        tgt = self.norm0(tgt)
        
        # TODO seperate self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.num_group, dim=0), dim=1) # [nq*ng, bs*3, h_d] -> [nq, bs*ng*3, h_d]
            k = torch.cat(k.split(num_queries // self.num_group, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.num_group, dim=0), dim=1)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0) # [nq, bs*ng, h_d] -> [nq*ng, bs, h_d]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        memory = torch.cat([memory, memory, memory], dim=1)
        pos = torch.cat([pos, pos, pos], dim=1)
        memory_key_padding_mask = torch.cat([memory_key_padding_mask, memory_key_padding_mask, memory_key_padding_mask], dim=0)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    
# Inplemention of unified decoder
class UnifiedDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, relation_norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.subject_norm = norm
        self.relation_norm = relation_norm
        self.return_intermediate = return_intermediate       
        
        self.fused_relation_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_subject_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_object_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_relation_norm = nn.LayerNorm(d_model) 
        self.fused_subject_norm = nn.LayerNorm(d_model) 
        self.fused_object_norm = nn.LayerNorm(d_model)  

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_sub, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                subject_pos: Optional[Tensor] = None,
                object_pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_subject = tgt_sub
        output_object = tgt_obj
        output_relation = tgt_rel
        
        nq, bs, hdim = output_subject.shape
        
        intermediate_relation = []
        intermediate_subject = []
        intermediate_object = []
        for layer_id in range(self.num_layers):
            
            # fuse visual part
            prefuse_triplet = torch.cat([output_subject, output_object, output_relation], -1)
            
            # Inplemention of Task-specific Queries
            fused_output_subject = self.fused_subject_fc(prefuse_triplet) # [900, 3, 256]
            output_subject = output_subject + self.fused_subject_norm(fused_output_subject)
            
            fused_output_object = self.fused_object_fc(prefuse_triplet)
            output_object = output_object + self.fused_object_norm(fused_output_object)
            
            fused_output_relation = self.fused_relation_fc(prefuse_triplet)
            output_relation = output_relation + self.fused_relation_norm(fused_output_relation)
            
            output_triplet = torch.cat([output_subject, output_object, output_relation], dim=1)
            triplet_pos = torch.cat([subject_pos, object_pos, relation_pos], dim=1)
            
            # Input task-specific queries into unified decoder
            output_triplet = self.relation_layers[layer_id](output_triplet, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=triplet_pos)             
            
            output_subject, output_object, output_relation = output_triplet.split(bs, dim=1)
            subject_pos, object_pos, relation_pos = triplet_pos.split(bs, dim=1)
            
            if self.return_intermediate:
                intermediate_subject.append(self.subject_norm(output_subject))
                intermediate_object.append(self.subject_norm(output_object))
                intermediate_relation.append(self.relation_norm(output_relation))                

        if self.subject_norm is not None:
            output_subject = self.subject_norm(output_subject)
            output_object = self.subject_norm(output_object)
            output_relation = self.relation_norm(output_relation)
            if self.return_intermediate:
                intermediate_subject.pop()
                intermediate_subject.append(output_subject)
                intermediate_object.pop()
                intermediate_object.append(output_object)
                intermediate_relation.pop()
                intermediate_relation.append(output_relation)

        if self.return_intermediate:
            return torch.stack(intermediate_subject), torch.stack(intermediate_object), torch.stack(intermediate_relation)
        
        return output_subject.unsqueeze(0), output_object.unsqueeze(0), output_relation.unsqueeze(0)
    
class STA_TripletDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate       

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_relation = tgt_rel
        
        nq, bs, hdim = output_relation.shape
        
        intermediate_relation = []
        
        for layer_id in range(self.num_layers):
            
            output_relation = self.relation_layers[layer_id](output_relation, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=relation_pos)             
            
            if self.return_intermediate:
                intermediate_relation.append(self.norm(output_relation))                

        if self.norm is not None:
            output_relation = self.norm(output_relation)
            if self.return_intermediate:
                intermediate_relation.pop()
                intermediate_relation.append(output_relation)

        if self.return_intermediate:
            return torch.stack(intermediate_relation)
        
        return output_relation.unsqueeze(0)
    
class TTS_TripletDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, relation_norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.subject_layers = _get_clones(decoder_layer, num_layers)
        self.object_layers = _get_clones(decoder_layer, num_layers)
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.subject_norm = norm
        self.relation_norm = relation_norm
        self.return_intermediate = return_intermediate       
        
        self.fused_relation_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_subject_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_object_fc = MLP(d_model*3, d_model*3, d_model, 3)
        self.fused_relation_norm = nn.LayerNorm(d_model) 
        self.fused_subject_norm = nn.LayerNorm(d_model) 
        self.fused_object_norm = nn.LayerNorm(d_model)  

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_sub, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                subject_pos: Optional[Tensor] = None,
                object_pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_subject = tgt_sub
        output_object = tgt_obj
        output_relation = tgt_rel
        
        nq, bs, hdim = output_subject.shape
        
        intermediate_relation = []
        intermediate_subject = []
        intermediate_object = []
        
        for layer_id in range(self.num_layers):
            
            prefuse_triplet = torch.cat([output_subject, output_object, output_relation], -1)
            
            fused_output_subject = self.fused_subject_fc(prefuse_triplet) # [900, 3, 256]
            output_subject = output_subject + self.fused_subject_norm(fused_output_subject)
            
            fused_output_object = self.fused_object_fc(prefuse_triplet)
            output_object = output_object + self.fused_object_norm(fused_output_object)
            
            fused_output_relation = self.fused_relation_fc(prefuse_triplet)
            output_relation = output_relation + self.fused_relation_norm(fused_output_relation)
            
            output_relation = self.relation_layers[layer_id](output_relation, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=relation_pos)             
            
            output_subject = self.subject_layers[layer_id](output_subject, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=subject_pos)     
            
            output_object = self.object_layers[layer_id](output_object, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=object_pos)     
            
            if self.return_intermediate:
                intermediate_subject.append(self.subject_norm(output_subject))
                intermediate_object.append(self.subject_norm(output_object))
                intermediate_relation.append(self.relation_norm(output_relation))                

        if self.subject_norm is not None:
            output_subject = self.subject_norm(output_subject)
            output_object = self.subject_norm(output_object)
            output_relation = self.relation_norm(output_relation)
            if self.return_intermediate:
                intermediate_subject.pop()
                intermediate_subject.append(output_subject)
                intermediate_object.pop()
                intermediate_object.append(output_object)
                intermediate_relation.pop()
                intermediate_relation.append(output_relation)

        if self.return_intermediate:
            return torch.stack(intermediate_subject), torch.stack(intermediate_object), torch.stack(intermediate_relation)
        
        return output_subject.unsqueeze(0), output_object.unsqueeze(0), output_relation.unsqueeze(0)

class STS_TripletDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, relation_norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.subject_norm = norm
        self.relation_norm = relation_norm
        self.return_intermediate = return_intermediate       

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt_sub, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                subject_pos: Optional[Tensor] = None,
                object_pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_subject = tgt_sub
        output_object = tgt_obj
        output_relation = tgt_rel
        
        nq, bs, hdim = output_subject.shape
        
        intermediate_relation = []
        intermediate_subject = []
        intermediate_object = []
        
        memory = torch.cat([memory, memory, memory], dim=1)
        pos = torch.cat([pos, pos, pos], dim=1)
        memory_key_padding_mask = torch.cat([memory_key_padding_mask, memory_key_padding_mask, memory_key_padding_mask], dim=0)
        
        for layer_id in range(self.num_layers):
            
            output_triplet = torch.cat([output_subject, output_object, output_relation], dim=1)
            triplet_pos = torch.cat([subject_pos, object_pos, relation_pos], dim=1)
            
            output_triplet = self.relation_layers[layer_id](output_triplet, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=triplet_pos)     
            
            output_subject, output_object, output_relation = output_triplet.split(bs, dim=1)
            subject_pos, object_pos, relation_pos = triplet_pos.split(bs, dim=1)
            
            if self.return_intermediate:
                intermediate_subject.append(self.subject_norm(output_subject))
                intermediate_object.append(self.subject_norm(output_object))
                intermediate_relation.append(self.relation_norm(output_relation))                

        if self.subject_norm is not None:
            output_subject = self.subject_norm(output_subject)
            output_object = self.subject_norm(output_object)
            output_relation = self.relation_norm(output_relation)
            if self.return_intermediate:
                intermediate_subject.pop()
                intermediate_subject.append(output_subject)
                intermediate_object.pop()
                intermediate_object.append(output_object)
                intermediate_relation.pop()
                intermediate_relation.append(output_relation)

        if self.return_intermediate:
            return torch.stack(intermediate_subject), torch.stack(intermediate_object), torch.stack(intermediate_relation)
        
        return output_subject.unsqueeze(0), output_object.unsqueeze(0), output_relation.unsqueeze(0)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(name, d_model, dropout, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, normalize_before, return_intermediate_dec, **kwargs):
    return TRANSFORMER_REGISTRY.get(name)(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        normalize_before=normalize_before,
        return_intermediate_dec=return_intermediate_dec,
        **kwargs
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")