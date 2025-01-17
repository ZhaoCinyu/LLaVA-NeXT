import os
import torch
import copy
import warnings
import math
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from .augment_index import aug_smollm
from transformers.utils import logging
import torch.distributed as dist
logger = logging.get_logger(__name__)

LAYER_NUM = int(os.environ.get("LAYER_NUM", 0))
HEAD_NUM = int(os.environ.get("HEAD_NUM", 0))
BETA = float(os.environ.get("BETA", 0))
THRES = float(os.environ.get("THRES", 0))
SAVE_ATTN_PATH = os.environ.get("SAVE_ATTN_PATH", None)

def atten_process_eval(attention_map, index=None):
    # layerwise attention eval
    logger.warning_once('i am evaluating calibrated attention')
    beta = BETA
    threshod = THRES
    
    head_indices = aug_smollm[index]
    print(head_indices)
    # if index==0:

    if head_indices == []:
        return attention_map
    
    is_first_fwd = True

    if SAVE_ATTN_PATH and not os.path.exists(SAVE_ATTN_PATH):
        os.makedirs(SAVE_ATTN_PATH)
    save_path = f'{SAVE_ATTN_PATH}/layer_{index}.pt'
    if os.path.exists(save_path):
        is_first_fwd = False
    
    if SAVE_ATTN_PATH and is_first_fwd:
        attn = {'before': attention_map.cpu().detach().numpy()}

    for head_num in head_indices:
        
        modified_head = attention_map[0][head_num]
        device = modified_head.device

        shape = modified_head.shape[0]
        try:
            indices = torch.nonzero(torch.where(torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device) > threshod, 1, 0))
        except:
            torch.save(modified_head, 'modified_head.pt')
            print('error')
            exit(-1)
        indices = indices[1:]
        copied_attention_map = copy.deepcopy(modified_head.detach())
        available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
        modified_head[:, indices] *= beta
        copied_attention_map[1:, indices] *= 0
        ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
        modified_head = modified_head + available_weights * ratios
        modified_head[0, 0] = 1

        attention_map[0][head_num] = modified_head
    
    if SAVE_ATTN_PATH and is_first_fwd:
        logger.warning('i am saving attention')
        attn['after'] = attention_map.cpu().detach().numpy()
        
        torch.save([attn], save_path)

    return attention_map

def atten_process_cal(attention_map):
    logger.warning_once('i am calibrating attention')
    beta = BETA
    threshod = THRES
    # attention_map: softmaxed attention score, len=1
    modified_head = attention_map[0][HEAD_NUM]
    device = modified_head.device
    
    if SAVE_ATTN_PATH:
        attn = {'before': modified_head.cpu().detach().numpy()}
    shape = modified_head.shape[0]
    # modified_head: tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        # [0.2524, 0.7476, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        # [0.2637, 0.6631, 0.0732,  ..., 0.0000, 0.0000, 0.0000],
    # torch.sum(modified_head, dim=1) = 1.*seq
    # filter the token position with high attn score
    indices = torch.nonzero(torch.where(torch.sum(modified_head, dim=0) / torch.arange(shape,0,-1).to(device) > threshod, 1, 0))
    # print(indices.shape, HEAD_NUM)
    indices = indices[1:]
    copied_attention_map = copy.deepcopy(modified_head.detach())
    # available_weights is reduced attention value sum after calibration
    available_weights = modified_head[:, indices].sum(dim=1) * (1-beta)
    modified_head[:, indices] *= beta
    copied_attention_map[1:, indices] *= 0
    # ratio of non-calibrated tokens attn score to the sum of them
    ratios = copied_attention_map / torch.sum(copied_attention_map,  dim=1, keepdim=True).to(copied_attention_map.dtype)
    modified_head = modified_head + available_weights * ratios
    modified_head[0, 0] = 1

    if SAVE_ATTN_PATH:
        attn['after'] = modified_head.cpu().detach().numpy()
        attn['indices'] = indices.cpu().detach().numpy()
        save_path = f'{SAVE_ATTN_PATH}/{LAYER_NUM}_{HEAD_NUM}.pt'
        
        if os.path.exists(save_path):
            existing_attn = torch.load(save_path)
            existing_attn.append(attn)
            torch.save(existing_attn, save_path)
        else:
            torch.save([attn], save_path)

    attention_map[0][HEAD_NUM] = modified_head

    return attention_map


def atten_aug_forward_eval_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        signal: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        
        logger.warning_once('i am in new llama attention (eval mode)')
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        ##############BEGIN:MODIFICATION##############
        if signal in range(30):
            attn_weights = atten_process_eval(attn_weights, index=signal)
        ##############END:MODIFICATION##############

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value

def atten_aug_forward_cal_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        signal: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        logger.warning_once('i am in new llama attention: cal mode')
        # exit(-1)
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        ##############BEGIN:MODIFICATION##############
        if signal:
            attn_weights = atten_process_cal(attn_weights)
        ##############END:MODIFICATION##############

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


