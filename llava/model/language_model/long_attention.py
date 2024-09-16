import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import numpy as np
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2FlashAttention2
from transformers.models.llama.modeling_llama import LlamaFlashAttention2
# from .modeling_llama import LlamaFlashAttention2
from .long_attention_utils import *
from .long_attention_utils import _make_causal_mask
import warnings
import pdb
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

class MsPoERotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads, 1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)

        t = t / compress_ratio
        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:,:seq_len].to(dtype=x.dtype),
            self.sin_cached[:,:seq_len].to(dtype=x.dtype),
        )


class SelfExtendFlashAttention(Qwen2FlashAttention2):
# class SelfExtendFlashAttention(LlamaFlashAttention2):
    '''
    self_extend_attention_forward = partial(SE.Qwen2.flash_self_extend_forward,
                                        group_size_1=group_size, 
                                        group_size_2=window_size,
                                        scale_base=scale_base)
    modifed_1 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "_flash_attention_forward", SE.selfextend_flash_attn.flash_attention2_forward_with_window_size)
    modifed_2 = modify_method_of_instance(loaded_model, "Qwen2FlashAttention2", "forward", self_extend_attention_forward)
    '''
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.group_size_1 = getattr(config, "group_size", 8)
        self.group_size_2 = getattr(config, "window_size", 2048)
        self.scale_base = getattr(config, "scale_base", -1)

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        # group_size_1: Optional[float] = 8,
        # group_size_2: Optional[float] = 2048,
        # scale_base: Optional[float] = -1,
        **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if self.scale_base > 0:
            scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(self.scale_base)).clip(1).to(query_states.dtype) # log scale 
            #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
        else:
            scaled_query = query_states

        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        query_position = position_ids
        key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) # only support batch=1 for now.


        attn_dropout = self.config.attention_dropout if self.training else 0.0
        if q_len == 1:
            _re_group_size_2 = 0 if position_ids.max() < self.group_size_2 else self.group_size_2
            neighbor_key_position = position_ids[:, -1] - key_position
            group_key_position = position_ids[:, -1]//self.group_size_1 - key_position//self.group_size_1 + (_re_group_size_2 - _re_group_size_2//self.group_size_1)
            decode_key_position = torch.cat([group_key_position[:, :-self.group_size_2], neighbor_key_position[:,-self.group_size_2:]], dim=1)
            
            # import pdb; pdb.set_trace()
            #neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position_ids) 
            decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
            _, decode_key_states = apply_rotary_pos_emb(None, key_states, cos, -sin, decode_key_position) 

            decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            
            attn_output = flash_attn_func(decode_query_states,
                                        decode_key_states,
                                        decode_value_states,
                                        attn_dropout, 
                                        softmax_scale=None, 
                                        causal=True)
        
        elif q_len == kv_seq_len:
            # set correct position_ids & apply RoPE.
            _re_group_size_2 = 0 if query_position.max() < self.group_size_2 else self.group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position

            neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position) 
            _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, cos, sin, key_position) 

            group_query_states, _ = apply_grouped_rotary_pos_emb(scaled_query, None, cos, sin, query_position, g_size_1=self.group_size_1, g_size_2=_re_group_size_2) 
            _, group_key_states = apply_grouped_rotary_pos_emb(None, key_states, cos, sin, key_position, g_size_1=self.group_size_1, g_size_2=_re_group_size_2) 


            neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
            neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            group_query_states = group_query_states.transpose(1, 2).contiguous()
            group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

            attn_output = self_extend_flash_forward(self,
                                                    query_position,
                                                    self.group_size_2,
                                                    neighbor_query_states,
                                                    neighbor_key_states,
                                                    group_query_states,
                                                    group_key_states,
                                                    value_states,
                                                    attention_mask,
                                                    bsz,
                                                    q_len,
                                                    kv_seq_len,
                                                    attn_dropout,
                                                )
        else:
            raise ValueError("q_len should be 1 or seq_len.")

        attn_output = attn_output.contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        window_size=[-1, -1],
        return_attn_probs=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            window_size ([Int, Int])
                The left & right window size for Flash Attention. Default to [-1, -1] which means no window size is used.
            return_attn_probs (`bool`, *optional*):
                Whether to return the attention softmax logssumexp and probabilities. Default to False.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad, softmax_lse, S_dmask = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                return_attn_probs=True,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output, softmax_lse, S_dmask = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                return_attn_probs=True,
            )

        if return_attn_probs:
            return attn_output, softmax_lse, S_dmask
        else:
            return attn_output


class SelfExtendAttention(Qwen2Attention):
    """
    replace forward by self_extend_attention_forward in 
    https://github.com/datamllab/LongLM/blob/master/self_extend_patch/Qwen2.py
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        group_size_1: Optional[float] = 8,
        group_size_2: Optional[float] = 2048,
        scale_base: Optional[float] = -1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if scale_base > 0:
            scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
            #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
        else:
            scaled_query = query_states
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        query_position = position_ids
        # only consider bsz=1 for now. 
        key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)


        neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position) 
        _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, cos, sin, key_position) 
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_states, _ = apply_grouped_rotary_pos_emb(scaled_query, None, cos, sin, query_position, g_size_1=group_size_1, g_size_2=_re_group_size_2) 
        _, group_key_states = apply_grouped_rotary_pos_emb(None, key_states, cos, sin, key_position, g_size_1=group_size_1, g_size_2=_re_group_size_2) 


        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 


        if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {group_attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            group_attn_weights = group_attn_weights + attention_mask
            neighbor_attn_weights = neighbor_attn_weights + attention_mask


        if q_len == 1:
            neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask[:, -group_size_2:] = 1
        elif q_len == kv_seq_len:
            neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask = torch.tril(neighbor_attention_mask)
            if q_len-group_size_2 > 0:
                group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
                neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask

        else:
            raise ValueError("q_len should be 1 or seq_len.")


        neighbor_attention_mask = neighbor_attention_mask.bool()
        attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MsPoEFlashAttention(Qwen2FlashAttention2):

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.compress_ratio_min = getattr(config, "compress_ratio_min", 1)
        self.compress_ratio_max = getattr(config, "compress_ratio_max", 3)
        warnings.warn(
                f"configurate compress_ratio_min {self.compress_ratio_min}, compress_ratio_max {self.compress_ratio_max}"
            )
        self.enable_head_metrics = True
        self.head_type = getattr(config, "head_type", None)
        self.head_order = None
        
        # reinitialize rotary emb
        self.rotary_emb = MsPoERotaryEmbedding(
            self.head_dim,
            min_cratio=self.compress_ratio_min,
            max_cratio=self.compress_ratio_max,
            num_heads=self.num_heads,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):
        query_states_new = query_states
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

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

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders

    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).to(attn_weights.dtype).mean(-1)[:, -1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # # remake causal mask # attention_mask is none during inference
        # past_key_values_length = past_key_value[0][0].shape[-2] - 1 if past_key_value is not None and q_len == 1 else 0
        # attention_mask = _make_causal_mask(
        #     bsz=bsz,
        #     tgt_len=q_len,
        #     past_key_values_length=past_key_values_length,
        #     dtype=query_states.dtype,
        #     device=query_states.device,
        # )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        #mspoe
        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1
        #mspoe

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        
        #mspoe
        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz,
                                                         attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]
        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)
        #mspoe

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids) #mspoe

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        warnings.warn('transpose qkv')
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask=None, query_length=q_len, dropout=dropout_rate
        )

        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        self._flash_attn_uses_top_left_mask = False
        self.is_causal = True

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output