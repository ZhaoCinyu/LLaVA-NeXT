import torch
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
"""
se utils
"""

# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos[:,:, -q.shape[2]:]) + (rotate_half(q) * sin[:,:, -q.shape[2]:]) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

def apply_grouped_rotary_pos_emb(q, k, cos, sin, position_ids, g_size_1=1, g_size_2=4096):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids_q = position_ids//g_size_1 + g_size_2 - g_size_2//g_size_1
    position_ids_k = position_ids//g_size_1

    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos_q = cos[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_q = sin[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos_k = cos[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_k = sin[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q) if q is not None else None
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k) if k is not None else None

    return q_embed, k_embed


"""
mspoe utils
"""

def _make_causal_mask(
        bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):
    cos = cos[:, position_ids]  # [head, bs, seq_len, dim]
    sin = sin[:, position_ids]  # [head, bs, seq_len, dim]

    cos = cos.transpose(0, 1)  # [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [bs, head, seq_len, dim]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def sample_rotary_emb(cos, sin, num_key_value_groups):
    cos = cos[::num_key_value_groups, ...]  # [head, bs, seq_len, dim]
    sin = sin[::num_key_value_groups, ...]  # [head, bs, seq_len, dim]
    return cos, sin

def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def self_extend_flash_forward(
        model_self,
        query_position,
        group_size_2,
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
    ):
    
    if query_position.max() >= group_size_2:
        neighbor_attn_output, neighbor_softmax_lse_right_padded, neighbor_prob = model_self._flash_attention_forward(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[group_size_2 - 1, 0],
            # right dim here does not matter and can be -1, or > 0 due to causal mask
            return_attn_probs=True,
        )

        group_attention_len = (
            kv_seq_len - group_size_2
        )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask

        group_attention_mask = attention_mask[:, :group_attention_len] if not attention_mask is None else None
        group_attn_output, group_softmax_lse_right_padded, group_prob = model_self._flash_attention_forward(
            group_query_states[:, -group_attention_len:, :, :],
            group_key_states[:, :group_attention_len, :, :],
            value_states[:, :group_attention_len, :, :],
            group_attention_mask,
            group_query_states[:, -group_attention_len:, :, :].shape[1],
            dropout=attn_dropout,
            window_size=[-1, -1],
            return_attn_probs=True,
        )  # note that kv and q's indexing are different! also query size could be different from kv length and very small during generation compared to prefilling


        # normalize lse first
        neighbor_seq_length = torch.Tensor([kv_seq_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask, axis=1, keepdim=True)  # [batch_size, 1]
        group_seq_length = torch.Tensor([group_attention_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask[:, :group_attention_len], axis=1, keepdim=True)  # [batch_size, 1]

        # convert align left to align right and convert exp(0) to 0
        neighbor_softmax_lse = torch.zeros_like(neighbor_softmax_lse_right_padded)
        group_softmax_lse = torch.zeros_like(group_softmax_lse_right_padded)
        for idx in range(bsz):
            if neighbor_seq_length[idx] > 0:
                neighbor_softmax_lse[idx, :, -neighbor_seq_length[idx] :] = neighbor_softmax_lse_right_padded[
                    idx, :, : neighbor_seq_length[idx]
                ]
            if group_seq_length[idx] > 0:
                group_softmax_lse[idx, :, -group_seq_length[idx] :] = group_softmax_lse_right_padded[
                    idx, :, : group_seq_length[idx]
                ]

        # attn_output size is [batch_size, max_seq_len (not the true one), query_length, dim]
        true_neighbor_seq_max_length = neighbor_softmax_lse.shape[
            -1
        ]  # it could be smaller than query_length due to the attention_mask
        true_group_seq_max_length = group_softmax_lse.shape[
            -1
        ]  # it could be smaller than group_query_layer[:, -group_attention_len:, :, :].shape[1] due to the attention_mask[:, :group_attention_len]

        neighbor_softmax_lse = neighbor_softmax_lse.transpose(1, 2).unsqueeze(
            -1
        )  # [batch_size, true_neighbor_seq_max_length, self.num_heads, 1]
        group_softmax_lse = group_softmax_lse.transpose(1, 2).unsqueeze(
            -1
        )  # [batch_size, true_group_seq_max_length, self.num_heads, 1]

        lse_gap = group_softmax_lse - neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :]
        #if  torch.isinf(neighbor_softmax_lse).any() or torch.isnan(neighbor_softmax_lse).any():
        #    import pdb; pdb.set_trace()
        
        neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] = 1 / (1 + torch.exp(lse_gap))
        neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :] = 1.
        group_softmax_lse = 1 / (1 + torch.exp(-lse_gap))



        neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
        )
        group_attn_output[:, -true_group_seq_max_length:, ...] = (
            group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
        )
        attn_output = torch.empty_like(neighbor_attn_output).copy_(
            neighbor_attn_output
        )  # might be slightly faster than clone
        #attn_output[:, group_size_2:, ...] += group_attn_output
        attn_output[:, group_size_2-kv_seq_len:, ...] += group_attn_output
        attn_output = torch.nan_to_num(attn_output, nan=0)  
    
    else:
        attn_output = model_self._flash_attention_forward(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[-1, -1],
        )

    return attn_output