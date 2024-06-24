# Mostly Borrowed Yu Zhang, Songlin Yang

import functools
from typing import Tuple

import torch
import triton
import triton.language as tl


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper


@triton.jit
def fused_recurrent_fwd_kernel(
    # B: batch_size, T: seq_len, D: d
    k,  # key [B, L, D_K]
    v,  # value [B, L, D_V].
    beta,  # beta [B, L]
    h,  # output [B, L, D_K, D_V]
    initial_state,
    s_k_b,  # stride size: L * D_K
    s_k_t,  # stride size: D_K
    s_k_d,  # stride size: 1
    s_v_b,  # stride size: L * D_V
    s_v_t,  # stride size: D_V
    s_v_d,  # stride size: 1
    s_h_b,  # stride size: L * D_K * D_V
    s_h_t,  # stride size: D_K * D_V
    s_h_dk,  # stride size: D_V
    s_h_dv,  # stride size: 1
    s_beta,  # stride size: L
    B,  # batch size
    T,  # seq_len
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
):

    # indices
    i_k, i_b = tl.program_id(0), tl.program_id(1)

    p_k = k + i_b * s_k_b + i_k * BK + tl.arange(0, BK)
    p_v = v + i_b * s_v_b + tl.arange(0, DV)
    p_beta = beta + i_b * s_beta

    p_h = h + i_b * s_h_b + (i_k * BK + tl.arange(0, BK)[:, None]) * DV + (tl.arange(0, DV)[None, :])

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (tl.arange(0, DV)) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    h = tl.zeros([BK, DV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = (
            initial_state + i_b * s_h_t + (i_k * BK + tl.arange(0, BK)[:, None]) * DV + (tl.arange(0, DV)[None, :])
        )
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _v_minus = tl.sum(h * _k[:, None], axis=0)
        _v -= _v_minus
        _beta = tl.load(p_beta).to(tl.float32)
        # in-place overwrite
        tl.store(p_v, _v.to(p_v.dtype.element_ty), mask=mask_bv)
        _v *= _beta
        h += _k[:, None] * _v[None, :]
        tl.store(p_h, h.to(p_h.dtype.element_ty), mask=mask_kv)

        p_k += DK
        p_v += DV
        p_h += DK * DV
        p_beta += 1


@triton.jit
def fused_recurrent_bwd_kernel(
    # B: batch_size, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    h,  # output [B, L, D_K, D_V]
    k,  # key [B, L, DW_V]
    v,  # value [B, L, D_V]
    beta,  # beta [B, L]
    do,  # gradient of output [B, L, D_K, D_V]
    dk,  # gradient of key [NV, B, L, D_K]
    dv,  # gradient of value [NK, B, L, D_V]
    dbeta,  # gradient of beta [B, H, L]
    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,
    s_k_b,  # stride size: L * D_K
    s_k_t,  # stride size: D_K
    s_k_d,  # stride size: 1
    s_v_b,  # stride size: L * D_V
    s_v_t,  # stride size: D_V
    s_v_d,  # stride size: 1
    s_h_b,  # stride size: L * D_K * D_V
    s_h_t,  # stride size: D_K * D_V
    s_h_dk,  # stride size: D_V
    s_h_dv,  # stride size: 1
    s_beta,  # stride size: L
    B,  # batch_size
    T,  # seq_len
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    DK: tl.constexpr,  # D_K
    DV: tl.constexpr,  # D_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
):
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = tl.arange(0, DV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    p_k = k + i_bh * s_k_b + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_v = v + i_bh * s_v_b + tl.arange(0, DV) + (T - 1) * DV
    p_h = (
        h
        + i_bh * s_h_b
        + (i_k * BK + tl.arange(0, BK)[:, None]) * DV
        + (tl.arange(0, DV)[None, :])
        + (T - 2) * DK * DV
    )
    p_beta = beta + i_bh * T + T - 1

    p_do = (
        do
        + i_bh * s_h_b
        + (i_k * BK + tl.arange(0, BK)[:, None]) * DV
        + (tl.arange(0, DV)[None, :])
        + (T - 1) * DK * DV
    )
    p_dbeta = dbeta + (i_bh + i_k * B) * T + T - 1
    p_dk = dk + i_bh * s_k_b + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B) * s_v_b + tl.arange(0, DV) + (T - 1) * DV

    d_h = tl.zeros([BK, DV], dtype=tl.float32)

    for i in range(T):
        _do = tl.load(p_do, mask=mask_kv, other=0).to(tl.float32)  # D_K, D_V
        if i < T - 1:
            _h = tl.load(p_h, mask=mask_kv, other=0).to(tl.float32)
        else:
            _h = tl.zeros([BK, DV], dtype=tl.float32)
            if USE_INITIAL_STATE:
                p_init_s = (
                    initial_state
                    + i_bh * s_h_t
                    + (i_k * BK + tl.arange(0, BK)[:, None]) * DV
                    + (tl.arange(0, DV)[None, :])
                )
                _h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _beta = tl.load(p_beta).to(tl.float32)

        d_h += _do

        d_v = tl.sum(d_h * _k[:, None], axis=0)
        d_beta = tl.sum(d_v * _v)
        d_v = d_v * _beta

        d_k = tl.sum(d_h * _v[None, :] * _beta, axis=1)
        d_k -= tl.sum(d_v[None, :] * _h, axis=1)

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)
        tl.store(p_dbeta, d_beta.to(p_dbeta.dtype.element_ty))

        d_h -= _k[:, None] * d_v[None, :]

        p_do -= DV * DK
        p_h -= DV * DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV
        p_dbeta -= 1
        p_beta -= 1


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, k, v, beta, initial_state=None):
        batch_size, seq_len, dim_k = k.shape
        dim_v = v.shape[-1]

        BK = min(triton.next_power_of_2(dim_v), 64)
        NK = triton.cdiv(dim_k, BK)
        num_stages = 1
        num_warps = 4

        h = torch.empty(batch_size, seq_len, dim_k, dim_v, device=k.device, dtype=k.dtype)

        grid = (NK, batch_size)
        fused_recurrent_fwd_kernel[grid](
            k,
            v,
            beta,
            h,
            initial_state,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            h.stride(0),
            h.stride(1),
            h.stride(2),
            h.stride(3),
            beta.stride(0),
            batch_size,
            seq_len,
            DK=dim_k,
            DV=dim_v,
            BK=BK,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
        )
        ctx.save_for_backward(h, k, v, beta, initial_state)
        return h

    @staticmethod
    @contiguous
    def backward(ctx, do):
        h, k, v, beta, initial_state = ctx.saved_tensors
        batch_size, seq_len, dim_k, dim_v = h.shape
        BK = min(triton.next_power_of_2(dim_v), 64)
        NK = triton.cdiv(dim_k, BK)
        num_stages = 1
        num_warps = 4

        dk = k.new_empty(batch_size, seq_len, dim_k)
        dv = v.new_empty(NK, batch_size, seq_len, dim_v)
        dbeta = beta.new_empty(NK, batch_size, seq_len)
        grid = (NK, batch_size)

        fused_recurrent_bwd_kernel[grid](
            h,
            k,
            v,
            beta,
            do,
            dk,
            dv,
            dbeta,
            initial_state,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            h.stride(0),
            h.stride(1),
            h.stride(2),
            h.stride(3),
            beta.stride(0),
            batch_size,
            seq_len,
            DK=dim_k,
            DV=dim_v,
            BK=BK,
            num_warps=num_warps,
            num_stages=num_stages,
            USE_INITIAL_STATE=initial_state is not None,
        )
        dv = dv.sum(0)
        dbeta = dbeta.sum(0)
        return dk.to(k), dv.to(v), dbeta.to(beta), None, None


def fused_recurrent_linear_attn_delta_rule(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    initial_state: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    final_state = FusedRecurrentFunction.apply(k, v, beta, initial_state)
    return final_state


def fused_recurrent_linear_attn_delta_rule_ref(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    initial_state: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()

    batch_size, seq_len, d_k = k.shape
    d_v = v.shape[-1]

    h0 = torch.zeros(batch_size, d_k, d_v, device=k.device, dtype=k.dtype)
    h = []

    if initial_state is not None:
        h0 += initial_state

    for i in range(seq_len):
        _k = k[:, i]  # B, D_K
        _v = v[:, i]  # B, D_V
        _beta = beta[:, i].unsqueeze(1) if beta is not None else 1
        _v_minus = torch.einsum("bkv, bk->bv", h0, _k)
        _v = _v - _v_minus
        _v = _v * _beta
        h0 = h0.clone() + torch.einsum("bk, bv->bkv", _k, _v)
        h.append(h0)

    h = torch.stack(h, dim=1)
    return h


if __name__ == "__main__":
    B = 8
    L = 1024
    DK = 2048
    DV = 128
    k = (torch.randn(B, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = (torch.randn(B, L, DV)).cuda().requires_grad_(True)
    beta = torch.randn(B, L).cuda().sigmoid().requires_grad_(True)
    do = torch.randn(B, L, DK, DV).cuda()

    # reference
    o = fused_recurrent_linear_attn_delta_rule_ref(k, v, beta)
    o.backward(do, retain_graph=True)
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    beta_grad, beta.grad = beta.grad, None

    o2 = fused_recurrent_linear_attn_delta_rule(k, v, beta)
    o2.backward(do)
    print(o, o2)
    print("All passed!")
