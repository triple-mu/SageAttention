"""
Copyright (c) 2026 by triple-mu.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _q_int8_kernel(
        Q_ptr,
        Q_int8_ptr,
        Scale_ptr,
        stride_qb, stride_qs, stride_qn, stride_qd,
        stride_ob, stride_os, stride_on, stride_od,
        stride_sb, stride_sn, stride_ss,
        qo_len, head_dim, num_q_scales,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    pid_blk_id = tl.program_id(0).to(tl.int64)
    pid_nh_id = tl.program_id(1).to(tl.int64)
    pid_b_id = tl.program_id(2).to(tl.int64)

    offs_s = pid_blk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM).to(tl.int64)

    q_ptrs = Q_ptr + \
             (pid_b_id * stride_qb) + \
             (pid_nh_id * stride_qn) + \
             (offs_s[:, None] * stride_qs) + \
             (offs_d[None, :] * stride_qd)

    mask = (offs_s[:, None] < qo_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)

    DIM0: tl.constexpr = BLOCK_SIZE // 8
    DIM1: tl.constexpr = tl.constexpr(8)

    q_reshaped = tl.reshape(q, (DIM0, DIM1, HEAD_DIM))
    q_abs = tl.abs(q_reshaped)

    scale_val = tl.max(q_abs, axis=2)
    scale_val = tl.max(scale_val, axis=0)

    scale = scale_val / 127.0 + 1e-7

    q_scaled = q_reshaped / scale[None, :, None]
    q_clamped = tl.clamp(q_scaled, -128, 127)
    q_int8 = tl.extra.cuda.libdevice.round(q_clamped).to(tl.int8)
    q_int8 = tl.reshape(q_int8, (BLOCK_SIZE, HEAD_DIM))

    out_ptrs = Q_int8_ptr + \
               (pid_b_id * stride_ob) + \
               (pid_nh_id * stride_on) + \
               (offs_s[:, None] * stride_os) + \
               (offs_d[None, :] * stride_od)
    tl.store(out_ptrs, q_int8, mask=mask)

    offs_scale_s = pid_blk_id * 8 + tl.arange(0, 8).to(tl.int64)

    scale_ptrs = Scale_ptr + \
                 (pid_b_id * stride_sb) + \
                 (pid_nh_id * stride_sn) + \
                 (offs_scale_s * stride_ss)

    mask_scale = offs_scale_s < num_q_scales
    tl.store(scale_ptrs, scale, mask=mask_scale)


@triton.jit
def _k_int8_kernel(
        K_ptr,
        K_int8_ptr,
        Scale_ptr,
        stride_kb, stride_ks, stride_kn, stride_kd,
        stride_ob, stride_os, stride_on, stride_od,
        stride_sb, stride_sh, stride_ss,
        kv_len, head_dim, num_k_scales,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
):
    pid_blk_id = tl.program_id(0).to(tl.int64)
    pid_nh_id = tl.program_id(1).to(tl.int64)
    pid_b_id = tl.program_id(2).to(tl.int64)

    offs_s = pid_blk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM).to(tl.int64)

    k_ptrs = K_ptr + \
             (pid_b_id * stride_kb) + \
             (pid_nh_id * stride_kn) + \
             (offs_s[:, None] * stride_ks) + \
             (offs_d[None, :] * stride_kd)

    mask_k = (offs_s[:, None] < kv_len) & (offs_d[None, :] < head_dim)
    k = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)

    DIM0: tl.constexpr = BLOCK_SIZE // 8

    k_reshaped = tl.reshape(k, (DIM0, 4, 2, HEAD_DIM))
    k_abs = tl.abs(k_reshaped)

    tmp = tl.max(k_abs, axis=3)
    tmp = tl.max(tmp, axis=2)
    scale_vals = tl.max(tmp, axis=0)

    scale = scale_vals / 127.0 + 1e-7

    k_scaled = k_reshaped / scale[None, :, None, None]
    k_clamped = tl.clamp(k_scaled, -128, 127)
    k_int8 = tl.extra.cuda.libdevice.round(k_clamped).to(tl.int8)
    k_int8 = tl.reshape(k_int8, (BLOCK_SIZE, HEAD_DIM))

    out_ptrs = K_int8_ptr + \
               (pid_b_id * stride_ob) + \
               (pid_nh_id * stride_on) + \
               (offs_s[:, None] * stride_os) + \
               (offs_d[None, :] * stride_od)
    tl.store(out_ptrs, k_int8, mask=mask_k)

    offs_scale_s = pid_blk_id * 4 + tl.arange(0, 4).to(tl.int64)

    scale_ptrs = Scale_ptr + \
                 (pid_b_id * stride_sb) + \
                 (pid_nh_id * stride_sh) + \
                 (offs_scale_s * stride_ss)

    mask_scale = offs_scale_s < num_k_scales
    tl.store(scale_ptrs, scale, mask=mask_scale)


def per_thread_int8(
        q: torch.Tensor,
        k: torch.Tensor,
        km: torch.Tensor | None = None,
        BLKQ: int = 64,
        WARPQ: int = 16,
        BLKK: int = 128,
        WARPK: int = 128,
        sm_scale: float | None = None,
        tensor_layout: str = 'HND',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_qb, stride_qn, stride_qs, stride_qd = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        stride_qob, stride_qon, stride_qos, stride_qod = \
            q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3)
        stride_kb, stride_kn, stride_ks, stride_kd = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        stride_kob, stride_kon, stride_kos, stride_kod = \
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_qb, stride_qs, stride_qn, stride_qd = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
        stride_qob, stride_qos, stride_qon, stride_qod = \
            q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3)
        stride_kb, stride_ks, stride_kn, stride_kd = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
        stride_kob, stride_kos, stride_kon, stride_kod = \
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3)

    num_qo_warps = (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ)
    num_q_scales = num_qo_warps * 8
    q_scale = torch.empty((b, h_qo, num_q_scales), dtype=torch.float32, device=q.device)

    num_kv_warps = (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK)
    num_k_scales = num_kv_warps * 4
    k_scale = torch.empty((b, h_kv, num_k_scales), dtype=torch.float32, device=q.device)

    _q_int8_kernel[(num_qo_warps, h_qo, b)](
        q, q_int8, q_scale,
        stride_qb, stride_qs, stride_qn, stride_qd,
        stride_qob, stride_qos, stride_qon, stride_qod,
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        qo_len, head_dim, num_q_scales,
        BLOCK_SIZE=WARPQ,
        HEAD_DIM=triton.next_power_of_2(head_dim),
    )

    _k_int8_kernel[(num_kv_warps, h_kv, b)](
        k, k_int8, k_scale,
        stride_kb, stride_ks, stride_kn, stride_kd,
        stride_kob, stride_kos, stride_kon, stride_kod,
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        kv_len, head_dim, num_k_scales,
        BLOCK_SIZE=WARPK,
        HEAD_DIM=triton.next_power_of_2(head_dim),
    )
    return q_int8, q_scale, k_int8, k_scale
