import math

import torch
from torch import nn
from torch.nn import functional as F

from util.model_average import *
from util.saver import *


class MultiHead(nn.Module):
    """QKV projection과 attention 연산을 분리한 multi-head attention.

    - projection: 단일 qkv_proj Linear(d_model → 3*d_model) GEMM 1회
      (self-attention일 때. cross-attention은 Q 따로 + K/V 묶음으로 분리 계산)
    - attention: (B, H, L, head_dim) batched matmul로 전 head 동시 계산

    기존 per-head 순차 Linear 대비 커널 launch가 대폭 줄어 GPU utilization이
    개선되며, fused weight가 구 weight의 concat이므로 수학적으로 동일하다.

    구 checkpoint는 load 시 자동 변환된다 (_fuse_legacy_weights pre-hook):
      - per-head 형식: heads.N.Qdense/Kdense/Vdense.*
      - 3-proj 형식:  q_proj/k_proj/v_proj.*
    """
    def __init__(self, d_model, n_head, isMask):
        super(MultiHead, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.isMask = isMask
        self.norm = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

        self._register_load_state_dict_pre_hook(self._fuse_legacy_weights)

    def _fuse_legacy_weights(self, state_dict, prefix, *args):
        """구 checkpoint(per-head 또는 3-proj 형식)를 단일 qkv 형식으로 변환."""
        # 1) per-head → 3-proj
        if prefix + "heads.0.Qdense.weight" in state_dict:
            for new, old in (("q_proj", "Qdense"), ("k_proj", "Kdense"), ("v_proj", "Vdense")):
                w = [state_dict.pop("{}heads.{}.{}.weight".format(prefix, i, old)) for i in range(self.n_head)]
                b = [state_dict.pop("{}heads.{}.{}.bias".format(prefix, i, old)) for i in range(self.n_head)]
                state_dict[prefix + new + ".weight"] = torch.cat(w, dim=0)
                state_dict[prefix + new + ".bias"] = torch.cat(b, dim=0)
        # 2) 3-proj → 단일 qkv
        if prefix + "q_proj.weight" in state_dict:
            for suf in ("weight", "bias"):
                parts = [state_dict.pop("{}{}_proj.{}".format(prefix, p, suf)) for p in ("q", "k", "v")]
                state_dict[prefix + "qkv_proj." + suf] = torch.cat(parts, dim=0)

    def _project_qkv(self, Q, K, V):
        """self-attention이면 GEMM 1회, cross-attention이면 Q / KV로 나눠 계산."""
        d = self.d_model
        if Q is K and K is V:
            q, k, v = self.qkv_proj(Q).chunk(3, dim=-1)
        else:
            w = self.qkv_proj.weight
            b = self.qkv_proj.bias
            q = F.linear(Q, w[:d], b[:d])
            if K is V:
                k, v = F.linear(K, w[d:], b[d:]).chunk(2, dim=-1)
            else:
                k = F.linear(K, w[d:2 * d], b[d:2 * d])
                v = F.linear(V, w[2 * d:], b[2 * d:])
        return q, k, v

    def _split_heads(self, x):
        """(B, L, D) -> (B, H, L, head_dim)"""
        B, L, _ = x.size()
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

    def average(self, module):
        linear_average(self.qkv_proj, module.qkv_proj)

    def forward(self, Q, K, V, mask=None):
        B, Lq, _ = Q.size()

        # --- QKV projection (분리된 단계) ---
        q, k, v = self._project_qkv(Q, K, V)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # --- attention (전 head batched) ---
        qk = torch.matmul(q, k.transpose(-2, -1)) / self.norm   # (B, H, Lq, Lk)

        masking_value = -1e30 if qk.dtype == torch.float32 else -1e4
        if mask is not None:
            # mask: (B, Lq, Lk), 1=valid → head 차원 broadcast
            qk = qk.masked_fill(mask.unsqueeze(1) == 0, masking_value)
        if self.isMask:
            causal = torch.triu(torch.ones(qk.size(-2), qk.size(-1), device=qk.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(causal, masking_value)

        weighted = F.softmax(qk, dim=-1)
        out = torch.matmul(weighted, v)                          # (B, H, Lq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    randQ = torch.rand([1, 100, 512])
    randK = torch.rand([1, 120, 512])
    randV = torch.rand([1, 120, 512])
    model = MultiHead(512, 8, False)
    out = model(randQ, randK, randV)
    print(out.size())
