import math

import torch
from torch import nn
from torch.nn import functional as F

from util.model_average import *
from util.saver import *


class ConformerAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super(ConformerAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ConformerMultiHead(d_model, n_head, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def average(self, module):
        layernorm_average(self.layer_norm, module.layer_norm)
        self.attention.average(module.attention)

    def clear(self):
        pass

    def forward(self, x, pos_enc):
        out = self.layer_norm(x)
        out = self.attention(out, out, out, pos_enc)
        out = self.dropout(out)
        return out


class ConformerMultiHead(nn.Module):
    """QKV(+Pos) projection과 attention을 분리한 relative-PE multi-head attention.

    - Q/K/V/Pos projection: head별 순차 Linear 대신 fused Linear(d_model→d_model) 1회
    - attention: (B, H, L, head_dim) batched matmul로 전 head 동시 계산
    - 구 checkpoint(heads.N.Qdense.* / heads.N.PosDense.*)는 load 시 자동 fuse

    수학적으로 기존 per-head 구현과 동일하다.
    """
    def __init__(self, d_model, n_head, dropout=0.0):
        super(ConformerMultiHead, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.norm = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.pos_proj = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

        self._register_load_state_dict_pre_hook(self._fuse_legacy_weights)

    def _fuse_legacy_weights(self, state_dict, prefix, *args):
        """구 checkpoint(per-head 또는 3-proj 형식)를 단일 qkv 형식으로 변환."""
        if prefix + "heads.0.Qdense.weight" in state_dict:
            for new, old in (("q_proj", "Qdense"), ("k_proj", "Kdense"),
                             ("v_proj", "Vdense"), ("pos_proj", "PosDense")):
                w = [state_dict.pop("{}heads.{}.{}.weight".format(prefix, i, old)) for i in range(self.n_head)]
                b = [state_dict.pop("{}heads.{}.{}.bias".format(prefix, i, old)) for i in range(self.n_head)]
                state_dict[prefix + new + ".weight"] = torch.cat(w, dim=0)
                state_dict[prefix + new + ".bias"] = torch.cat(b, dim=0)
        if prefix + "q_proj.weight" in state_dict:
            for suf in ("weight", "bias"):
                parts = [state_dict.pop("{}{}_proj.{}".format(prefix, p, suf)) for p in ("q", "k", "v")]
                state_dict[prefix + "qkv_proj." + suf] = torch.cat(parts, dim=0)

    def _project_qkv(self, Q, K, V):
        """self-attention이면 GEMM 1회, 입력이 다르면 slice로 나눠 계산."""
        d = self.d_model
        if Q is K and K is V:
            return self.qkv_proj(Q).chunk(3, dim=-1)
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
        linear_average(self.pos_proj, module.pos_proj)

    def relative_shift(self, pos_score):
        """(B, H, L1, L2) relative position score shift (기존 3D 구현의 4D 확장)."""
        B, H, L1, L2 = pos_score.size()
        zeros = pos_score.new_zeros(B, H, L1, 1)
        padded = torch.cat([zeros, pos_score], dim=-1)   # (B, H, L1, L2+1)
        padded = padded.view(B, H, L2 + 1, L1)
        return padded[:, :, 1:].view_as(pos_score)

    def forward(self, Q, K, V, pos_enc, mask=None):
        B, Lq, _ = Q.size()

        # --- projection (head 통합, self-attention이면 qkv GEMM 1회) ---
        q, k, v = self._project_qkv(Q, K, V)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        p = self._split_heads(self.pos_proj(pos_enc))    # pos_enc: (1, L, D) → batch broadcast

        # --- attention (전 head batched) ---
        qk = torch.matmul(q, k.transpose(-2, -1))        # (B, H, Lq, Lk)
        qp = torch.matmul(q, p.transpose(-2, -1))        # (B, H, Lq, L)
        qp = self.relative_shift(qp)
        qk = (qk + qp) / self.norm

        if mask is not None:
            masking_value = -1e30 if qk.dtype == torch.float32 else -1e4
            qk = qk.masked_fill(mask.unsqueeze(1) == 0, masking_value)

        weighted = F.softmax(qk, dim=-1)
        output = torch.matmul(weighted, v)               # (B, H, Lq, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    _input = torch.rand([1, 320, 256])
    _pos = torch.rand([1, 320, 256])
    attention = ConformerAttention(256, 4)
    _output = attention(_input, _pos)
    print(_output.size())
