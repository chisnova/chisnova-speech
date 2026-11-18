import math

import torch
from torch import nn
from torch.nn import functional as F

from util.model_average import *
from util.saver import *


class ConformerStreamAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super(ConformerStreamAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = ConformerStreamMultiHead(d_model, n_head, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, att_mask=None, cos=None, sin=None):
        out = self.layer_norm(x)
        out = self.attention(out, out, out, att_mask=att_mask, cos=cos, sin=sin)
        out = self.dropout(out)
        return out

    def clear(self):
        self.attention.clear()

    def stream(self, x, cos=None, sin=None, left_context=None):
        out = self.layer_norm(x)
        out = self.attention.stream(out, out, out, cos=cos, sin=sin, left_context=left_context)
        return out


class ConformerStreamMultiHead(nn.Module):
    """QKV projection과 attention을 분리한 스트리밍 multi-head attention.

    - QKV projection: 단일 qkv_proj Linear(d_model→3*d_model) GEMM 1회
      (스트림 인코더는 항상 self-attention이므로 무조건 fused 경로)
    - attention: (B, H, L, head_dim) batched matmul로 전 head 동시 계산
    - KV cache: head 차원을 포함한 단일 텐서 (B, H, T, head_dim)로 관리
    - 구 checkpoint(per-head heads.N.Qdense.* / 3-proj q_proj.*)는 load 시 자동 fuse

    수학적으로 기존 per-head 구현과 동일하다.
    """
    def __init__(self, d_model, n_head, dropout=0.0):
        super(ConformerStreamMultiHead, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.norm = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

        self.streamK = None   # (B, H, T_cache, head_dim)
        self.streamV = None

        self._register_load_state_dict_pre_hook(self._fuse_legacy_weights)

    def _fuse_legacy_weights(self, state_dict, prefix, *args):
        """구 checkpoint(per-head 또는 3-proj 형식)를 단일 qkv 형식으로 변환."""
        if prefix + "heads.0.Qdense.weight" in state_dict:
            for new, old in (("q_proj", "Qdense"), ("k_proj", "Kdense"), ("v_proj", "Vdense")):
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

    def _merge_heads(self, x):
        """(B, H, L, head_dim) -> (B, L, D)"""
        B, H, L, hd = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    # ---------------- RoPE ----------------
    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        # q, k: (B, H, L, head_dim) / cos, sin: (1, L, head_dim) → head 차원 broadcast
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def clear(self):
        self.streamK = None
        self.streamV = None

    def forward(self, Q, K, V, att_mask=None, cos=None, sin=None):
        q, k, v = self._project_qkv(Q, K, V)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if cos is not None:
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        qk = torch.matmul(q, k.transpose(-2, -1)) / self.norm    # (B, H, L, L)

        if att_mask is None:
            # per-frame causal (기존 동작)
            L = qk.size(-1)
            att_mask = torch.triu(torch.ones(L, L, device=qk.device, dtype=torch.bool), diagonal=1)
        qk = qk.masked_fill(att_mask, -1e4)                      # (L,L) 또는 (B,L,L)→broadcast

        weighted = F.softmax(qk, dim=-1)
        output = torch.matmul(weighted, v)
        output = self._merge_heads(output)
        output = self.linear(output)
        return output

    def stream(self, Q, K, V, cos=None, sin=None, left_context=None):
        q, k, v = self._project_qkv(Q, K, V)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if cos is not None:
            # cache된 k는 이미 절대 위치로 회전되어 있으므로 새 chunk만 회전
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if self.streamK is None:
            tmp_k = k
            tmp_v = v
        else:
            tmp_k = torch.cat([self.streamK, k], dim=2)
            tmp_v = torch.cat([self.streamV, v], dim=2)

        len_q = q.size(2)
        if left_context is not None and tmp_k.size(2) > len_q + left_context:
            tmp_k = tmp_k[:, :, -(len_q + left_context):, :]
            tmp_v = tmp_v[:, :, -(len_q + left_context):, :]

        # 추론 시 autograd graph가 cache를 통해 누적되지 않도록 저장본만 detach
        self.streamK = tmp_k.detach()
        self.streamV = tmp_v.detach()

        qk = torch.matmul(q, tmp_k.transpose(-2, -1)) / self.norm  # (B, H, len_q, len_k)

        # chunk 내부는 양방향(non-causal): 현재 chunk 전 프레임 + cache(좌컨텍스트)를
        # 모두 참조한다. 청크 하나만큼 buffering 후 처리하는 chunk 기반 스트리밍 설계로,
        # 학습의 chunk mask(청크 내부 양방향)와 receptive field가 일치한다.
        weighted = F.softmax(qk, dim=-1)
        output = torch.matmul(weighted, tmp_v)
        output = self._merge_heads(output)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    _input = torch.rand([1, 320, 256])
    attention = ConformerStreamAttention(256, 4)
    _output = attention(_input)
    print(_output.size())
