import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.max_seq_len = 1500 
        self.inv_freq = self._compute_llama3_parameters(self.max_seq_len, dim)

    def _compute_default_rope_parameters(self, seq_len, dim, base=10000):
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        return inv_freq
    
    def _compute_llama3_parameters(self, seq_len, dim):
        # Gets the default RoPE parameters
        inv_freq = self._compute_default_rope_parameters(seq_len, dim)
    
        factor=8
        low_freq_factor=1
        high_freq_factor=4
        old_context_len=1500
    
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
    
        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    
        return inv_freq_llama

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float().to(x.device)).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _build_cache(self, max_seq_len):
        position_ids = torch.arange(max_seq_len).unsqueeze(0)  # [1, T]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()  # [1, 1, T]
    
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [1, T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, T, D]
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        #cos = cos.unsqueeze(unsqueeze_dim)
        #sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
if __name__ == "__main__":
    emb = RotaryEmbedding(256) 

    batch_size = 1
    seq_len = 10
    d_model = 256
    
    x = torch.randn(batch_size, seq_len, d_model)  # [B, T, D]
    print(x.size())
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  # [B, T]
    print(position_ids)
    cos, sin = emb(x, position_ids)
    print(cos)
    print(cos.size())
    print(sin)
    print(sin.size())

    q, k = emb.apply_rotary_pos_emb(x, x, cos, sin)
    #print(q.size())
    #print(k.size())

    print("========")
    cos_c, sin_c = emb._build_cache(seq_len)
    print(cos_c)
    print(cos_c.size())
    print(sin_c)
    print(sin_c.size())
