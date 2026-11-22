import numpy as np
import os
import torch
import gc
import librosa
from threading import Thread
import threading
import random
import copy
import torchaudio.transforms as T

class BayesianCMVNChunk:
    """Normal–Inverse-Gamma(N-IG) 기반 CMVN – Chunk 버전."""
    def __init__(self, mu0, var0, kappa0=800, alpha0=2.5, eps=1e-8):
        """
        mu0, var0 : (D,) 전역 평균·분산
        kappa0    : 평균 사전 가상 프레임 수
        alpha0    : 분산 사전 자유도 (>1)
        """
        self.mu     = mu0.clone()                 # posterior μ
        self.kappa  = float(kappa0)
        self.alpha  = float(alpha0)
        self.beta   = (alpha0 - 1) * var0.clone() # posterior β
        self.eps    = eps

    def update_chunk(self, x):
        """
        x : (T, D) or (B, T, D)
        반환 : 정규화된 tensor
        """
        #x = x.double()
        orig_shape = x.shape
        x2d = x.view(-1, x.size(-1))                       # (N_chunk, D)

        chunk_N    = x2d.size(0)
        chunk_mean = x2d.mean(dim=0)
        print(f"chunk_mean: {chunk_mean}")
        chunk_ss   = ((x2d - chunk_mean)**2).sum(dim=0)    # within-chunk SSE
        print(f"chunk_ss: {chunk_ss}")
        # --- 파라미터 갱신 ---
        kappa_new = self.kappa + chunk_N
        mu_new    = (self.kappa * self.mu + chunk_N * chunk_mean) / kappa_new
        print(f"mu_new: {mu_new}")
        alpha_new = self.alpha + chunk_N / 2
        beta_new  = (self.beta
                     + 0.5 * chunk_ss
                     + 0.5 * (self.kappa * chunk_N / kappa_new) * (chunk_mean - self.mu).pow(2))
        print(f"beta_new: {beta_new}")
        # posterior 기대값
        var_post  = beta_new / (alpha_new - 1)
        print(f"var_post: {var_post}")

        # 상태 갱신
        self.mu, self.kappa, self.alpha, self.beta = mu_new, kappa_new, alpha_new, beta_new

        sub = x2d - self.mu
        print(f"sub: {sub}")
        std = torch.sqrt(var_post + self.eps)
        print(f"std: {std}")
        norm = sub / std
        print(f"norm: {norm}")
        return norm.view(orig_shape).type_as(x)


def save_file(val, name):
    val.detach().numpy().tofile(name)


if __name__ == "__main__":
    mu0 = torch.randn(10)
    std = torch.randn(10)
    x = torch.randn(10, 10)

    print(f"mu0: {mu0}")
    print(f"std: {std}")
    print(f"x: {x}")

    cmvn = BayesianCMVNChunk(mu0, std)
    out = cmvn.update_chunk(x)

    print(f"out: {out}")
    # Save tensors to .pt files
    save_file(mu0, 'mu0.pt')
    save_file(std, 'std.pt') 
    save_file(x, 'input.pt')
    save_file(out, 'output.pt')
    