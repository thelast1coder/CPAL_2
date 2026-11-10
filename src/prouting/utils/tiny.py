"""Tiny random MoE models built from config classes (no downloads). Seeded, float32 by default.

Sizes follow PHASE0_SETUP.md Section 4: hidden 32, (moe_)intermediate 64, 2 layers, 4 heads, 8 experts,
k = 4 (k = 2 also used), vocab 128, max positions 128.
"""
from __future__ import annotations

import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

COMMON = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
              vocab_size=128, max_position_embeddings=128, num_experts_per_tok=4)


def _init_weights(model, seed: int, std: float = 0.2) -> None:
    """Re-initialise every parameter with a seeded normal (config init can leave routers at zero)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if p.dim() == 1 and ("norm" in name or "layernorm" in name):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)


def _finish(model, seed, dtype):
    _init_weights(model, seed)
    model.config._experts_implementation = "eager"   # reference kernel (docs/decisions.md)
    return model.to(dtype).eval()


def tiny_olmoe(k: int = 4, seed: int = 0, dtype=torch.float32, num_layers: int = 2):
    from transformers import OlmoeConfig, OlmoeForCausalLM
    cfg = OlmoeConfig(**{**COMMON, "num_experts_per_tok": k, "num_hidden_layers": num_layers},
                      intermediate_size=64, num_experts=8, norm_topk_prob=False)
    torch.manual_seed(seed)
    return _finish(OlmoeForCausalLM(cfg), seed, dtype)


def tiny_qwen3_moe(k: int = 4, seed: int = 0, dtype=torch.float32):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(**{**COMMON, "num_experts_per_tok": k}, intermediate_size=64, moe_intermediate_size=64,
                         num_experts=8, norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[],
                         head_dim=8)
    torch.manual_seed(seed)
    return _finish(Qwen3MoeForCausalLM(cfg), seed, dtype)


def tiny_qwen2_moe(k: int = 4, seed: int = 0, dtype=torch.float32):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    cfg = Qwen2MoeConfig(**{**COMMON, "num_experts_per_tok": k}, intermediate_size=64, moe_intermediate_size=64,
                         shared_expert_intermediate_size=64, num_experts=8, norm_topk_prob=False,
                         decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(seed)
    return _finish(Qwen2MoeForCausalLM(cfg), seed, dtype)


def tiny_deepseek_v2(k: int = 4, seed: int = 0, dtype=torch.float32, topk_method: str = "greedy"):
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    cfg = DeepseekV2Config(**{**COMMON, "num_experts_per_tok": k, "num_hidden_layers": 3}, intermediate_size=64,
                           moe_intermediate_size=64, n_routed_experts=8, num_experts=8, n_shared_experts=1,
                           first_k_dense_replace=1, topk_method=topk_method, n_group=4, topk_group=2,
                           routed_scaling_factor=1.5, q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=4,
                           qk_nope_head_dim=4, v_head_dim=8, head_dim=4)
    torch.manual_seed(seed)
    return _finish(DeepseekV2ForCausalLM(cfg), seed, dtype)


BUILDERS = {"olmoe": tiny_olmoe, "qwen3_moe": tiny_qwen3_moe, "qwen2_moe": tiny_qwen2_moe,
            "deepseek_v2": tiny_deepseek_v2}


def random_tokens(n_seq: int, seq_len: int, vocab: int = 128, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n_seq, seq_len), generator=gen)


def structured_tokens(n_seq: int, seq_len: int, vocab: int = 128, seed: int = 0) -> torch.Tensor:
    """Synthetic sequences with learnable-looking structure (repeating n-grams) for the tiny run."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(n_seq, seq_len, dtype=torch.long)
    for i in range(n_seq):
        period = int(torch.randint(2, 6, (1,), generator=gen))
        motif = torch.randint(0, vocab, (period,), generator=gen)
        out[i] = motif.repeat(seq_len // period + 1)[:seq_len]
    return out
