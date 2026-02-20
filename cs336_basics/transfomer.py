import torch
import torch.nn as nn
import einops
import math
from jaxtyping import Bool, Float, Int

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sigma = math.sqrt(2 / (in_features + out_features))
        self.weights = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features), std=sigma, a = -3*sigma, b = 3*sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features
        return einops.einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")

    def load_weights(self, weights: torch.Tensor):
        assert weights.shape == (self.out_features, self.in_features)
        self.weights.data = weights

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = nn.Parameter(nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=1.0, a = -3.0, b = 3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights[x]

    def load_weights(self, weights: torch.Tensor):
        assert weights.shape == (self.vocab_size, self.d_model)
        self.weights.data = weights

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        output = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weights
        return output.to(in_dtype)

    def load_weights(self, weights: torch.Tensor):
        assert weights.shape == (self.d_model,)
        self.weights.data = weights

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wx_1 = self.w1(x)
        silu_wx_1 = silu(wx_1)
        return self.w2(silu_wx_1 * self.w3(x))

    def load_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        self.w1.weights.data = w1
        self.w2.weights.data = w2
        self.w3.weights.data = w3

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        ro_weights = torch.zeros((max_seq_len, d_k, d_k), device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angle = i / pow(theta, 2 * k / d_k)
                R = torch.tensor([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ], device=device)
                ro_weights[i, 2*k:2*k+2, 2*k:2*k+2] = R
        self.register_buffer('ro_weights', ro_weights, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        output = einops.einsum(x, self.ro_weights[token_positions, :, :], "... seq_len j, ... seq_len i j -> ... seq_len i")
        return output

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)[0]
    x = x - x_max
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    QK = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        QK = QK.masked_fill(mask == 0, float('-inf'))
    QK = softmax(QK, dim=-1)
    return einops.einsum(QK, V, "... queries keys, ... keys d_v -> ... queries d_v")

def multihead_self_attention(d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    d_in = q_proj_weight.shape[-1]
    d_k = q_proj_weight.shape[-2] // num_heads
    d_v = v_proj_weight.shape[-2] // num_heads
    seq_len = in_features.shape[-2]
    assert d_k == d_v
    assert d_model == num_heads * d_k

    q_proj_weights = q_proj_weight.reshape(num_heads, d_k, d_in)
    k_proj_weights = k_proj_weight.reshape(num_heads, d_k, d_in)
    v_proj_weights = v_proj_weight.reshape(num_heads, d_v, d_in)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    q = einops.einsum(in_features, q_proj_weights, "batch seq_len d_in, num_heads d_k d_in -> batch num_heads seq_len d_k")
    k = einops.einsum(in_features, k_proj_weights, "batch seq_len d_in, num_heads d_k d_in -> batch num_heads seq_len d_k")
    v = einops.einsum(in_features, v_proj_weights, "batch seq_len d_in, num_heads d_v d_in -> batch num_heads seq_len d_v")
    
    attention = torch.concat([
        scaled_dot_product_attention(
            q[:, i, :, :], 
            k[:, i, :, :], 
            v[:, i, :, :], 
            causal_mask
        ) for i in range(num_heads)
    ], dim=-1)
    
    return einops.einsum(attention, o_proj_weight, "batch seq_len d_v_nh, d_model d_v_nh -> batch seq_len d_model")

def multihead_self_attention_with_rope(d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    d_in = q_proj_weight.shape[-1]
    d_k = q_proj_weight.shape[-2] // num_heads
    d_v = v_proj_weight.shape[-2] // num_heads
    seq_len = in_features.shape[-2]
    assert d_k == d_v
    assert d_model == num_heads * d_k

    q_proj_weights = q_proj_weight.reshape(num_heads, d_k, d_in)
    k_proj_weights = k_proj_weight.reshape(num_heads, d_k, d_in)
    v_proj_weights = v_proj_weight.reshape(num_heads, d_v, d_in)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    q = einops.einsum(in_features, q_proj_weights, "batch seq_len d_in, num_heads d_k d_in -> batch num_heads seq_len d_k")
    k = einops.einsum(in_features, k_proj_weights, "batch seq_len d_in, num_heads d_k d_in -> batch num_heads seq_len d_k")
    v = einops.einsum(in_features, v_proj_weights, "batch seq_len d_in, num_heads d_v d_in -> batch num_heads seq_len d_v")

    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    q_with_rope = rope(q, token_positions)
    k_with_rope = rope(k, token_positions)

    attention = torch.concat([
        scaled_dot_product_attention(
            q_with_rope[:, i, :, :], 
            k_with_rope[:, i, :, :], 
            v[:, i, :, :], 
            causal_mask
        ) for i in range(num_heads)
    ], dim=-1)
    return einops.einsum(attention, o_proj_weight, "batch seq_len d_v_nh, d_model d_v_nh -> batch seq_len d_model")

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def load_weights(self, weights: dict[str, torch.Tensor]):
        self.q_proj.load_weights(weights["attn.q_proj.weight"])
        self.k_proj.load_weights(weights["attn.k_proj.weight"])
        self.v_proj.load_weights(weights["attn.v_proj.weight"])
        self.o_proj.load_weights(weights["attn.output_proj.weight"])
        self.ln1.load_weights(weights["ln1.weight"])
        self.ffn.load_weights(weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])
        self.ln2.load_weights(weights["ln2.weight"])

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device).unsqueeze(0)
        token_positions = token_positions.expand(in_features.shape[0], -1)
        residual = in_features
        x = self.ln1(residual)
        x = multihead_self_attention_with_rope(
            self.d_model,
            self.num_heads,
            self.max_seq_len,
            self.theta,
            self.q_proj.weights,
            self.k_proj.weights,
            self.v_proj.weights,
            self.o_proj.weights,
            x,
            token_positions
        )
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual
        return x

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def load_weights(self, weights: dict[str, torch.Tensor]):
        self.token_embeddings.load_weights(weights["token_embeddings.weight"])
        for i, layer in enumerate(self.layers):
            layer_weights = {
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
            }
            layer.load_weights(layer_weights)
        self.ln_final.load_weights(weights["ln_final.weight"])
        self.lm_head.load_weights(weights["lm_head.weight"])

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x