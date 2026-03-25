import torch
import torch.nn as nn


class AttentionProbeHead(nn.Module):
    """
    Attention-based value probe:
    - Query projection: hidden_dim -> n_heads
    - Value projection: hidden_dim -> (n_heads * n_outputs)
    - Learned position weights: n_heads
    Produces per-sequence scalar predictions.
    """
    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        n_outputs: int = 1,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.n_outputs = n_outputs

        self.query_proj = nn.Linear(hidden_size, n_heads, device=device, dtype=dtype)
        self.value_proj = nn.Linear(hidden_size, n_heads * n_outputs, device=device, dtype=dtype)
        
        # ALiBi-style learned slope
        self.position_weights = nn.Parameter(torch.zeros(n_heads, device=device, dtype=dtype))

        # Initialize
        nn.init.normal_(self.query_proj.weight, std=0.01)
        nn.init.normal_(self.value_proj.weight, std=0.01)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, hidden: torch.Tensor):
        """
        hidden: [batch, seq, hidden_size]
        returns: [batch, seq, n_outputs]
        """
        B, S, H = hidden.shape

        # [B, S, H] @ [H, n_heads] = [B, S, n_heads]
        attn_logits = self.query_proj(hidden)

        # ALiBi-like positional slope: [1, S, 1] * [1, 1, n_heads]
        pos = torch.arange(S, device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)  # [S, 1]
        attn_logits = attn_logits + pos * self.position_weights[None, None, :]

        # compute softmax over positions
        attn_weights = torch.softmax(attn_logits, dim=1)  # [B, S, n_heads]

        # values: [B, S, n_heads * n_outputs] -> [B, S, n_heads, n_outputs]
        values = self.value_proj(hidden).view(B, S, self.n_heads, self.n_outputs)

        # weighted sum: sum over seq dimension
        # (B, S, n_heads, 1) * (B, S, n_heads, n_outputs)
        out = (attn_weights.unsqueeze(-1) * values).sum(dim=1)  # [B, n_heads, n_outputs]

        # combine heads (mean is simplest; can use a learned mix)
        out = out.mean(dim=1)  # [B, n_outputs]

        return out


class PerTokenAttentionProbe(nn.Module):
    """
    Attention-based value probe producing per-token outputs:
    - Query projection: hidden_dim -> n_heads
    - Value projection: hidden_dim -> (n_heads * n_outputs)
    - Learned position weights: n_heads
    Produces per-token scalar predictions.
    """
    def __init__(self, hidden_size, n_heads=4, n_outputs=1, device=None, dtype=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_outputs = n_outputs

        self.query_proj = nn.Linear(hidden_size, n_heads, device=device, dtype=dtype)
        self.value_proj = nn.Linear(hidden_size, n_heads * n_outputs, device=device, dtype=dtype)
        self.position_weights = nn.Parameter(torch.zeros(n_heads, device=device, dtype=dtype))

    def forward(self, hidden):
        # hidden: [B, S, H]
        B, S, H = hidden.shape

        # queries: [B, S, n_heads]
        attn_logits = self.query_proj(hidden)  # each token has its own query

        # ALiBi-like positional bias: [S, S, n_heads]
        pos = torch.arange(S, device=hidden.device).unsqueeze(0) - torch.arange(S, device=hidden.device).unsqueeze(1)
        pos = pos.clamp(min=0).unsqueeze(-1)  # causal mask style
        attn_logits = attn_logits.unsqueeze(2) + pos * self.position_weights[None, None, :]  # [B, S, S, n_heads]

        # softmax over tokens (axis=2)
        attn_weights = torch.softmax(attn_logits, dim=2)  # [B, S, S, n_heads]

        # values: [B, S, n_heads, n_outputs]
        values = self.value_proj(hidden).view(B, S, self.n_heads, self.n_outputs)

        # weighted sum for each token
        out = (attn_weights.unsqueeze(-1) * values.unsqueeze(1)).sum(dim=2)  # [B, S, n_heads, n_outputs]

        # combine heads
        out = out.mean(dim=2)  # [B, S, n_outputs]

        return out