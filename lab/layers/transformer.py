import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, dim_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale the embedding by the square root of the model dimension
        # to counteract the effect of the variance of the embeddings (relative to positional encodings)
        return self.embedding(x) * torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32))


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-torch.log(torch.tensor(10000.0)) / dim_model)
        )
        pe = torch.zeros(seq_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, dim_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim_model))
        self.bias = nn.Parameter(torch.zeros(dim_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_per_head = dim_model // num_heads

        self.query_linear = nn.Linear(dim_model, dim_model)
        self.key_linear = nn.Linear(dim_model, dim_model)
        self.value_linear = nn.Linear(dim_model, dim_model)
        self.out_linear = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float32))

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        Q = (
            self.query_linear(query)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )
        K = (
            self.key_linear(key)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )
        V = (
            self.value_linear(value)
            .view(batch_size, -1, self.num_heads, self.dim_per_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)

        output = self.out_linear(output)
        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ff = dim_ff

        self.attention = MultiHeadAttention(dim_model, num_heads, dropout)
        self.layer_norm1 = LayerNorm(dim_model)
        self.feed_forward = FeedForwardBlock(dim_model, dim_ff, dropout)
        self.layer_norm2 = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_ff = dim_ff

        self.self_attention = MultiHeadAttention(dim_model, num_heads, dropout)
        self.layer_norm1 = LayerNorm(dim_model)
        self.cross_attention = MultiHeadAttention(dim_model, num_heads, dropout)
        self.layer_norm2 = LayerNorm(dim_model)
        self.feed_forward = FeedForwardBlock(dim_model, dim_ff, dropout)
        self.layer_norm3 = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout(self_attn_output))

        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout(ff_output))

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder_block = TransformerEncoderBlock(dim_model, num_heads, dim_ff, dropout)
        self.decoder_block = TransformerDecoderBlock(dim_model, num_heads, dim_ff, dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encoder_block(src, src_mask)
        dec_output = self.decoder_block(tgt, enc_output, src_mask, tgt_mask)
        return dec_output
