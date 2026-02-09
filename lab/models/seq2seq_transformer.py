import torch
import torch.nn as nn

from .transformer import (
    InputEmbedding,
    PositionalEncoding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.pad_token_id = pad_token_id

        self.src_embedding = InputEmbedding(dim_model, src_vocab_size)
        self.tgt_embedding = InputEmbedding(dim_model, tgt_vocab_size)

        self.pos_encoding = PositionalEncoding(dim_model, max_seq_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(dim_model, num_heads, dim_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(dim_model, num_heads, dim_ff, dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        self.output_projection = nn.Linear(dim_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)

    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return (seq != self.pad_token_id).unsqueeze(1).unsqueeze(2).bool()

    def create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
        return mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.src_embedding(src)  # (batch_size, src_len, dim_model)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt)  # (batch_size, tgt_len, dim_model)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        logits = self.output_projection(dec_output)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = src.device
        batch_size = src.size(0)

        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_mask)

        tgt = torch.full((batch_size, 1), bos_token_id, device=device)

        for _ in range(max_len):
            tgt_len = tgt.size(1)
            tgt_causal_mask = self.create_causal_mask(tgt_len, device)
            tgt_pad_mask = self.create_padding_mask(tgt)
            tgt_mask = tgt_causal_mask & tgt_pad_mask

            dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)

            next_token_logits = self.output_projection(dec_output[:, -1, :])

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return tgt
