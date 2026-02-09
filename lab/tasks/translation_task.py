import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.seq2seq_transformer import Seq2SeqTransformer


class TranslationDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        src_tokenizer: AutoTokenizer,
        tgt_tokenizer: AutoTokenizer,
        max_len: int = 128,
        src_lang: str = "en",
        tgt_lang: str = "de",
    ):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]

        src_tokens = self.src_tokenizer(
            src_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tgt_tokens = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "src_input_ids": src_tokens["input_ids"].squeeze(0),
            "src_attention_mask": src_tokens["attention_mask"].squeeze(0),
            "tgt_input_ids": tgt_tokens["input_ids"].squeeze(0),
            "tgt_attention_mask": tgt_tokens["attention_mask"].squeeze(0),
        }


class TranslationModel(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        warmup_steps: int = 4000,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Seq2SeqTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            dim_model=dim_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_ff=dim_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, label_smoothing=label_smoothing
        )

        self.train_loss_history = []
        self.val_loss_history = []

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        src_mask = self.model.create_padding_mask(src)

        tgt_causal_mask = self.model.create_causal_mask(tgt_len, tgt.device)
        tgt_pad_1d = tgt != self.hparams.pad_token_id
        tgt_causal_expanded = tgt_causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        tgt_pad_expanded = tgt_pad_1d.unsqueeze(1) & tgt_pad_1d.unsqueeze(2)

        tgt_mask = tgt_causal_expanded & tgt_pad_expanded  # (batch, tgt_len, tgt_len)
        tgt_mask = tgt_mask.unsqueeze(1)  # (batch, 1, tgt_len, tgt_len) for multi-head attention

        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask, tgt_mask = self.create_masks(src, tgt)

        return self.model(src, tgt, src_mask, tgt_mask)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        src = batch["src_input_ids"]
        tgt = batch["tgt_input_ids"]

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        logits = self.forward(src, tgt_input)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True)

        perplexity = torch.exp(loss)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True)

        self.train_loss_history.append(loss.item())

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        src = batch["src_input_ids"]
        tgt = batch["tgt_input_ids"]

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        logits = self.forward(src, tgt_input)

        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

        predictions = torch.argmax(logits, dim=-1)
        mask = tgt_target != self.hparams.pad_token_id
        correct = (predictions == tgt_target) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        perplexity = torch.exp(loss)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True)
        self.val_loss_history.append(loss.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )

        def lr_lambda(step):
            if step == 0:
                return 1e-8
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            return (self.hparams.warmup_steps / step) ** 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def translate(
        self, src_tokens: torch.Tensor, max_len: int = 100, temperature: float = 1.0
    ) -> torch.Tensor:
        return self.model.generate(
            src_tokens,
            max_len=max_len,
            bos_token_id=self.hparams.bos_token_id,
            eos_token_id=self.hparams.eos_token_id,
            temperature=temperature,
        )


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "opus_books",
        language_pair: str = "en-de",
        batch_size: int = 32,
        max_len: int = 128,
        num_workers: int = 4,
        train_size: int | None = None,
        val_size: int | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.language_pair = language_pair
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_size = train_size
        self.val_size = val_size

        self.src_lang, self.tgt_lang = language_pair.split("-")

        print("Loading tokenizers...")
        self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.src_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.tgt_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        self.src_tokenizer.pad_token = "<pad>"
        self.tgt_tokenizer.pad_token = "<pad>"

    def setup(self, stage: str | None = None):
        print(f"Loading dataset: {self.dataset_name} ({self.language_pair})")

        try:
            dataset = load_dataset(self.dataset_name, self.language_pair)

            train_data = list(dataset["train"])

            if self.train_size:
                train_data = train_data[: self.train_size]

            if "validation" in dataset and not self.val_size:
                val_data = list(dataset["validation"])
            else:
                val_split_size = self.val_size or min(1000, len(train_data) // 10)
                val_data = train_data[:val_split_size]
                train_data = train_data[val_split_size:]

            self.train_dataset = TranslationDataset(
                train_data,
                self.src_tokenizer,
                self.tgt_tokenizer,
                self.max_len,
                self.src_lang,
                self.tgt_lang,
            )

            self.val_dataset = TranslationDataset(
                val_data,
                self.src_tokenizer,
                self.tgt_tokenizer,
                self.max_len,
                self.src_lang,
                self.tgt_lang,
            )

            print(f"Train size: {len(self.train_dataset)}")
            print(f"Val size: {len(self.val_dataset)}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating dummy data for testing...")

            dummy_data = [
                {
                    "translation": {
                        self.src_lang: f"Hello world {i}",
                        self.tgt_lang: f"Bonjour monde {i}",
                    }
                }
                for i in range(100)
            ]

            self.train_dataset = TranslationDataset(
                dummy_data[:80],
                self.src_tokenizer,
                self.tgt_tokenizer,
                self.max_len,
                self.src_lang,
                self.tgt_lang,
            )

            self.val_dataset = TranslationDataset(
                dummy_data[80:],
                self.src_tokenizer,
                self.tgt_tokenizer,
                self.max_len,
                self.src_lang,
                self.tgt_lang,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def main():
    print("Initializing Translation Task...")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        print("Set CUDA matmul precision to medium for better Tensor Core utilization")

    data_module = TranslationDataModule(
        dataset_name="opus_books",
        language_pair="en-fr",
        batch_size=16,
        max_len=128,
        num_workers=2,
        train_size=5000,
        val_size=500,
    )

    data_module.setup()

    pad_token_id = data_module.src_tokenizer.convert_tokens_to_ids("<pad>")
    bos_token_id = data_module.tgt_tokenizer.convert_tokens_to_ids("<bos>")
    eos_token_id = data_module.tgt_tokenizer.convert_tokens_to_ids("<eos>")

    model = TranslationModel(
        src_vocab_size=len(data_module.src_tokenizer),
        tgt_vocab_size=len(data_module.tgt_tokenizer),
        dim_model=512,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_ff=1024,
        max_seq_len=128,
        dropout=0.1,
        learning_rate=1e-4,
        warmup_steps=1000,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        label_smoothing=0.1,
    )

    wandb_logger = WandbLogger(
        project="ml-architectures-lab",
        name="en-fr-translation",
        tags=["transformer", "translation", "en-fr"],
        log_model=True,
        save_dir="/workspaces/ml-architectures-lab/lab/logs",
    )

    wandb_logger.log_hyperparams(
        {
            "src_vocab_size": len(data_module.src_tokenizer),
            "tgt_vocab_size": len(data_module.tgt_tokenizer),
            "dim_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_ff": 1024,
            "max_seq_len": 128,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "label_smoothing": 0.1,
            "batch_size": 16,
            "max_epochs": 5,
        }
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        callbacks=[],
    )

    print("Starting training...")

    trainer.fit(model, data_module)

    print("Training completed!")

    model_path = "/workspaces/ml-architectures-lab/lab/models/translation_model.ckpt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    trainer.save_checkpoint(model_path)
    print(f"Model saved to {model_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
