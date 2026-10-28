"""Load the checkpoint's Rust tokenizer without importing Transformers or torch."""

import json
from pathlib import Path

from tokenizers import Tokenizer as Backend


class Tokenizer:
    def __init__(self, path: Path):
        path = Path(path)
        self.backend = Backend.from_file(str(path / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.cls_token = config.get("cls_token", "[CLS]")
        self.sep_token = config.get("sep_token", "[SEP]")
        self.mask_token = config.get("mask_token", "[MASK]")
        self.pad_token = config.get("pad_token", "[PAD]")
        vocab = self.backend.get_vocab()
        self.cls_token_id = vocab[self.cls_token]
        self.sep_token_id = vocab[self.sep_token]
        self.mask_token_id = vocab[self.mask_token]
        self.pad_token_id = vocab.get(self.pad_token, 0)

    def __call__(self, text: str, add_special_tokens: bool = False) -> list:
        enc = self.backend.encode(text, add_special_tokens=add_special_tokens)
        return enc.ids
