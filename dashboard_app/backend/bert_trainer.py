"""
bert_trainer.py — DistilBERT fine-tuning engine for Case Insights.
Trains distilbert-base-uncased on labeled support cases.
Auto-detects Apple Silicon MPS → CUDA → CPU.
"""

import os, json, threading
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class _TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTTrainer:
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.label2id: dict = {}
        self.id2label: dict = {}

        # Status (thread-safe via GIL for simple int/str reads)
        self.status = "idle"          # idle | training | ready | error
        self.progress = 0             # 0–100
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss = None
        self.error_msg = None
        self.is_trained = os.path.exists(os.path.join(save_dir, "config.json"))

        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self._device_name = "Apple MPS"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self._device_name = "CUDA GPU"
        else:
            self.device = torch.device("cpu")
            self._device_name = "CPU"
        print(f"BERTTrainer ready — device: {self._device_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_training_async(self, texts: list, labels: list,
                             epochs: int = 3, batch_size: int = 16):
        """Start training in a background thread. Returns immediately."""
        if self.status == "training":
            return {"status": "already_training"}
        t = threading.Thread(
            target=self._train_safe, args=(texts, labels, epochs, batch_size), daemon=True
        )
        t.start()
        return {"status": "started", "device": self._device_name}

    def predict(self, texts: list, batch_size: int = 32):
        """
        Run inference. Returns (labels, confidences, top3_list).
        Each top3 item: [{"category": str, "confidence": float}, ...]
        """
        if self.model is None:
            self.load()
        self.model.eval()

        all_labels, all_probs, all_top3 = [], [], []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, padding=True,
                                 max_length=256, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attn_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                best_i = int(np.argmax(row))
                top3_idx = np.argsort(row)[::-1][:3]
                all_labels.append(self.id2label[best_i])
                all_probs.append(float(row[best_i]))
                all_top3.append([{"category": self.id2label[ti],
                                   "confidence": round(float(row[ti]), 3)}
                                  for ti in top3_idx])
        return all_labels, all_probs, all_top3

    def load(self):
        """Load a previously saved model from disk."""
        lm_path = os.path.join(self.save_dir, "label_map.json")
        if not os.path.exists(lm_path):
            raise FileNotFoundError("No trained BERT model found at " + self.save_dir)
        with open(lm_path) as f:
            maps = json.load(f)
        self.label2id = maps["label2id"]
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.save_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.save_dir)
        self.model.to(self.device)
        self.model.eval()
        self.status = "ready"
        self.is_trained = True
        print("BERT model loaded from disk.")

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "is_trained": self.is_trained,
            "device": self._device_name,
            "train_loss": self.train_loss,
            "error": self.error_msg,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_safe(self, texts, labels, epochs, batch_size):
        """Wrapper that catches exceptions and updates status."""
        try:
            self._train(texts, labels, epochs, batch_size)
        except Exception as e:
            import traceback
            self.error_msg = str(e)
            self.status = "error"
            traceback.print_exc()

    def _train(self, texts: list, labels: list, epochs: int, batch_size: int):
        from collections import Counter
        self.status = "training"
        self.progress = 0
        self.total_epochs = epochs
        self.error_msg = None

        # Filter rare classes (need ≥ 2 examples)
        counts = Counter(labels)
        pairs = [(t, l) for t, l in zip(texts, labels) if counts[l] >= 2]
        if len(pairs) < 20:
            raise ValueError(f"Only {len(pairs)} usable samples after filtering. Need more labeled data.")
        texts, labels = zip(*pairs)
        texts, labels = list(texts), list(labels)

        # Build label maps
        unique = sorted(set(labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        label_ids = [self.label2id[l] for l in labels]
        num_labels = len(unique)

        print(f"BERT: {len(texts)} samples, {num_labels} classes, {epochs} epochs, batch={batch_size}, device={self.device}")

        # Load base model
        print("Loading DistilBERT from HuggingFace (cached after first run)...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME, num_labels=num_labels,
            id2label=self.id2label, label2id=self.label2id
        )
        self.model.to(self.device)

        # Tokenize (truncate long texts)
        print("Tokenizing dataset...")
        enc = self.tokenizer(texts, truncation=True, padding=True,
                             max_length=256, return_tensors="pt")
        dataset = _TextDataset(enc, label_ids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer + LR scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps
        )

        global_step = 0
        self.model.train()
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                loss = self.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels_t).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1
                self.progress = int((global_step / total_steps) * 94)

                if batch_idx % 25 == 0:
                    print(f"  Epoch {self.current_epoch}/{epochs} step {batch_idx}/{len(loader)} loss={loss.item():.4f}")

            avg = epoch_loss / len(loader)
            self.train_loss = round(avg, 4)
            print(f"Epoch {self.current_epoch} avg loss: {avg:.4f}")

        # Save
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        with open(os.path.join(self.save_dir, "label_map.json"), "w") as f:
            json.dump({"label2id": self.label2id,
                       "id2label": {str(k): v for k, v in self.id2label.items()}}, f, indent=2)
        self.is_trained = True
        self.status = "ready"
        self.progress = 100
        print(f"✓ BERT training complete. Model saved to {self.save_dir}")
