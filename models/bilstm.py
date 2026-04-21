import os
import json
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from box import Box
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from tqdm import tqdm


class BiLSTMClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 200,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 embed_matrix: np.ndarray | None = None,
                 freeze_embed: bool = False):
        super().__init__()
 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embed_matrix, dtype=torch.float32),
                requires_grad=not freeze_embed
            )
 
        self.spatial_dropout = nn.Dropout2d(dropout)
 
        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout        = dropout if num_layers > 1 else 0.0,
        )
 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x) # (batch, seq, embed_dim)
        emb = emb.unsqueeze(2) # (batch, seq, 1, embed_dim)
        emb = self.spatial_dropout(emb).squeeze(2) # spatial dropout over seq dim
 
        out, (hn, _) = self.lstm(emb)
        # hn: (num_layers * 2, batch, hidden_dim)
        # Concat the final forward and backward hidden states of the last layer
        fwd = hn[-2]   # last layer, forward
        bwd = hn[-1]   # last layer, backward
        combined = torch.cat([fwd, bwd], dim=1) # (batch, hidden_dim * 2)
 
        return self.classifier(combined)


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

GLOVE_PATH    = "/home/gpuhead-2/data_mining/DataMiningCourseProject/data/glove.twitter.27B.200d.txt"
MAX_LEN = 64
VOCAB_SIZE = 60_000
 
class Vocabulary:
    def __init__(self, max_size: int = VOCAB_SIZE):
        self.max_size   = max_size
        self.word2idx   = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word   = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._freq: dict = {}
 
    def build(self, texts: list[str]):
        for text in texts:
            for token in text.split():
                self._freq[token] = self._freq.get(token, 0) + 1
 
        most_common = sorted(self._freq, key=self._freq.get, reverse=True)
        most_common  = most_common[: self.max_size - 2]   # reserve PAD & UNK
 
        for word in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
 
        print(f"[Vocabulary] size = {len(self.word2idx):,}")
 
    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]   # 1 = UNK
        # pad / truncate
        ids   += [0] * (max_len - len(ids))
        return ids
 
    def __len__(self):
        return len(self.word2idx)


def load_glove(glove_path: str, vocab: Vocabulary, embed_dim: int = 200) -> np.ndarray:
    """Load GloVe vectors for tokens that appear in our vocabulary."""
    print(f"[GloVe] Loading from {glove_path} …")
    matrix = np.random.normal(scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = 0.0   # PAD → all zeros
 
    found = 0
    with open(glove_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab.word2idx:
                matrix[vocab.word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
 
    coverage = found / (len(vocab) - 2) * 100   # exclude PAD & UNK
    print(f"[GloVe] Covered {found:,} / {len(vocab)-2:,} vocab tokens ({coverage:.1f} %)")
    return matrix


class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary):
        self.X = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.y = [torch.tensor(lbl, dtype=torch.long)             for lbl in labels]
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_bilstm(config: Box):
    return BiLSTMClassifier(vocab_size=VOCAB_SIZE)


def run_train_epoch(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

@torch.inference_mode()
def collect_predictions(model, loader, device) -> tuple[list, list, list]:
    """Return (labels, preds, probs) over a full DataLoader."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        probs   = F.softmax(logits, dim=1).cpu().tolist()
        preds   = logits.argmax(dim=1).cpu().tolist()
        all_labels.extend(y_batch.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs)
    return all_labels, all_preds, all_probs

def get_metrics(labels: list, preds: list, probs: list) -> dict:
    """Mirrors the logreg get_metrics() structure exactly."""
    acc               = accuracy_score(labels, preds)
    prec, rec, f1, _  = precision_recall_fscore_support(labels, preds,
                                                         average="macro",
                                                         zero_division=0)
    loss              = log_loss(labels, probs)
    return {
        "loss":      float(loss),
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1_macro":  float(f1),
    }

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
 
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
 
    return total_loss / total, correct / total

@torch.inference_mode()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y_batch.cpu().tolist())
 
    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    return total_loss / n, acc, prec, rec, f1_macro, all_preds, all_labels

def train_bilstm(model,
                 config,
                 dataset):
    print(f"USING: {config.device}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if config.dataset == "twitter_partisan":
        stratify_key = dataset['source'] + "_" + dataset['party']
    elif config.dataset == "mbib":
        stratify_key = dataset["label"]
    
    run_data = {
        "metadata": {
            "model_type": "BiLSTM",
            "random_state": RANDOM_STATE,
            "config": dict(config)
        },
        "folds": [],
        "final_summary": {}
    }
    
    best_overall_f1    = -1.0
    best_overall_state = None
    best_vocab         = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, stratify_key)):
        print(f"Processing Fold {fold + 1}...")
        fold_start = time.perf_counter()

        train_set = dataset.iloc[train_idx]
        val_set = dataset.iloc[val_idx]

        vocab = Vocabulary(max_size=VOCAB_SIZE)
        vocab.build(train_set["text"].tolist())

        embed_matrix = load_glove(GLOVE_PATH, vocab, config.embed.embed_dim)

        train_ds = TweetDataset(train_set["text"].tolist(), train_set["label"].tolist(), vocab)
        val_ds   = TweetDataset(val_set["text"].tolist(), val_set["label"].tolist(), vocab)

        train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)


        model = BiLSTMClassifier(
            vocab_size   = len(vocab),
            embed_dim    = config.embed.embed_dim,
            hidden_dim   = config.model.hidden_dim,
            num_layers   = config.model.num_layers,
            num_classes  = config.model.num_classes,
            dropout      = config.model.dropout,
            embed_matrix = embed_matrix,
            freeze_embed = False,
        ).to(config.device)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=config.training.patience)


        train_start = time.perf_counter()
        best_val_f1 = -1.0
        best_state  = None
        no_improve  = 0

        for epoch in tqdm(range(1, 101)):
            # train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config.device)
            # val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = eval_epoch(model, val_loader, criterion, config.device)
            run_train_epoch(model, train_loader, optimizer, criterion, config.device)

            vl_labels, vl_preds, vl_probs = collect_predictions(model, val_loader, config.device)
            val_f1 = get_metrics(vl_labels, vl_preds, vl_probs)["f1_macro"]
            scheduler.step(val_f1)

            # _, _, train_prec, train_rec, train_f1, _, _ = eval_epoch(model, train_loader, criterion, config.device)
            # scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = copy.deepcopy(model.state_dict())
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= config.training.patience:
                    print(f"  [EarlyStopping] epoch {epoch} -- "
                          f"no improvement for {config.training.patience} epochs.")
                    break

        train_end = time.perf_counter()

        model.load_state_dict(best_state)

        tr_labels, tr_preds, tr_probs = collect_predictions(
            model, train_loader, config.device)
        vl_labels, vl_preds, vl_probs = collect_predictions(
            model, val_loader, config.device)
 
        fold_end = time.perf_counter()

        fold_entry = {
            "fold": fold + 1,
            "timings": {
                "training_seconds":   round(train_end - train_start, 4),
                "total_fold_seconds": round(fold_end  - fold_start,  4),
            },
            "train":      get_metrics(tr_labels, tr_preds, tr_probs),
            "validation": get_metrics(vl_labels, vl_preds, vl_probs),
        }
        run_data["folds"].append(fold_entry)
 
        print(f"  val_f1={fold_entry['validation']['f1_macro']:.4f}  "
              f"val_loss={fold_entry['validation']['loss']:.4f}  "
              f"({fold_entry['timings']['training_seconds']:.1f}s train)")
 
        # keep the single best model state across all folds
        if best_val_f1 > best_overall_f1:
            best_overall_f1    = best_val_f1
            best_overall_state = copy.deepcopy(best_state)
            best_vocab         = vocab
    
    val_f1s    = [f["validation"]["f1_macro"] for f in run_data["folds"]]
    val_losses = [f["validation"]["loss"]     for f in run_data["folds"]]
 
    run_data["final_summary"] = {
        "mean_val_f1":   float(np.mean(val_f1s)),
        "std_val_f1":    float(np.std(val_f1s)),
        "mean_val_loss": float(np.mean(val_losses)),
    }
 
    print(f"\nMean val F1 : {run_data['final_summary']['mean_val_f1']:.4f} "
          f"+/- {run_data['final_summary']['std_val_f1']:.4f}")

    log_name = (config.config_path.split("/")[-1]
                .replace("config", "bilstm_train")
                .replace("yaml",   "json"))
    log_dir  = os.path.join(config.log_dir, "train")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    print(f"Writing logs to {log_path}")
    with open(log_path, "w") as fh:
        json.dump(run_data, fh, indent=4)
 
    # -- checkpoint (best-fold weights + vocab) -------------------------------
    checkpoint_name = (config.config_path.split("/")[-1]
                       .replace("config", "bilstm_train")
                       .replace("yaml",   "pt"))
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    torch.save({"model_state": best_overall_state, "vocab": best_vocab},
               checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
 
    # return run_data

def test_bilstm(config,
                dataset):
    checkpoint_name = (config.config_path.split("/")[-1]
                       .replace("config", "bilstm_train")
                       .replace("yaml",   "pt"))
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_state = ckpt["model_state"]
    vocab = ckpt["vocab"]
    embed_matrix = load_glove(GLOVE_PATH, vocab, config.embed.embed_dim)

    model = BiLSTMClassifier(
            vocab_size   = len(vocab),
            embed_dim    = config.embed.embed_dim,
            hidden_dim   = config.model.hidden_dim,
            num_layers   = config.model.num_layers,
            num_classes  = config.model.num_classes,
            dropout      = config.model.dropout,
            embed_matrix = embed_matrix,
            freeze_embed = False,
        ).to(config.device)
    model.load_state_dict(model_state)

    test_ds = TweetDataset(dataset["text"].tolist(), dataset["label"].tolist(), vocab)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False,  num_workers=2, pin_memory=True)

    test_labels, test_preds, test_probs = collect_predictions(model, test_loader, config.device)
    test_metrics = get_metrics(test_labels, test_preds, test_probs)

    log_dir  = os.path.join(config.log_dir, "test")
    os.makedirs(log_dir, exist_ok=True)

    log_name = (config.config_path.split("/")[-1]
                .replace("config", "bilstm_test")
                .replace("yaml",   "json"))
    log_path = os.path.join(log_dir, log_name)
    print(f"Writing logs to {log_path}")
    with open(log_path, "w") as fh:
        json.dump(test_metrics, fh, indent=4)