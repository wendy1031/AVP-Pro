import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = "/data/wenxinr/AVP-Fusion experiments"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_BASE_DIR, "esm2_t30_150M_UR50D")

for path_item in [str(SCRIPT_DIR), DEFAULT_BASE_DIR, str(SCRIPT_DIR.parent)]:
    if path_item not in sys.path:
        sys.path.insert(0, path_item)

import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from loss_light_ohem import ContrastiveLoss
from util.Queue_ohem import OHEMQueue
from util.blosum62_probabilistic import augment_sequence_with_second_best_mutation
from util.data import esm_encode, generate_features, load_data
from util.focal_loss import FocalLoss
from util.seed import set_seed

AVP_Fusion_v3 = getattr(importlib.import_module("model_att"), "AVP_" + "H" + "NCL_v3")

RANDOM_SEED = 1064
N_FOLDS = 5
ESM_DIM = 640
CNN_OUT_CHANNELS = 256
LSTM_HIDDEN_DIM = 256
NUM_CLASSES = 2
DROPOUT_RATE = 0.45
MAX_LENGTH = 100
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1.2e-4
WEIGHT_DECAY = 1e-2
QUEUE_SIZE = 3000
K_HARD_NEGATIVES = 10
GRAD_CLIP_NORM = 1.0
NUM_FRAGMENTS = 6
MUTATION_RATE = 0.6
INSERTION_RATE = 0.5
DELETION_RATE = 0.5
MULTI_STEP = 1
CONTRASTIVE_WEIGHT = 0.10
CONSISTENCY_WEIGHT = 0.05
CONSISTENCY_TEMP = 2.0
USE_RETRIEVAL = True
KNN_K = 8
RETR_FUSE_WEIGHT = 1.0
RETR_LOSS_WEIGHT = 0.04
RETR_ATT_TAU = 0.7
USE_MIL = True
KMER = 5
MIL_FUSE_WEIGHT = 1.0
MIL_SPARSITY = 5e-4
PATIENCE = 7


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SequenceDataset(Dataset):
    def __init__(self, sequences, features, labels):
        self.sequences = list(sequences)
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.features[idx], self.labels[idx]


def symmetric_kl(logits_a, logits_b, temperature=1.0):
    p_log = F.log_softmax(logits_a / temperature, dim=1)
    q_log = F.log_softmax(logits_b / temperature, dim=1)
    p = p_log.exp()
    q = q_log.exp()
    return 0.5 * (F.kl_div(p_log, q, reduction="batchmean") + F.kl_div(q_log, p, reduction="batchmean"))


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sn = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "gmean": float(np.sqrt(sn * sp)),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "f1": float(f1_score(labels, preds)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def find_best_threshold(labels, probs):
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in np.linspace(0.05, 0.95, 19):
        preds = (np.asarray(probs) >= threshold).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return float(best_threshold), float(best_mcc)


@torch.no_grad()
def update_negative_scores(pos_queue, neg_queue, batch_size, device):
    pos_emb = pos_queue.get_all_embeddings()
    neg_emb = neg_queue.get_all_embeddings()
    if pos_emb is None or neg_emb is None or pos_emb.shape[0] == 0 or neg_emb.shape[0] == 0:
        neg_queue.difficulty_scores.zero_()
        return
    proto = torch.mean(pos_emb, dim=0, keepdim=True)
    scores = torch.zeros(neg_emb.shape[0], device=device)
    for i in range(0, neg_emb.shape[0], batch_size):
        scores[i:i + batch_size] = F.cosine_similarity(neg_emb[i:i + batch_size], proto)
    if neg_queue.is_full():
        neg_queue.difficulty_scores[:] = scores
    else:
        neg_queue.difficulty_scores[:neg_queue.ptr] = scores


class RetrievalAugmentor(nn.Module):
    def __init__(self, esm_dim, model_emb_dim, k=8, att_tau=1.0):
        super().__init__()
        self.k = k
        self.att_tau = att_tau
        self.proj_neigh = nn.Linear(esm_dim, model_emb_dim)
        self.clf_fuse = nn.Linear(model_emb_dim * 2, NUM_CLASSES)
        self.register_buffer("index_emb", torch.empty(0))
        self.register_buffer("index_labels", torch.empty(0, dtype=torch.long))
        self.seq2idxs = {}

    @torch.no_grad()
    def build_index_from_esm(self, sequences, labels, esm_model, tokenizer, device):
        embs = []
        for i in range(0, len(sequences), 64):
            inputs = tokenizer(sequences[i:i + 64], return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to(device)
            pooled = esm_model(**inputs).last_hidden_state.mean(dim=1).float()
            embs.append(F.normalize(pooled, dim=1).detach().cpu())
        self.index_emb = torch.cat(embs, dim=0)
        self.index_labels = torch.tensor(labels, dtype=torch.long)
        self.seq2idxs = {}
        for idx, seq in enumerate(sequences):
            self.seq2idxs.setdefault(seq, []).append(idx)

    @torch.no_grad()
    def rebuild_index_from_model(self, loader, model, device, tokenizer, esm_model):
        model.eval()
        all_embs = []
        all_labels = []
        all_seqs = []
        for batch_seqs, batch_features, batch_y in tqdm(loader, desc="Rebuild retrieval index"):
            features_t = torch.as_tensor(batch_features, device=device)
            esm_tokens = esm_encode(list(batch_seqs), esm_model, tokenizer, device, max_length=MAX_LENGTH)
            _, embs = model(esm_tokens, features_t)
            all_embs.append(F.normalize(embs.float(), dim=1).detach().cpu())
            all_labels.append(torch.as_tensor(batch_y, dtype=torch.long))
            all_seqs.extend(list(batch_seqs))
        self.index_emb = torch.cat(all_embs, dim=0)
        self.index_labels = torch.cat(all_labels, dim=0)
        self.seq2idxs = {}
        for idx, seq in enumerate(all_seqs):
            self.seq2idxs.setdefault(seq, []).append(idx)

    def forward(self, query_sequences, query_esm_pooled, model_embs, device):
        with torch.cuda.amp.autocast(enabled=False):
            index = self.index_emb.to(device).float()
            query = F.normalize(model_embs.float(), dim=1) if index.shape[1] == model_embs.shape[1] else F.normalize(query_esm_pooled.float(), dim=1)
            sims = torch.matmul(query, index.t())
            for b, seq in enumerate(query_sequences):
                for idx in self.seq2idxs.get(seq, []):
                    sims[b, idx] = -1e4
            topk = torch.topk(sims, k=min(self.k, sims.shape[1]), dim=1).indices
            neigh_raw = self.index_emb.to(device)[topk]
            neigh_lbl = self.index_labels.to(device)[topk]
        neigh_proj = neigh_raw if neigh_raw.shape[-1] == model_embs.shape[1] else self.proj_neigh(neigh_raw)
        att = torch.softmax(torch.matmul(model_embs.unsqueeze(1), neigh_proj.transpose(1, 2)) / ((model_embs.shape[1] ** 0.5) * self.att_tau), dim=-1)
        z_neigh = torch.matmul(att, neigh_proj).squeeze(1)
        logits_retr = self.clf_fuse(torch.cat([model_embs, z_neigh], dim=1))
        with torch.no_grad():
            pos_mask = (neigh_lbl == 1).float()
            neg_mask = (neigh_lbl == 0).float()
            pos_proto = (neigh_proj * pos_mask.unsqueeze(-1)).sum(1) / (pos_mask.sum(1, keepdim=True) + 1e-6)
            neg_proto = (neigh_proj * neg_mask.unsqueeze(-1)).sum(1) / (neg_mask.sum(1, keepdim=True) + 1e-6)
        retr_align_loss = F.mse_loss(model_embs, pos_proto, reduction="mean") - 0.5 * F.mse_loss(model_embs, neg_proto, reduction="mean")
        return logits_retr, retr_align_loss


class ConvKmerMIL(nn.Module):
    def __init__(self, esm_dim, model_emb_dim, kmer=5):
        super().__init__()
        self.conv = nn.Conv1d(esm_dim, esm_dim, kernel_size=kmer, stride=1, padding=kmer // 2, bias=False)
        nn.init.dirac_(self.conv.weight)
        self.proj = nn.Linear(esm_dim, model_emb_dim)
        self.att = nn.Linear(model_emb_dim, 1)
        self.clf_fuse = nn.Linear(model_emb_dim * 2, NUM_CLASSES)

    def forward(self, esm_tokens, model_embs):
        kmer_proj = self.proj(self.conv(esm_tokens.transpose(1, 2)).transpose(1, 2))
        weights = torch.softmax(self.att(kmer_proj).squeeze(-1), dim=1)
        z_mil = (weights.unsqueeze(-1) * kmer_proj).sum(1)
        logits_mil = self.clf_fuse(torch.cat([model_embs, z_mil], dim=1))
        sparsity = (weights * torch.log(weights + 1e-12)).sum(dim=1).mean()
        return logits_mil, sparsity


class Stage1Runner:
    def __init__(self, dataset, base_dir, esm_model_path, device):
        self.dataset = dataset
        self.base_dir = base_dir
        self.esm_model_path = esm_model_path
        self.device = device
        if dataset == "set1":
            self.dataset_name = "Set1-nonAVP"
            self.set_dir = os.path.join(base_dir, "dataset/Set 1")
        elif dataset == "set2":
            self.dataset_name = "Set2-nonAMP"
            self.set_dir = os.path.join(base_dir, "dataset/Set 2")
        else:
            raise ValueError("dataset must be set1 or set2")
        self.train_file = os.path.join(self.set_dir, "train.txt")
        self.test_file = os.path.join(self.set_dir, "test.txt")
        self.output_dir = os.path.join(self.set_dir, "AVP_Fusion_stage1_final")
        self.cv_dir = os.path.join(self.output_dir, "fivefold_validation")
        self.final_dir = os.path.join(self.output_dir, "benchmark_final")
        os.makedirs(self.cv_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        self.esm_model = None
        self.tokenizer = None

    def load_resources(self):
        self.esm_model = AutoModel.from_pretrained(self.esm_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm_model_path)
        self.esm_model.eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False
        pos_train, neg_train = load_data(self.train_file)
        pos_test, neg_test = load_data(self.test_file)
        self.train_sequences = pos_train + neg_train
        self.test_sequences = pos_test + neg_test
        self.y_train = np.array([1] * len(pos_train) + [0] * len(neg_train), dtype=np.int64)
        self.y_test = np.array([1] * len(pos_test) + [0] * len(neg_test), dtype=np.int64)
        self.train_features = np.asarray(generate_features(self.train_file).values, dtype=np.float32)
        self.test_features = np.asarray(generate_features(self.test_file).values, dtype=np.float32)
        print(f"Dataset: {self.dataset_name}")
        print(f"Official train size: {len(self.train_sequences)}")
        print(f"Official test size: {len(self.test_sequences)}")
        print(f"Official train positives/negatives: {int(np.sum(self.y_train == 1))}/{int(np.sum(self.y_train == 0))}")
        print(f"Official test positives/negatives: {int(np.sum(self.y_test == 1))}/{int(np.sum(self.y_test == 0))}")

    def preprocess_features(self, train_features, eval_features):
        scaler = StandardScaler()
        train_std = scaler.fit_transform(np.log1p(np.exp(np.clip(train_features, -10, 10))))
        eval_std = [scaler.transform(np.log1p(np.exp(np.clip(x, -10, 10)))) for x in eval_features]
        return train_std, eval_std

    def loader(self, sequences, features, labels, shuffle):
        return DataLoader(SequenceDataset(sequences, features, labels), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)

    def build_modules(self, feature_dim, train_sequences, train_labels):
        model = AVP_Fusion_v3(
            esm_dim=ESM_DIM,
            additional_dim=feature_dim,
            cnn_out_channels=CNN_OUT_CHANNELS,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        ).to(self.device)
        model_emb_dim = getattr(model, "embedding_dim", 512)
        contrastive = ContrastiveLoss(temperature=0.5, learnable_temperature=True, regularization=1e-4).to(self.device)
        n0 = int(np.sum(train_labels == 0))
        n1 = int(np.sum(train_labels == 1))
        n = len(train_labels)
        alpha = torch.tensor([n / (2.0 * max(n0, 1)), (n / (2.0 * max(n1, 1))) * 1.5], dtype=torch.float32, device=self.device)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        retr = RetrievalAugmentor(ESM_DIM, model_emb_dim, k=KNN_K, att_tau=RETR_ATT_TAU).to(self.device) if USE_RETRIEVAL else None
        mil = ConvKmerMIL(ESM_DIM, model_emb_dim, kmer=KMER).to(self.device) if USE_MIL else None
        if retr is not None:
            retr.build_index_from_esm(train_sequences, train_labels, self.esm_model, self.tokenizer, self.device)
        params = list(model.parameters())
        if retr is not None:
            params += list(retr.parameters())
        if mil is not None:
            params += list(mil.parameters())
        optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        return model, model_emb_dim, contrastive, criterion, retr, mil, params, optimizer, GradScaler()

    def train_once(self, run_name, run_dir, train_sequences, train_features, train_labels, val_sequences=None, val_features=None, val_labels=None, fixed_epochs=None, external_threshold=None):
        os.makedirs(run_dir, exist_ok=True)
        best_model_path = os.path.join(run_dir, "best_model.pth")
        epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
        eval_features = [self.test_features] if val_sequences is None else [val_features, self.test_features]
        train_std, eval_std = self.preprocess_features(train_features, eval_features)
        val_std = None if val_sequences is None else eval_std[0]
        test_std = eval_std[0] if val_sequences is None else eval_std[1]
        train_loader = self.loader(train_sequences, train_std, train_labels, True)
        test_loader = self.loader(self.test_sequences, test_std, self.y_test, False)
        val_loader = None if val_sequences is None else self.loader(val_sequences, val_std, val_labels, False)
        model, model_emb_dim, contrastive, criterion, retr, mil, params, optimizer, scaler = self.build_modules(train_std.shape[1], train_sequences, train_labels)
        run_epochs = int(fixed_epochs) if fixed_epochs is not None else EPOCHS
        total_steps = max(1, len(train_loader) * run_epochs)
        warmup_steps = len(train_loader) * 5

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        def model_forward(tokens, features_t):
            if model.training and tokens.size(0) == 1:
                logits2, embs2 = model(torch.cat([tokens, tokens], dim=0), torch.cat([features_t, features_t], dim=0))
                return logits2[:1], embs2[:1]
            return model(tokens, features_t)

        def forward_fusion(batch_seqs, esm_tokens, features_t, training):
            logits_base, embs = model_forward(esm_tokens, features_t)
            logits = logits_base
            extra_loss = 0.0
            if retr is not None:
                q_pooled = esm_tokens.mean(dim=1).float()
                logits_retr, retr_loss = retr(list(batch_seqs), q_pooled, embs, self.device)
                logits = logits + RETR_FUSE_WEIGHT * logits_retr
                if training:
                    extra_loss = extra_loss + RETR_LOSS_WEIGHT * retr_loss
            if mil is not None:
                logits_mil, sparsity = mil(esm_tokens, embs)
                logits = logits + MIL_FUSE_WEIGHT * logits_mil
                if training:
                    extra_loss = extra_loss + MIL_SPARSITY * sparsity
            return logits, embs, extra_loss

        @torch.no_grad()
        def evaluate(dataloader, desc, threshold=0.5):
            model.eval()
            labels = []
            probs = []
            running_loss = 0.0
            total = 0
            for batch_seqs, batch_features, batch_y in tqdm(dataloader, desc=desc):
                features_t = torch.as_tensor(batch_features, device=self.device)
                y_t = torch.as_tensor(batch_y, device=self.device)
                esm_tokens = esm_encode(list(batch_seqs), self.esm_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                with autocast():
                    logits, _, _ = forward_fusion(batch_seqs, esm_tokens, features_t, False)
                    loss = criterion(logits, y_t)
                probs.extend(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
                labels.extend(y_t.detach().cpu().numpy())
                running_loss += float(loss.item()) * y_t.size(0)
                total += y_t.size(0)
            labels = np.asarray(labels)
            probs = np.asarray(probs)
            metrics = compute_metrics(labels, probs, threshold)
            metrics["loss"] = float(running_loss / max(total, 1))
            return metrics, labels, probs

        pos_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=model_emb_dim).to(self.device)
        neg_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=model_emb_dim).to(self.device)
        model.eval()
        with torch.no_grad():
            for batch_seqs, batch_features, batch_y in tqdm(train_loader, desc=f"{run_name} queue initialization"):
                features_t = torch.as_tensor(batch_features, device=self.device)
                y_t = torch.as_tensor(batch_y, device=self.device)
                esm_tokens = esm_encode(list(batch_seqs), self.esm_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                _, embs = model(esm_tokens, features_t)
                pos_idx = (y_t == 1).nonzero(as_tuple=True)[0]
                neg_idx = (y_t == 0).nonzero(as_tuple=True)[0]
                if len(pos_idx) > 0:
                    pos_queue.enqueue(embs[pos_idx].to(torch.float32))
                if len(neg_idx) > 0:
                    neg_queue.enqueue(embs[neg_idx].to(torch.float32))
                if pos_queue.is_full() and neg_queue.is_full():
                    break

        best_mcc = -1.0
        best_epoch = 0
        best_threshold = 0.5
        patience_count = 0
        epoch_rows = []
        fixed_mode = fixed_epochs is not None
        if fixed_mode and external_threshold is None:
            raise ValueError("external_threshold is required for final fixed-epoch training")

        for epoch in range(run_epochs):
            if retr is not None and epoch in (1, 3, 5, 8):
                retr.rebuild_index_from_model(train_loader, model, self.device, self.tokenizer, self.esm_model)
            update_negative_scores(pos_queue, neg_queue, BATCH_SIZE, self.device)
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for batch_seqs, batch_features, batch_y in tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}"):
                features_t = torch.as_tensor(batch_features, device=self.device)
                y_t = torch.as_tensor(batch_y, device=self.device)
                pos_idx = (y_t == 1).nonzero(as_tuple=True)[0]
                pos_seqs = [batch_seqs[i] for i in pos_idx.tolist()] if len(pos_idx) > 0 else []
                aug_sequences = [augment_sequence_with_second_best_mutation(seq, NUM_FRAGMENTS, MUTATION_RATE, INSERTION_RATE, DELETION_RATE, MULTI_STEP) for seq in pos_seqs]
                esm_tokens = esm_encode(list(batch_seqs), self.esm_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                esm_tokens_aug = esm_encode(aug_sequences, self.esm_model, self.tokenizer, self.device, max_length=MAX_LENGTH) if len(aug_sequences) > 0 else None
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits_total, embs, extra_loss = forward_fusion(batch_seqs, esm_tokens, features_t, True)
                    cls_loss = criterion(logits_total, y_t)
                    con_loss = 0.0
                    if len(pos_idx) > 0 and esm_tokens_aug is not None and neg_queue.size() > K_HARD_NEGATIVES:
                        _, embs_aug = model_forward(esm_tokens_aug, features_t[pos_idx])
                        scores = neg_queue.difficulty_scores[:neg_queue.size()]
                        consider = min(len(scores), K_HARD_NEGATIVES * 10)
                        _, topk = torch.topk(scores, k=consider, largest=True)
                        hard_idx = topk[torch.randperm(topk.size(0), device=self.device)[:K_HARD_NEGATIVES]]
                        con_loss = contrastive(embs[pos_idx], embs_aug, neg_queue.embeddings[hard_idx])
                    pos_all = (y_t == 1).nonzero(as_tuple=True)[0]
                    neg_all = (y_t == 0).nonzero(as_tuple=True)[0]
                    if len(pos_all) > 0:
                        pos_queue.enqueue(embs[pos_all].detach().to(torch.float32))
                    if len(neg_all) > 0:
                        neg_queue.enqueue(embs[neg_all].detach().to(torch.float32))
                    cons_loss = 0.0
                    if len(pos_idx) > 0 and esm_tokens_aug is not None:
                        logits_aug, _, _ = forward_fusion(pos_seqs, esm_tokens_aug, features_t[pos_idx], False)
                        cons_loss = symmetric_kl(logits_total[pos_idx], logits_aug, temperature=CONSISTENCY_TEMP)
                    loss = cls_loss + CONTRASTIVE_WEIGHT * con_loss + CONSISTENCY_WEIGHT * cons_loss + extra_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += float(loss.item()) * y_t.size(0)
                pred = torch.argmax(logits_total, dim=1)
                total += y_t.size(0)
                correct += int((pred == y_t).sum().item())
            row = {"epoch": epoch + 1, "train_loss": running_loss / max(total, 1), "train_accuracy": correct / max(total, 1)}
            if not fixed_mode:
                metrics_05, val_labels_current, val_probs_current = evaluate(val_loader, f"{run_name} validation", 0.5)
                selected_threshold, selected_mcc = find_best_threshold(val_labels_current, val_probs_current)
                metrics_selected = compute_metrics(val_labels_current, val_probs_current, selected_threshold)
                row.update({
                    "validation_loss": metrics_05["loss"],
                    "validation_mcc_at_05": metrics_05["mcc"],
                    "validation_threshold": selected_threshold,
                    "validation_mcc": selected_mcc,
                    "validation_accuracy": metrics_selected["accuracy"],
                    "validation_sensitivity": metrics_selected["sensitivity"],
                    "validation_specificity": metrics_selected["specificity"],
                })
                if selected_mcc > best_mcc:
                    best_mcc = selected_mcc
                    best_epoch = epoch + 1
                    best_threshold = selected_threshold
                    checkpoint = {"model": model.state_dict(), "best_epoch": best_epoch, "best_threshold": best_threshold, "best_mcc": best_mcc}
                    if retr is not None:
                        checkpoint["retr"] = retr.state_dict()
                    if mil is not None:
                        checkpoint["mil"] = mil.state_dict()
                    torch.save(checkpoint, best_model_path)
                    patience_count = 0
                else:
                    patience_count += 1
                epoch_rows.append(row)
                pd.DataFrame(epoch_rows).to_csv(epoch_log_path, index=False)
                print(f"{run_name} epoch {epoch + 1}: train_loss={row['train_loss']:.4f}, val_mcc={selected_mcc:.4f}, threshold={selected_threshold:.3f}")
                if patience_count >= PATIENCE:
                    break
            else:
                epoch_rows.append(row)
                pd.DataFrame(epoch_rows).to_csv(epoch_log_path, index=False)
                print(f"{run_name} epoch {epoch + 1}: train_loss={row['train_loss']:.4f}, train_acc={row['train_accuracy']:.4f}")
                if epoch + 1 == run_epochs:
                    best_epoch = epoch + 1
                    best_threshold = float(external_threshold)
                    checkpoint = {"model": model.state_dict(), "best_epoch": best_epoch, "best_threshold": best_threshold}
                    if retr is not None:
                        checkpoint["retr"] = retr.state_dict()
                    if mil is not None:
                        checkpoint["mil"] = mil.state_dict()
                    torch.save(checkpoint, best_model_path)

        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        if retr is not None and "retr" in checkpoint:
            retr.load_state_dict(checkpoint["retr"])
        if mil is not None and "mil" in checkpoint:
            mil.load_state_dict(checkpoint["mil"])
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        best_threshold = float(checkpoint.get("best_threshold", best_threshold))
        if val_loader is not None:
            _, val_labels_final, val_probs_final = evaluate(val_loader, f"{run_name} final validation probabilities", 0.5)
            best_threshold, best_mcc = find_best_threshold(val_labels_final, val_probs_final)
            val_metrics = compute_metrics(val_labels_final, val_probs_final, best_threshold)
        else:
            val_labels_final = None
            val_probs_final = None
            val_metrics = None
        _, final_labels, final_probs = evaluate(test_loader, f"{run_name} benchmark evaluation", 0.5)
        final_metrics = compute_metrics(final_labels, final_probs, best_threshold)
        output = {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "validation_mcc": float(best_mcc) if val_loader is not None else float("nan"),
            "validation_metrics": val_metrics,
            "benchmark_metrics": final_metrics,
        }
        with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        pd.DataFrame([{
            "run_name": run_name,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            **{f"benchmark_{k}": v for k, v in final_metrics.items() if k != "confusion_matrix"},
        }]).to_csv(os.path.join(run_dir, "final_metrics.csv"), index=False)
        del model, retr, mil, optimizer, scaler, train_loader, test_loader, val_loader
        reset_memory()
        return output, val_labels_final, val_probs_final, pd.DataFrame(epoch_rows)

    def run_fivefold_validation(self):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_rows = []
        split_rows = []
        oof_probs = np.full(len(self.y_train), np.nan, dtype=np.float64)
        oof_labels = self.y_train.copy()
        oof_fold = np.full(len(self.y_train), -1, dtype=np.int64)
        best_epochs = []
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(self.train_sequences, self.y_train), start=1):
            fold_dir = os.path.join(self.cv_dir, f"fold_{fold_id}")
            train_seqs = [self.train_sequences[i] for i in train_idx]
            val_seqs = [self.train_sequences[i] for i in val_idx]
            result, val_labels, val_probs, epoch_df = self.train_once(
                run_name=f"fivefold_fold_{fold_id}",
                run_dir=fold_dir,
                train_sequences=train_seqs,
                train_features=self.train_features[train_idx],
                train_labels=self.y_train[train_idx],
                val_sequences=val_seqs,
                val_features=self.train_features[val_idx],
                val_labels=self.y_train[val_idx],
            )
            if len(val_probs) != len(val_idx) or not np.array_equal(val_labels, self.y_train[val_idx]):
                raise RuntimeError(f"Fold {fold_id} validation probability order mismatch")
            oof_probs[val_idx] = val_probs
            oof_fold[val_idx] = fold_id
            best_epochs.append(int(result["best_epoch"]))
            vm = result["validation_metrics"]
            fold_rows.append({
                "fold": fold_id,
                "best_epoch": int(result["best_epoch"]),
                "best_threshold": float(result["best_threshold"]),
                "val_accuracy": vm["accuracy"],
                "val_mcc": vm["mcc"],
                "val_sensitivity": vm["sensitivity"],
                "val_specificity": vm["specificity"],
                "val_gmean": vm["gmean"],
                "val_auprc": vm["auprc"],
                "val_auroc": vm["auroc"],
                "val_f1": vm["f1"],
                "val_tn": vm["tn"],
                "val_fp": vm["fp"],
                "val_fn": vm["fn"],
                "val_tp": vm["tp"],
            })
            for idx in train_idx:
                split_rows.append({"fold": fold_id, "index": int(idx), "split": "train", "label": int(self.y_train[idx]), "sequence": self.train_sequences[idx]})
            for idx in val_idx:
                split_rows.append({"fold": fold_id, "index": int(idx), "split": "validation", "label": int(self.y_train[idx]), "sequence": self.train_sequences[idx]})
            epoch_df.to_csv(os.path.join(fold_dir, "epoch_log.csv"), index=False)
        if np.isnan(oof_probs).any():
            raise RuntimeError("OOF predictions are incomplete")
        summary_df = pd.DataFrame(fold_rows)
        summary_df.to_csv(os.path.join(self.cv_dir, "fivefold_summary.csv"), index=False)
        pd.DataFrame(split_rows).to_csv(os.path.join(self.cv_dir, "fivefold_split_indices.csv"), index=False)
        oof_threshold, oof_mcc = find_best_threshold(oof_labels, oof_probs)
        oof_metrics = compute_metrics(oof_labels, oof_probs, oof_threshold)
        selected_epoch = int(np.median(np.asarray(best_epochs)).round())
        selected_epoch = max(1, min(EPOCHS, selected_epoch))
        pd.DataFrame({
            "index": np.arange(len(self.y_train), dtype=int),
            "fold": oof_fold.astype(int),
            "label": oof_labels.astype(int),
            "prob_avp": oof_probs.astype(float),
            "pred": (oof_probs >= oof_threshold).astype(int),
            "sequence": self.train_sequences,
        }).to_csv(os.path.join(self.cv_dir, "fivefold_oof_predictions.csv"), index=False)
        summary = {
            "dataset": self.dataset_name,
            "n_folds": N_FOLDS,
            "oof_threshold": float(oof_threshold),
            "oof_mcc": float(oof_mcc),
            "selected_final_epoch": int(selected_epoch),
            "oof_metrics": oof_metrics,
            "fold_mean_val_mcc": float(summary_df["val_mcc"].mean()),
            "fold_std_val_mcc": float(summary_df["val_mcc"].std(ddof=1)),
        }
        with open(os.path.join(self.cv_dir, "fivefold_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(self.cv_dir, "fivefold_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            for k, v in summary.items():
                if k != "oof_metrics":
                    f.write(f"{k}: {v}\n")
            for k, v in oof_metrics.items():
                f.write(f"oof_{k}: {v}\n")
        print(summary_df)
        print(f"OOF threshold: {oof_threshold:.3f}")
        print(f"Selected final epoch: {selected_epoch}")
        return summary

    def run_benchmark_final(self, cv_summary):
        result, _, _, _ = self.train_once(
            run_name="benchmark_full_train_final",
            run_dir=self.final_dir,
            train_sequences=self.train_sequences,
            train_features=self.train_features,
            train_labels=self.y_train,
            fixed_epochs=int(cv_summary["selected_final_epoch"]),
            external_threshold=float(cv_summary["oof_threshold"]),
        )
        metrics = result["benchmark_metrics"]
        row = {
            "dataset": self.dataset_name,
            "final_epoch": int(result["best_epoch"]),
            "threshold": float(result["best_threshold"]),
            "threshold_source": "fivefold_oof_predictions",
            "official_test_used_for_model_selection": False,
            "official_test_used_for_threshold_selection": False,
            "accuracy": metrics["accuracy"],
            "mcc": metrics["mcc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "gmean": metrics["gmean"],
            "auprc": metrics["auprc"],
            "auroc": metrics["auroc"],
            "f1": metrics["f1"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        }
        pd.DataFrame([row]).to_csv(os.path.join(self.final_dir, "benchmark_final_summary.csv"), index=False)
        with open(os.path.join(self.final_dir, "benchmark_final_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"benchmark": row, "cv_summary": cv_summary}, f, indent=2)
        with open(os.path.join(self.final_dir, "benchmark_final_summary.txt"), "w", encoding="utf-8") as f:
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
        print(pd.DataFrame([row]))
        return row

    def run(self):
        self.load_resources()
        cv_summary = self.run_fivefold_validation()
        seed_everything(RANDOM_SEED)
        set_seed()
        reset_memory()
        return self.run_benchmark_final(cv_summary)


def parse_args():
    parser = argparse.ArgumentParser(description="AVP_Fusion first-stage training and independent benchmark evaluation")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--esm-model-path", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset", default="set1", choices=["set1", "set2"])
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(RANDOM_SEED)
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runner = Stage1Runner(args.dataset, args.base_dir, args.esm_model_path, device)
    runner.run()


if __name__ == "__main__":
    main()
