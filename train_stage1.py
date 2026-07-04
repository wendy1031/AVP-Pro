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

for p in [str(SCRIPT_DIR), DEFAULT_BASE_DIR, str(SCRIPT_DIR.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score, f1_score
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from util.data import load_data, generate_features, esm_encode
from util.blosum62_probabilistic import augment_sequence_with_second_best_mutation
from util.Queue_ohem import OHEMQueue
from loss_light_ohem import ContrastiveLoss
from util.focal_loss import FocalLoss

import model_att

AVP_Fusion_v3 = getattr(model_att, "AVP_Fusion_v3", None)
if AVP_Fusion_v3 is None:
    AVP_Fusion_v3 = getattr(model_att, "AVP_" + "HN" + "CL_v3")

RANDOM_SEED = 1064
ESM_DIM = 640
CNN_OUT_CHANNELS = 256
LSTM_HIDDEN_DIM = 256
NUM_CLASSES = 2
DROPOUT_RATE = 0.45
MAX_LENGTH = 100
EPOCHS = 7
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
FINAL_THRESHOLD = 0.30


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
    def __init__(self, sequences, additional_features, labels):
        self.sequences = list(sequences)
        self.additional_features = np.asarray(additional_features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.additional_features[idx], self.labels[idx]


def symmetric_kl(logits_a, logits_b, temperature=1.0):
    p_log = F.log_softmax(logits_a / temperature, dim=1)
    q_log = F.log_softmax(logits_b / temperature, dim=1)
    p = p_log.exp()
    q = q_log.exp()
    kl_pq = F.kl_div(p_log, q, reduction="batchmean")
    kl_qp = F.kl_div(q_log, p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)


def compute_metrics(labels_np, probs_np, threshold):
    labels_np = np.asarray(labels_np).astype(int)
    probs_np = np.asarray(probs_np).astype(float)
    preds = (probs_np >= threshold).astype(int)
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sn = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    gmean = float(np.sqrt(sn * sp))
    mcc = matthews_corrcoef(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    try:
        auprc = average_precision_score(labels_np, probs_np)
    except Exception:
        auprc = float("nan")
    try:
        auroc = roc_auc_score(labels_np, probs_np)
    except Exception:
        auroc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "mcc": float(mcc),
        "sensitivity": float(sn),
        "specificity": float(sp),
        "gmean": float(gmean),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def update_neg_queue_difficulty_scores(pos_queue, neg_queue, batch_size, device):
    all_neg_embeddings = neg_queue.get_all_embeddings()
    all_pos_embeddings = pos_queue.get_all_embeddings()
    if all_neg_embeddings is None or all_pos_embeddings is None:
        neg_queue.difficulty_scores.zero_()
        return
    if all_neg_embeddings.shape[0] == 0 or all_pos_embeddings.shape[0] == 0:
        neg_queue.difficulty_scores.zero_()
        return
    pos_proto = torch.mean(all_pos_embeddings, dim=0, keepdim=True)
    n = all_neg_embeddings.shape[0]
    scores = torch.zeros(n, device=device)
    for i in range(0, n, batch_size):
        batch = all_neg_embeddings[i:i + batch_size]
        scores[i:i + batch_size] = F.cosine_similarity(batch, pos_proto)
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
    def build_index_from_esm(self, sequences, labels, esm_model, tokenizer, device, max_length):
        batch_size = 64
        embs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
            tokens = esm_model(**inputs).last_hidden_state
            pooled = tokens.mean(dim=1).float()
            embs.append(F.normalize(pooled, dim=1).detach().cpu())
        self.index_emb = torch.cat(embs, dim=0)
        self.index_labels = torch.tensor(labels, dtype=torch.long)
        self.seq2idxs = {}
        for idx, seq in enumerate(sequences):
            self.seq2idxs.setdefault(seq, []).append(idx)

    @torch.no_grad()
    def rebuild_index_from_model(self, train_loader, model, device, tokenizer, esm_model, max_length):
        model.eval()
        all_embs = []
        all_labels = []
        all_seqs = []
        for batch_seqs, batch_add, batch_y in tqdm(train_loader, desc="Rebuild retrieval index"):
            batch_add_t = torch.as_tensor(batch_add, device=device)
            esm_tokens = esm_encode(list(batch_seqs), esm_model, tokenizer, device, max_length=max_length)
            _, embs = model(esm_tokens, batch_add_t)
            all_embs.append(F.normalize(embs.float(), dim=1).detach().cpu())
            all_labels.append(torch.as_tensor(batch_y, dtype=torch.long))
            all_seqs.extend(list(batch_seqs))
        self.index_emb = torch.cat(all_embs, dim=0)
        self.index_labels = torch.cat(all_labels, dim=0)
        self.seq2idxs = {}
        for idx, seq in enumerate(all_seqs):
            self.seq2idxs.setdefault(seq, []).append(idx)

    def forward(self, query_seq_batch, query_esm_pooled, model_embs, device):
        with torch.cuda.amp.autocast(enabled=False):
            index = self.index_emb.to(device).float()
            if index.shape[1] == model_embs.shape[1]:
                query = F.normalize(model_embs.float(), dim=1)
            else:
                query = F.normalize(query_esm_pooled.float(), dim=1)
            sims = torch.matmul(query, index.t())
            for b, seq in enumerate(query_seq_batch):
                for idx in self.seq2idxs.get(seq, []):
                    sims[b, idx] = -1e4
            topk = torch.topk(sims, k=min(self.k, sims.shape[1]), dim=1).indices
            neigh_raw = self.index_emb.to(device)[topk]
            neigh_lbl = self.index_labels.to(device)[topk]
        if neigh_raw.shape[-1] == model_embs.shape[1]:
            neigh_proj = neigh_raw
        else:
            neigh_proj = self.proj_neigh(neigh_raw)
        query_model = model_embs.unsqueeze(1)
        att = torch.softmax(torch.matmul(query_model, neigh_proj.transpose(1, 2)) / ((model_embs.shape[1] ** 0.5) * self.att_tau), dim=-1)
        z_neigh = torch.matmul(att, neigh_proj).squeeze(1)
        logits_retr = self.clf_fuse(torch.cat([model_embs, z_neigh], dim=1))
        with torch.no_grad():
            mask_pos = (neigh_lbl == 1).float()
            mask_neg = (neigh_lbl == 0).float()
            eps = 1e-6
            pos_proto = (neigh_proj * mask_pos.unsqueeze(-1)).sum(1) / (mask_pos.sum(1, keepdim=True) + eps)
            neg_proto = (neigh_proj * mask_neg.unsqueeze(-1)).sum(1) / (mask_neg.sum(1, keepdim=True) + eps)
        proto_pull = F.mse_loss(model_embs, pos_proto, reduction="mean")
        proto_push = -F.mse_loss(model_embs, neg_proto, reduction="mean")
        retr_align_loss = proto_pull + 0.5 * proto_push
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
        x = esm_tokens.transpose(1, 2)
        kmer_tokens = self.conv(x).transpose(1, 2)
        kmer_proj = self.proj(kmer_tokens)
        weights = torch.softmax(self.att(kmer_proj).squeeze(-1), dim=1)
        z_mil = (weights.unsqueeze(-1) * kmer_proj).sum(1)
        logits_mil = self.clf_fuse(torch.cat([model_embs, z_mil], dim=1))
        sparsity = (weights * torch.log(weights + 1e-12)).sum(dim=1).mean()
        return logits_mil, sparsity


class AVPFusionStage1Runner:
    def __init__(self, base_dir, esm_model_path, output_dir, device):
        self.base_dir = base_dir
        self.esm_model_path = esm_model_path
        self.output_dir = output_dir
        self.device = device
        self.set_dir = os.path.join(base_dir, "dataset/Set 1")
        self.train_file = os.path.join(self.set_dir, "train.txt")
        self.test_file = os.path.join(self.set_dir, "test.txt")
        os.makedirs(output_dir, exist_ok=True)

    def load_resources(self):
        self.esm2_model = AutoModel.from_pretrained(self.esm_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.esm_model_path)
        self.esm2_model.eval()

        pos_train, neg_train = load_data(self.train_file)
        pos_test, neg_test = load_data(self.test_file)

        self.train_sequences = pos_train + neg_train
        self.test_sequences = pos_test + neg_test
        self.y_train = np.array([1] * len(pos_train) + [0] * len(neg_train), dtype=np.int64)
        self.y_test = np.array([1] * len(pos_test) + [0] * len(neg_test), dtype=np.int64)

        train_features = np.asarray(generate_features(self.train_file).values, dtype=np.float32)
        test_features = np.asarray(generate_features(self.test_file).values, dtype=np.float32)

        self.train_features, self.test_features = self.preprocess_features(train_features, test_features)

    def preprocess_features(self, train_features, test_features):
        train_clip = np.clip(train_features, -10, 10)
        test_clip = np.clip(test_features, -10, 10)
        train_sp = np.log1p(np.exp(train_clip))
        test_sp = np.log1p(np.exp(test_clip))
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train_sp)
        test_std = scaler.transform(test_sp)
        return train_std, test_std

    def build_loader(self, sequences, features, labels, shuffle):
        dataset = SequenceDataset(sequences, features, labels)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)

    def build_model_bundle(self):
        model = AVP_Fusion_v3(
            esm_dim=ESM_DIM,
            additional_dim=self.train_features.shape[1],
            cnn_out_channels=CNN_OUT_CHANNELS,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
        ).to(self.device)

        model_emb_dim = getattr(model, "embedding_dim", 512)

        contrastive_loss_fn = ContrastiveLoss(temperature=0.5, learnable_temperature=True, regularization=1e-4).to(self.device)

        n_samples = len(self.y_train)
        n0 = int(np.sum(self.y_train == 0))
        n1 = int(np.sum(self.y_train == 1))
        alpha_tensor = torch.tensor(
            [n_samples / (2.0 * max(n0, 1)), (n_samples / (2.0 * max(n1, 1))) * 1.5],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)

        retr = RetrievalAugmentor(esm_dim=ESM_DIM, model_emb_dim=model_emb_dim, k=KNN_K, att_tau=RETR_ATT_TAU).to(self.device)
        retr.build_index_from_esm(self.train_sequences, self.y_train, self.esm2_model, self.tokenizer, self.device, MAX_LENGTH)

        mil = ConvKmerMIL(esm_dim=ESM_DIM, model_emb_dim=model_emb_dim, kmer=KMER).to(self.device)

        params = list(model.parameters()) + list(retr.parameters()) + list(mil.parameters())
        optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler()

        return model, model_emb_dim, contrastive_loss_fn, criterion, retr, mil, params, optimizer, scaler

    def model_forward_bn_ok(self, model, tokens, add_feats):
        if model.training and tokens.size(0) == 1:
            tokens2 = torch.cat([tokens, tokens], dim=0)
            add2 = torch.cat([add_feats, add_feats], dim=0)
            logits2, embs2 = model(tokens2, add2)
            return logits2[:1], embs2[:1]
        return model(tokens, add_feats)

    def forward_with_fusion(self, model, retr, mil, batch_seqs, esm_tokens, add_feats, training):
        logits_base, embs = self.model_forward_bn_ok(model, esm_tokens, add_feats)
        logits = logits_base
        extra_loss = 0.0

        if USE_RETRIEVAL:
            q_pooled = esm_tokens.mean(dim=1).float()
            logits_retr, retr_align = retr(list(batch_seqs), q_pooled, embs, self.device)
            logits = logits + RETR_FUSE_WEIGHT * logits_retr
            if training:
                extra_loss = extra_loss + RETR_LOSS_WEIGHT * retr_align

        if USE_MIL:
            logits_mil, sparsity = mil(esm_tokens, embs)
            logits = logits + MIL_FUSE_WEIGHT * logits_mil
            if training:
                extra_loss = extra_loss + MIL_SPARSITY * sparsity

        return logits, embs, extra_loss

    @torch.no_grad()
    def evaluate(self, model, retr, mil, criterion, dataloader):
        model.eval()
        labels = []
        probs = []
        running_loss = 0.0
        total = 0

        for batch_seqs, batch_add, batch_y in tqdm(dataloader, desc="Evaluation"):
            batch_add_t = torch.as_tensor(batch_add, device=self.device)
            y_t = torch.as_tensor(batch_y, device=self.device)
            esm_tokens = esm_encode(list(batch_seqs), self.esm2_model, self.tokenizer, self.device, max_length=MAX_LENGTH)

            with autocast():
                logits, _, _ = self.forward_with_fusion(model, retr, mil, batch_seqs, esm_tokens, batch_add_t, training=False)
                loss = criterion(logits, y_t)

            prob = F.softmax(logits, dim=1)[:, 1]
            probs.extend(prob.detach().cpu().numpy())
            labels.extend(y_t.detach().cpu().numpy())
            running_loss += float(loss.item()) * y_t.size(0)
            total += y_t.size(0)

        labels_np = np.array(labels)
        probs_np = np.array(probs)
        metrics = compute_metrics(labels_np, probs_np, FINAL_THRESHOLD)
        metrics["loss"] = float(running_loss / max(total, 1))
        return metrics, labels_np, probs_np

    def train(self):
        train_loader = self.build_loader(self.train_sequences, self.train_features, self.y_train, shuffle=True)
        test_loader = self.build_loader(self.test_sequences, self.test_features, self.y_test, shuffle=False)

        model, model_emb_dim, contrastive_loss_fn, criterion, retr, mil, params, optimizer, scaler = self.build_model_bundle()

        total_steps = len(train_loader) * EPOCHS
        warmup_steps = len(train_loader) * 5

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        pos_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=model_emb_dim).to(self.device)
        neg_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=model_emb_dim).to(self.device)

        model.eval()
        with torch.no_grad():
            for batch_seqs, batch_add, batch_y in tqdm(train_loader, desc="Queue initialization"):
                batch_add_t = torch.as_tensor(batch_add, device=self.device)
                y_t = torch.as_tensor(batch_y, device=self.device)
                esm_tokens = esm_encode(list(batch_seqs), self.esm2_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                _, embs = model(esm_tokens, batch_add_t)
                pos_idx = (y_t == 1).nonzero(as_tuple=True)[0]
                neg_idx = (y_t == 0).nonzero(as_tuple=True)[0]
                if len(pos_idx) > 0:
                    pos_queue.enqueue(embs[pos_idx].to(torch.float32))
                if len(neg_idx) > 0:
                    neg_queue.enqueue(embs[neg_idx].to(torch.float32))
                if pos_queue.is_full() and neg_queue.is_full():
                    break

        epoch_logs = []

        for epoch in range(EPOCHS):
            if USE_RETRIEVAL and epoch in (1, 3, 5):
                retr.rebuild_index_from_model(train_loader, model, self.device, self.tokenizer, self.esm2_model, MAX_LENGTH)

            update_neg_queue_difficulty_scores(pos_queue, neg_queue, BATCH_SIZE, self.device)
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for batch_seqs, batch_add, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
                batch_add_t = torch.as_tensor(batch_add, device=self.device)
                y_t = torch.as_tensor(batch_y, device=self.device)

                pos_idx_batch = (y_t == 1).nonzero(as_tuple=True)[0]
                pos_seqs = [batch_seqs[i] for i in pos_idx_batch.tolist()] if len(pos_idx_batch) > 0 else []
                aug_sequences = [
                    augment_sequence_with_second_best_mutation(seq, NUM_FRAGMENTS, MUTATION_RATE, INSERTION_RATE, DELETION_RATE, MULTI_STEP)
                    for seq in pos_seqs
                ] if len(pos_idx_batch) > 0 else []

                esm_tokens = esm_encode(list(batch_seqs), self.esm2_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                if len(pos_idx_batch) > 0 and len(aug_sequences) > 0:
                    esm_tokens_aug_pos = esm_encode(aug_sequences, self.esm2_model, self.tokenizer, self.device, max_length=MAX_LENGTH)
                else:
                    esm_tokens_aug_pos = None

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    logits_total, embs, extra_loss = self.forward_with_fusion(model, retr, mil, batch_seqs, esm_tokens, batch_add_t, training=True)
                    cls_loss = criterion(logits_total, y_t)
                    con_loss = 0.0

                    if len(pos_idx_batch) > 0 and esm_tokens_aug_pos is not None and neg_queue.size() > K_HARD_NEGATIVES:
                        add_pos = batch_add_t[pos_idx_batch]
                        _, embs_aug_pos = self.model_forward_bn_ok(model, esm_tokens_aug_pos, add_pos)
                        anchor = embs[pos_idx_batch]
                        positive = embs_aug_pos
                        scores = neg_queue.difficulty_scores[:neg_queue.size()]
                        consider = min(len(scores), K_HARD_NEGATIVES * 10)
                        _, topk = torch.topk(scores, k=consider, largest=True)
                        perm = torch.randperm(topk.size(0), device=self.device)
                        hard_idx = topk[perm[:K_HARD_NEGATIVES]]
                        hard_negs = neg_queue.embeddings[hard_idx]
                        con_loss = contrastive_loss_fn(anchor, positive, hard_negs)

                    pos_idx_all = (y_t == 1).nonzero(as_tuple=True)[0]
                    neg_idx_all = (y_t == 0).nonzero(as_tuple=True)[0]
                    if len(pos_idx_all) > 0:
                        pos_queue.enqueue(embs[pos_idx_all].detach().to(torch.float32))
                    if len(neg_idx_all) > 0:
                        neg_queue.enqueue(embs[neg_idx_all].detach().to(torch.float32))

                    cons_loss = 0.0
                    if len(pos_idx_batch) > 0 and esm_tokens_aug_pos is not None:
                        logits_pos = logits_total[pos_idx_batch]
                        logits_pos_aug, _, _ = self.forward_with_fusion(
                            model,
                            retr,
                            mil,
                            [batch_seqs[i] for i in pos_idx_batch.tolist()],
                            esm_tokens_aug_pos,
                            batch_add_t[pos_idx_batch],
                            training=False,
                        )
                        cons_loss = symmetric_kl(logits_pos, logits_pos_aug, temperature=CONSISTENCY_TEMP)

                    loss = cls_loss + CONTRASTIVE_WEIGHT * con_loss + CONSISTENCY_WEIGHT * cons_loss + extra_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += float(loss.item()) * y_t.size(0)
                _, pred = torch.max(logits_total, 1)
                total += y_t.size(0)
                correct += int((pred == y_t).sum().item())

            epoch_logs.append({
                "epoch": epoch + 1,
                "train_loss": float(running_loss / max(total, 1)),
                "train_accuracy": float(correct / max(total, 1)),
            })
            pd.DataFrame(epoch_logs).to_csv(os.path.join(self.output_dir, "epoch_log.csv"), index=False)
            print(f"Epoch {epoch + 1} | loss={epoch_logs[-1]['train_loss']:.4f} | acc={epoch_logs[-1]['train_accuracy']:.4f}")

        metrics, labels_np, probs_np = self.evaluate(model, retr, mil, criterion, test_loader)

        torch.save({
            "model": model.state_dict(),
            "retr": retr.state_dict(),
            "mil": mil.state_dict(),
            "threshold": FINAL_THRESHOLD,
            "metrics": metrics,
        }, os.path.join(self.output_dir, "AVP_Fusion_stage1_model.pth"))

        pd.DataFrame({
            "sequence": self.test_sequences,
            "label": labels_np.astype(int),
            "probability": probs_np.astype(float),
            "prediction": (probs_np >= FINAL_THRESHOLD).astype(int),
        }).to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)

        with open(os.path.join(self.output_dir, "stage1_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(self.output_dir, "stage1_metrics.txt"), "w", encoding="utf-8") as f:
            for key in ["threshold", "accuracy", "mcc", "sensitivity", "specificity", "gmean", "f1", "auprc", "auroc", "tn", "fp", "fn", "tp"]:
                f.write(f"{key}: {metrics[key]}\n")
            f.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")

        print("Final results")
        print(f"Threshold: {metrics['threshold']:.3f}")
        print(f"ACC: {metrics['accuracy']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"SN: {metrics['sensitivity']:.4f}")
        print(f"SP: {metrics['specificity']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(np.array(metrics["confusion_matrix"]))


def parse_args():
    parser = argparse.ArgumentParser(description="AVP_Fusion Stage 1 training and evaluation")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--esm-model-path", type=str, default=os.path.join(DEFAULT_BASE_DIR, "esm2_t30_150M_UR50D"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(DEFAULT_BASE_DIR, "dataset/Set 1/AVP_Fusion_stage1_final"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    reset_memory()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runner = AVPFusionStage1Runner(
        base_dir=args.base_dir,
        esm_model_path=args.esm_model_path,
        output_dir=args.output_dir,
        device=device,
    )
    runner.load_resources()
    runner.train()


if __name__ == "__main__":
    main()
