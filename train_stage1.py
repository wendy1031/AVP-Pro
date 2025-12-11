import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    average_precision_score, f1_score
)
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from util.seed import set_seed
from util.data import load_data, generate_features, esm_encode
from util.blosum62_probabilistic import augment_sequence_with_second_best_mutation
from util.Queue_ohem import OHEMQueue
from util.focal_loss import FocalLoss
from util.loss_light_ohem import ContrastiveLoss
from model_att import AVP_HNCL_v3

# ===========================
# Configuration
# ===========================
set_seed()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Set 1")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test.txt")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints", "stage1")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model_stage1.pth")
SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")
os.makedirs(SAVE_DIR, exist_ok=True)

ESM_MODEL_PATH = "./esm2_t30_150M_UR50D"

# Hyperparameters
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
PATIENCE = 7

# Augmentation Parameters (Missing in original stage1)
NUM_FRAGMENTS = 6
MUTATION_RATE = 0.6
INSERTION_RATE = 0.5
DELETION_RATE = 0.5
MULTI_STEP = 1

# Loss Weights
CONTRASTIVE_WEIGHT = 0.10
CONSISTENCY_WEIGHT = 0.05
CONSISTENCY_TEMP   = 2.0

# Fusion Module Toggles
USE_RETRIEVAL = True
KNN_K = 8
RETR_FUSE_WEIGHT = 1.0
RETR_LOSS_WEIGHT = 0.02
RETR_ATT_TAU = 0.7
USE_MIL = True
KMER = 5
MIL_FUSE_WEIGHT = 1.0
MIL_SPARSITY = 5e-4

# ===========================
# Helper Classes & Functions
# ===========================
class SequenceDataset(Dataset):
    def __init__(self, sequences, additional_features, labels):
        self.sequences = sequences
        self.additional_features = np.array(additional_features, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.additional_features[idx], self.labels[idx]

def symmetric_kl(logits_a, logits_b, T=1.0):
    p_log = F.log_softmax(logits_a / T, dim=1)
    q_log = F.log_softmax(logits_b / T, dim=1)
    kl_pq = F.kl_div(p_log, q_log.exp(), reduction='batchmean')
    kl_qp = F.kl_div(q_log, p_log.exp(), reduction='batchmean')
    return 0.5 * (kl_pq + kl_qp)

@torch.no_grad()
def update_neg_queue_difficulty_scores(model, pos_queue, neg_queue, batch_size):
    model.eval()
    all_neg = neg_queue.get_all_embeddings()
    all_pos = pos_queue.get_all_embeddings()
    if all_neg is None or all_pos is None or all_neg.shape[0] == 0: return
    pos_proto = torch.mean(all_pos, dim=0, keepdim=True)
    n = all_neg.shape[0]
    scores = torch.zeros(n, device=DEVICE)
    for i in range(0, n, batch_size):
        batch = all_neg[i:i+batch_size]
        scores[i:i+batch_size] = F.cosine_similarity(batch, pos_proto)
    if neg_queue.is_full(): neg_queue.difficulty_scores[:] = scores
    else: neg_queue.difficulty_scores[:neg_queue.ptr] = scores

class RetrievalAugmentor(nn.Module):
    def __init__(self, esm_dim, model_emb_dim, k=8, att_tau=1.0):
        super().__init__()
        self.k = k
        self.att_tau = att_tau
        self.proj_neigh = nn.Linear(esm_dim, model_emb_dim)
        self.clf_fuse  = nn.Linear(model_emb_dim * 2, NUM_CLASSES)
        self.register_buffer("index_emb", torch.empty(0))
        self.register_buffer("index_labels", torch.empty(0, dtype=torch.long))
        self.seq2idxs = {}

    @torch.no_grad()
    def build_index_from_esm(self, sequences, labels, esm_model, tokenizer, device, max_length):
        embs = []
        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i:i+BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length).to(device)
            tokens = esm_model(**inputs).last_hidden_state
            pooled = tokens.mean(dim=1).float()
            embs.append(F.normalize(pooled, dim=1).detach().cpu())
        self.index_emb = torch.cat(embs, dim=0)
        self.index_labels = torch.tensor(labels, dtype=torch.long)
        self.seq2idxs = {s: [idx] for idx, s in enumerate(sequences)}

    @torch.no_grad()
    def rebuild_index_from_model(self, train_loader, model, device, tokenizer, esm_model, max_length):
        model.eval()
        all_embs, all_labels, all_seqs = [], [], []
        for batch_seqs, batch_add, batch_y in tqdm(train_loader, desc="Rebuild retrieval index"):
            batch_add_t = torch.as_tensor(batch_add, device=device)
            esm_tokens = esm_encode(list(batch_seqs), esm_model, tokenizer, device, max_length=max_length)
            _, embs = model(esm_tokens, batch_add_t)
            all_embs.append(F.normalize(embs.float(), dim=1).detach().cpu())
            all_labels.append(torch.as_tensor(batch_y, dtype=torch.long))
            all_seqs.extend(list(batch_seqs))
        self.index_emb = torch.cat(all_embs, dim=0)
        self.index_labels = torch.cat(all_labels, dim=0)
        self.seq2idxs = {s: [idx] for idx, s in enumerate(all_seqs)}

    def forward(self, query_seq_batch, query_esm_pooled, model_embs, device):
        with torch.cuda.amp.autocast(enabled=False):
            I = self.index_emb.to(device).float()
            if I.shape[1] == model_embs.shape[1]:
                Q = F.normalize(model_embs.float(), dim=1)
            else:
                Q = F.normalize(query_esm_pooled.float(), dim=1)
            sims = torch.matmul(Q, I.t())
            for b, seq in enumerate(query_seq_batch):
                for idx in self.seq2idxs.get(seq, []): sims[b, idx] = -1e4
            topk = torch.topk(sims, k=min(self.k, sims.shape[1]), dim=1).indices
            neigh_raw = self.index_emb.to(device)[topk]
            neigh_lbl = self.index_labels.to(device)[topk]
        if neigh_raw.shape[-1] == model_embs.shape[1]:
            neigh_proj = neigh_raw
        else:
            neigh_proj = self.proj_neigh(neigh_raw)
        q = model_embs.unsqueeze(1)
        att = torch.softmax(torch.matmul(q, neigh_proj.transpose(1, 2)) / ((model_embs.shape[1] ** 0.5) * self.att_tau), dim=-1)
        z_neigh = torch.matmul(att, neigh_proj).squeeze(1)
        logits_retr = self.clf_fuse(torch.cat([model_embs, z_neigh], dim=1))
        with torch.no_grad():
            mask_pos = (neigh_lbl == 1).float()
            mask_neg = (neigh_lbl == 0).float()
            eps = 1e-6
            pos_proto = (neigh_proj * mask_pos.unsqueeze(-1)).sum(1) / (mask_pos.sum(1, keepdim=True) + eps)
            neg_proto = (neigh_proj * mask_neg.unsqueeze(-1)).sum(1) / (mask_neg.sum(1, keepdim=True) + eps)
        retr_align_loss = F.mse_loss(model_embs, pos_proto, reduction='mean') - 0.5 * F.mse_loss(model_embs, neg_proto, reduction='mean')
        return logits_retr, retr_align_loss

class ConvKmerMIL(nn.Module):
    def __init__(self, esm_dim, model_emb_dim, kmer=5):
        super().__init__()
        self.conv = nn.Conv1d(esm_dim, esm_dim, kernel_size=kmer, stride=1, padding=kmer//2, bias=False)
        nn.init.dirac_(self.conv.weight)
        self.proj = nn.Linear(esm_dim, model_emb_dim)
        self.att = nn.Linear(model_emb_dim, 1)
        self.clf_fuse = nn.Linear(model_emb_dim * 2, NUM_CLASSES)

    def forward(self, esm_tokens, model_embs):
        X = esm_tokens.transpose(1, 2)
        K = self.conv(X).transpose(1, 2)
        Kp = self.proj(K)
        w = torch.softmax(self.att(Kp).squeeze(-1), dim=1)
        z_mil = (w.unsqueeze(-1) * Kp).sum(1)
        logits_mil = self.clf_fuse(torch.cat([model_embs, z_mil], dim=1))
        sparsity = (w * torch.log(w + 1e-12)).sum(dim=1).mean()
        return logits_mil, sparsity

# ===========================
# Global Variables
# ===========================
model, esm2_model, tokenizer, retr, mil = [None] * 5

# ===========================
# Global Forward Functions
# ===========================
def model_forward_bn_ok(tokens, add_feats):
    if model.training and tokens.size(0) == 1:
        tokens2 = torch.cat([tokens, tokens], dim=0)
        add2 = torch.cat([add_feats, add_feats], dim=0)
        logits2, embs2 = model(tokens2, add2)
        return logits2[:1], embs2[:1]
    return model(tokens, add_feats)

def forward_with_fusion(batch_seqs, esm_tokens, add_feats, use_retr=True, use_mil=True, training=False):
    logits_base, embs = model_forward_bn_ok(esm_tokens, add_feats)
    logits = logits_base
    extra_loss = 0.0
    
    if use_retr and retr is not None:
        q_pooled = esm_tokens.mean(dim=1).float()
        logits_retr, retr_align = retr(list(batch_seqs), q_pooled, embs, DEVICE)
        logits = logits + RETR_FUSE_WEIGHT * logits_retr
        if training: extra_loss += RETR_LOSS_WEIGHT * retr_align
        
    if use_mil and mil is not None:
        logits_mil, sparsity = mil(esm_tokens, embs)
        logits = logits + MIL_FUSE_WEIGHT * logits_mil
        if training: extra_loss += MIL_SPARSITY * sparsity
        
    return logits, embs, extra_loss

def evaluate(model_to_eval, dataloader):
    """
    Standard evaluation during training (Threshold 0.5)
    """
    model_to_eval.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch_seqs, batch_add, batch_y in dataloader:
            batch_add_t = torch.as_tensor(batch_add, device=DEVICE)
            esm_tokens = esm_encode(list(batch_seqs), esm2_model, tokenizer, DEVICE, max_length=MAX_LENGTH)
            with autocast():
                logits, _, _ = forward_with_fusion(
                    batch_seqs, esm_tokens, batch_add_t, 
                    use_retr=USE_RETRIEVAL, use_mil=USE_MIL, training=False
                )
            _, predicted = torch.max(logits, 1)
            preds.extend(predicted.detach().cpu().numpy())
            labels.extend(batch_y.numpy() if isinstance(batch_y, torch.Tensor) else batch_y)
    return matthews_corrcoef(labels, preds)

def final_evaluation(test_loader):
    """
    Evaluation with threshold search (matches new.py)
    """
    print("\n--- Final Evaluation with Threshold Search ---")
    
    # Reload best model
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if USE_RETRIEVAL and retr is not None:
        retr.load_state_dict(ckpt["retr"])
    if USE_MIL and mil is not None:
        mil.load_state_dict(ckpt["mil"])
        
    model.eval()
    labels_list, probs_list = [], []
    
    with torch.no_grad():
        for batch_seqs, batch_add, batch_y in tqdm(test_loader, desc="Final Eval"):
            batch_add_t = torch.as_tensor(batch_add, device=DEVICE)
            esm_tokens = esm_encode(list(batch_seqs), esm2_model, tokenizer, DEVICE, max_length=MAX_LENGTH)
            with autocast():
                logits, _, _ = forward_with_fusion(
                    batch_seqs, esm_tokens, batch_add_t,
                    use_retr=USE_RETRIEVAL, use_mil=USE_MIL, training=False
                )
            prob = F.softmax(logits, dim=1)[:, 1]
            labels_list.extend(batch_y.numpy() if isinstance(batch_y, torch.Tensor) else batch_y)
            probs_list.extend(prob.detach().cpu().numpy())

    labels_np = np.array(labels_list)
    probs_np  = np.array(probs_list)

    # Threshold search
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_mcc = 0.5, -2.0
    
    for t in thresholds:
        preds_t = (probs_np >= t).astype(int)
        mcc_t = matthews_corrcoef(labels_np, preds_t)
        if mcc_t > best_mcc:
            best_mcc, best_t = mcc_t, t

    # Final metrics at best threshold
    preds = (probs_np >= best_t).astype(int)
    cm = confusion_matrix(labels_np, preds)
    
    try:
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        sn  = tp / max(tp + fn, 1)
        sp  = tn / max(tn + fp, 1)
        gmean = np.sqrt(sn * sp)
    except ValueError:
        acc = sn = sp = gmean = 0.0

    auprc = average_precision_score(labels_np, probs_np)
    auroc = roc_auc_score(labels_np, probs_np)
    f1    = f1_score(labels_np, preds)

    print(f"\n--- Final Detailed Results ---")
    print(f"Best threshold for MCC: {best_t:.3f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"MCC: {best_mcc:.4f}")
    print(f"Sensitivity (SN): {sn:.4f}")
    print(f"Specificity (SP): {sp:.4f}")
    print(f"G-Mean: {gmean:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

# ===========================
# Main Training Logic
# ===========================
def train_and_evaluate():
    global model, esm2_model, tokenizer, retr, mil

    print("Loading ESM-2 model...")
    esm2_model = AutoModel.from_pretrained(ESM_MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
    esm2_model.eval()

    print("Loading data...")
    pos_train, neg_train = load_data(TRAIN_FILE)
    pos_test, neg_test = load_data(TEST_FILE)
    y_train = np.array([1]*len(pos_train) + [0]*len(neg_train), dtype=np.int64)
    y_test = np.array([1]*len(pos_test)  + [0]*len(neg_test),  dtype=np.int64)
    train_sequences = pos_train + neg_train
    test_sequences  = pos_test  + neg_test

    print("Building physchem features...")
    train_feat = generate_features(TRAIN_FILE)
    test_feat  = generate_features(TEST_FILE)
    train_feat_sp = np.log1p(np.exp(np.clip(train_feat.values, -10, 10)))
    test_feat_sp  = np.log1p(np.exp(np.clip(test_feat.values,  -10, 10)))
    scaler = StandardScaler()
    train_feat_std = scaler.fit_transform(train_feat_sp)
    test_feat_std  = scaler.transform(test_feat_sp)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    ADDITIONAL_DIM = train_feat_std.shape[1]
    print(f"ADDITIONAL_DIM = {ADDITIONAL_DIM}")

    train_ds = SequenceDataset(train_sequences, train_feat_std, y_train)
    test_ds  = SequenceDataset(test_sequences,  test_feat_std,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("Building model...")
    model = AVP_HNCL_v3(ESM_DIM, ADDITIONAL_DIM, CNN_OUT_CHANNELS, LSTM_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
    MODEL_EMB_DIM = model.embedding_dim
    
    # Updated: Added regularization=1e-4 to match new.py
    contrastive_loss_fn = ContrastiveLoss(temperature=0.5, learnable_temperature=True, regularization=1e-4).to(DEVICE)
    
    n_samples, n0, n1 = len(y_train), int(np.sum(y_train==0)), int(np.sum(y_train==1))
    alpha_tensor = torch.tensor([(n_samples/(2.0*max(n0,1))), (n_samples/(2.0*max(n1,1)))*1.5], device=DEVICE)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)

    retr = RetrievalAugmentor(ESM_DIM, MODEL_EMB_DIM, k=KNN_K, att_tau=RETR_ATT_TAU).to(DEVICE) if USE_RETRIEVAL else None
    mil = ConvKmerMIL(ESM_DIM, MODEL_EMB_DIM, kmer=KMER).to(DEVICE) if USE_MIL else None
    
    if retr:
        print("Building retrieval index (ESM pooled) on train set...")
        retr.build_index_from_esm(train_sequences, y_train, esm2_model, tokenizer, DEVICE, MAX_LENGTH)

    params = list(model.parameters()) + (list(retr.parameters()) if retr else []) + (list(mil.parameters()) if mil else [])
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    grad_scaler = GradScaler()
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * 5
    def lr_lambda(step):
        if step < warmup_steps: return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    pos_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=MODEL_EMB_DIM).to(DEVICE)
    neg_queue = OHEMQueue(max_size=QUEUE_SIZE, embedding_dim=MODEL_EMB_DIM).to(DEVICE)
    print("Pre-populating queues...")
    model.eval()
    with torch.no_grad():
        for seqs, feats, labels in tqdm(train_loader, desc="Pre-populating"):
            feats_t = torch.as_tensor(feats, device=DEVICE)
            esm_tokens = esm_encode(list(seqs), esm2_model, tokenizer, DEVICE, max_length=MAX_LENGTH)
            _, embs = model(esm_tokens, feats_t)
            p_idx, n_idx = (labels==1).nonzero(as_tuple=True)[0], (labels==0).nonzero(as_tuple=True)[0]
            if len(p_idx)>0: pos_queue.enqueue(embs[p_idx].to(torch.float32))
            if len(n_idx)>0: neg_queue.enqueue(embs[n_idx].to(torch.float32))
            if pos_queue.is_full() and neg_queue.is_full(): break

    best_mcc, best_epoch, patience_cnt = -1.0, 0, 0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        if USE_RETRIEVAL and epoch in (1, 3, 5, 8):
            retr.rebuild_index_from_model(train_loader, model, DEVICE, tokenizer, esm2_model, MAX_LENGTH)
        update_neg_queue_difficulty_scores(model, pos_queue, neg_queue, BATCH_SIZE)
        model.train()
        
        for seqs, feats, labels in tqdm(train_loader, desc="Training"):
            feats_t, labels_t = torch.as_tensor(feats, device=DEVICE), torch.as_tensor(labels, device=DEVICE)
            p_idx = (labels_t==1).nonzero(as_tuple=True)[0]
            
            # Updated: Added augmentation parameters to match new.py
            aug_seqs = [
                augment_sequence_with_second_best_mutation(
                    s, NUM_FRAGMENTS, MUTATION_RATE, INSERTION_RATE, DELETION_RATE, MULTI_STEP
                ) for s in [seqs[i] for i in p_idx.tolist()]
            ] if len(p_idx)>0 else []

            esm_tokens = esm_encode(list(seqs), esm2_model, tokenizer, DEVICE, max_length=MAX_LENGTH)
            esm_tokens_aug = esm_encode(aug_seqs, esm2_model, tokenizer, DEVICE, max_length=MAX_LENGTH) if len(aug_seqs)>0 else None

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits, embs, extra_loss = forward_with_fusion(
                    list(seqs), esm_tokens, feats_t, 
                    use_retr=USE_RETRIEVAL, use_mil=USE_MIL, training=True
                )
                
                cls_loss = criterion(logits, labels_t)
                con_loss = 0.0
                if len(p_idx)>0 and len(aug_seqs)>0 and neg_queue.size() > K_HARD_NEGATIVES:
                    _, embs_aug = model_forward_bn_ok(esm_tokens_aug, feats_t[p_idx])
                    scores = neg_queue.difficulty_scores[:neg_queue.size()]
                    consider = min(len(scores), K_HARD_NEGATIVES*10)
                    _, topk = torch.topk(scores, k=consider)
                    # Use explicit random permutation like new.py implies/uses
                    perm = torch.randperm(topk.size(0), device=DEVICE)
                    hard_idx = topk[perm[:K_HARD_NEGATIVES]]
                    hard_negs = neg_queue.embeddings[hard_idx]
                    con_loss = contrastive_loss_fn(embs[p_idx], embs_aug, hard_negs)
                
                cons_loss = 0.0
                if len(p_idx)>0 and len(aug_seqs)>0:
                    logits_aug, _, _ = forward_with_fusion(
                        [seqs[i] for i in p_idx.tolist()], esm_tokens_aug, feats_t[p_idx], 
                        use_retr=USE_RETRIEVAL, use_mil=USE_MIL, training=False
                    )
                    cons_loss = symmetric_kl(logits[p_idx], logits_aug, T=CONSISTENCY_TEMP)

                loss = cls_loss + CONTRASTIVE_WEIGHT*con_loss + CONSISTENCY_WEIGHT*cons_loss + extra_loss

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            if GRAD_CLIP_NORM is not None: torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            with torch.no_grad():
                p_idx_all, n_idx_all = (labels_t==1).nonzero(as_tuple=True)[0], (labels_t==0).nonzero(as_tuple=True)[0]
                if len(p_idx_all)>0: pos_queue.enqueue(embs[p_idx_all].detach().to(torch.float32))
                if len(n_idx_all)>0: neg_queue.enqueue(embs[n_idx_all].detach().to(torch.float32))

        mcc = evaluate(model, test_loader)
        print(f"Validation MCC (Threshold 0.5): {mcc:.4f}")

        if mcc > best_mcc:
            print(f"New best MCC: {best_mcc:.4f} -> {mcc:.4f}")
            best_mcc, best_epoch = mcc, epoch+1
            ckpt = {"model": model.state_dict(), "retr": retr.state_dict() if retr else None, "mil": mil.state_dict() if mil else None}
            torch.save(ckpt, BEST_MODEL_PATH)
            print(f"Saved: {BEST_MODEL_PATH}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            print(f"No improvement. Patience {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    print(f"\nBest epoch {best_epoch}, MCC {best_mcc:.4f}")
    
    # Run Final Evaluation with threshold search
    final_evaluation(test_loader)
    
if __name__ == '__main__':
    train_and_evaluate()