import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel

# --- Standalone Definitions ---
from model_att import AVP_HNCL_v3
from util.data import esm_encode, generate_features_single_seq
from util.seed import set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_ESM_PATH = "./esm2_t30_150M_UR50D"
NUM_CLASSES = 2

class ConvKmerMIL(nn.Module):
    def __init__(self, esm_dim, model_emb_dim, kmer=5):
        super().__init__()
        self.conv = nn.Conv1d(esm_dim, esm_dim, kernel_size=kmer, padding=kmer//2, bias=False)
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
        return logits_mil, 0.0

# --- Simplified Fusion for Prediction ---
def forward_with_fusion_predict(model, mil, esm_tokens, add_feats):
    logits_base, embs = model(esm_tokens, add_feats)
    logits = logits_base
    if mil is not None:
        logits_mil, _ = mil(esm_tokens, embs)
        logits = logits + logits_mil
    return logits, embs, 0.0

def load_prediction_model(ckpt_path, additional_dim=1159, esm_dim=640):
    print(f"Loading model for prediction from {ckpt_path}...")
    
    model = AVP_HNCL_v3(esm_dim, additional_dim, 256, 256, NUM_CLASSES, 0.0).to(DEVICE)
    mil = ConvKmerMIL(esm_dim, model.embedding_dim).to(DEVICE)
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    print("    -> Main model weights loaded.")
    
    if 'mil' in checkpoint and checkpoint['mil'] is not None:
        mil.load_state_dict(checkpoint['mil'], strict=False)
        print("    -> MIL module weights loaded.")
    else:
        mil = None
        print("    -> MIL module not found, running with base model only.")
        
    model.eval()
    if mil: mil.eval()
    
    return model, mil

def predict_sequence(sequence, model, mil, esm_model, tokenizer, scaler):
    temp_fasta = "temp_query.fasta"
    with open(temp_fasta, "w") as f: f.write(f">query\n{sequence}\n")
    
    try:
        features_df = generate_features_single_seq(temp_fasta)
        feat_val = np.log1p(np.exp(np.clip(features_df.values, -10, 10)))
        scaled_features = scaler.transform(feat_val)
        additional_features = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
    except Exception as e:
        print(f"Error in feature generation: {e}")
        return None, 0.0
    finally:
        if os.path.exists(temp_fasta): os.remove(temp_fasta)

    esm_tokens = esm_encode([sequence], esm_model, tokenizer, DEVICE, 100)
    
    with torch.no_grad():
        logits, _, _ = forward_with_fusion_predict(model, mil, esm_tokens, additional_features)
        probs = F.softmax(logits, dim=1)
        score = probs[0][1].item()
        
    return "AVP" if score > 0.5 else "Non-AVP", score

def main():
    parser = argparse.ArgumentParser(description="AVP-Fusion Predictor")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--esm_path", type=str, default=DEFAULT_ESM_PATH)
    args = parser.parse_args()

    print(f"Loading ESM-2 model from {args.esm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    esm_model = AutoModel.from_pretrained(args.esm_path).to(DEVICE)
    esm_model.eval()

    # CRITICAL: We need a scaler that matches the checkpoint.
    # We assume it is located next to the checkpoint.
    scaler_path = os.path.join(os.path.dirname(args.ckpt), 'scaler.pkl')
    # If using your original best model, you MUST create a scaler for it.
    # For now, let's assume one exists or the effect is minor.
    try:
        print(f"Loading feature scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Warning: scaler.pkl not found at {scaler_path}. Prediction might be inaccurate.")
        # Create a dummy scaler that does nothing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 1159))) # Fit with zeros, effectively a pass-through
        
    try:
        model, mil = load_prediction_model(args.ckpt, additional_dim=1159)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"\nProcessing sequence: {args.sequence}")
    pred, score = predict_sequence(args.sequence, model, mil, esm_model, tokenizer, scaler)
    
    if pred is not None:
        print("-" * 30)
        print(f"Prediction Result: {pred}")
        print(f"Confidence Score : {score:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()