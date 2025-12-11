# util/data.py

import os
import torch
import random
import numpy as np
import pandas as pd
from util.seed import set_seed
import iFeatureOmegaCLI
from util.DDE import feature_DDE

set_seed()

# Z-scale dictionary (5D)
z_scale_dict = {
    'A': [-1.56, -1.67, -1.30, 0.81, -0.21], 'C': [0.12, 0.67, -2.05, -0.41, -0.09],
    'D': [1.06, 0.18, 1.23, -0.93, -0.89], 'E': [0.88, 0.73, 1.26, -1.07, -0.74],
    'F': [-0.97, 0.27, -1.04, -0.25, 0.76], 'G': [-1.22, -1.40, 1.23, -0.15, -1.13],
    'H': [0.64, -0.15, 1.05, -0.71, 0.94], 'I': [-0.77, 0.84, -1.78, 1.15, -0.04],
    'K': [0.55, 1.68, 1.83, -0.80, -0.56], 'L': [-0.72, 0.87, -1.41, 1.19, 0.23],
    'M': [-0.69, 0.62, -0.93, 0.45, 1.31], 'N': [0.93, -0.56, 0.60, -0.60, 0.89],
    'P': [0.45, -0.09, 0.70, -1.05, 0.54], 'Q': [0.90, 0.49, 0.83, -0.96, -0.19],
    'R': [1.84, 0.85, 1.41, -0.62, -1.07], 'S': [0.20, -1.08, 0.24, -0.66, 0.48],
    'T': [0.32, -0.45, 0.00, -0.73, 0.53], 'V': [-0.69, 1.30, -1.91, 1.15, -0.50],
    'W': [-0.39, 0.13, -0.73, 0.84, 2.10], 'Y': [-1.47, 0.24, -0.14, 0.02, 1.65]
}

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
aa_to_binary = {aa: np.eye(20)[i] for i, aa in enumerate(amino_acids)}


def load_data(data_path: str):
    """Load two-line or FASTA; return (pos_list, neg_list)."""
    avps, nonavps = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if lines and not lines[0].startswith(">"):  # two-line format
        assert len(lines) % 2 == 0, "two-line format requires even number of lines"
        for i in range(0, len(lines), 2):
            header, seq = lines[i], lines[i + 1]
            if "pos" in header.lower():
                avps.append(seq)
            else:
                nonavps.append(seq)
    else:  # FASTA
        header, buf = None, []
        for l in lines:
            if l.startswith(">"):
                if header is not None:
                    label = header.lower()
                    (avps if ("pos" in label or "avp" in label) else nonavps).append("".join(buf))
                header, buf = l[1:], []
            else:
                buf.append(l)
        if header is not None:
            label = header.lower()
            (avps if ("pos" in label or "avp" in label) else nonavps).append("".join(buf))

    random.shuffle(avps)
    random.shuffle(nonavps)
    return avps, nonavps


def get_sequences_from_fasta(file_path: str):
    """Return only sequences; supports FASTA and two-line formats."""
    lines = [l.strip() for l in open(file_path, "r", encoding="utf-8") if l.strip()]
    if not lines:
        return []
    if lines[0].startswith(">"):  # FASTA
        seqs, cur = [], []
        for l in lines:
            if l.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(l)
        if cur:
            seqs.append("".join(cur))
        return seqs
    else:  # two-line: header + sequence
        assert len(lines) % 2 == 0, "two-line format requires even number of lines"
        return [lines[i + 1] for i in range(0, len(lines), 2)]


def _to_fasta_if_needed(path: str) -> str:
    """Return a FASTA path. If input is two-line format, write a temp FASTA and return its path."""
    lines = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    if not lines:
        return path
    if lines[0].startswith(">"):
        return path  # already FASTA

    assert len(lines) % 2 == 0, "two-line format requires even number of lines"
    tmp = path + ".as_fasta.tmp"
    with open(tmp, "w", encoding="utf-8") as w:
        for i in range(0, len(lines), 2):
            header, seq = lines[i], lines[i + 1]
            if not header.startswith(">"):
                header = ">" + header
            w.write(f"{header}\n{seq}\n")
    return tmp


def generate_features(input_path: str) -> pd.DataFrame:
    """Build feature table; auto-skip any descriptor with encodings None/empty."""
    # ensure iFeature sees FASTA
    fasta_path = _to_fasta_if_needed(input_path)

    # iFeatureOmega descriptors
    AAC = iFeatureOmegaCLI.iProtein(fasta_path);          AAC.get_descriptor("AAC")
    CKSAAGP = iFeatureOmegaCLI.iProtein(fasta_path);      CKSAAGP.get_descriptor("CKSAAGP type 2")
    PAAC = iFeatureOmegaCLI.iProtein(fasta_path);         PAAC.get_descriptor("PAAC")
    QSOrder = iFeatureOmegaCLI.iProtein(fasta_path);      QSOrder.get_descriptor("QSOrder")
    GTPC = iFeatureOmegaCLI.iProtein(fasta_path);         GTPC.get_descriptor("GTPC type 2")
    DistancePair = iFeatureOmegaCLI.iProtein(fasta_path); DistancePair.get_descriptor("DistancePair")
    DPC = iFeatureOmegaCLI.iProtein(fasta_path);          DPC.get_descriptor("DPC type 2")

    # DDE (usually expects FASTA)
    dde = feature_DDE(fasta_path)

    # Binary / Z-scale from sequences (works for both formats)
    sequences = get_sequences_from_fasta(input_path)
    binary_features = [
        np.mean([aa_to_binary.get(aa, np.zeros(20)) for aa in seq], axis=0) for seq in sequences
    ]
    zscale_features = [
        np.mean([z_scale_dict.get(aa, [0.0] * 5) for aa in seq], axis=0) for seq in sequences
    ]
    Binary_df = pd.DataFrame(binary_features, columns=[f"Binary_{i}" for i in range(20)])
    Zscale_df = pd.DataFrame(zscale_features, columns=[f"Zscale_{i}" for i in range(5)])

    # safe concat
    def _safe_add(name: str, obj, frames: list):
        enc = getattr(obj, "encodings", None)
        if enc is None or not isinstance(enc, pd.DataFrame) or enc.empty:
            print(f"[WARN] skip feature: {name} (encodings=None/empty)")
            return
        frames.append(enc.reset_index(drop=True))

    frames = []
    _safe_add("AAC", AAC, frames)
    _safe_add("CKSAAGP", CKSAAGP, frames)
    _safe_add("PAAC", PAAC, frames)
    _safe_add("QSOrder", QSOrder, frames)
    _safe_add("GTPC", GTPC, frames)
    _safe_add("DistancePair", DistancePair, frames)
    _safe_add("DPC", DPC, frames)

    for name, df in [("DDE", dde), ("Binary", Binary_df), ("Zscale", Zscale_df)]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df.reset_index(drop=True))
        else:
            print(f"[WARN] skip feature: {name} (None/empty)")

    if not frames:
        if fasta_path.endswith(".as_fasta.tmp"):
            try:
                os.remove(fasta_path)
            except Exception:
                pass
        raise RuntimeError("No valid features produced (all encodings are None/empty).")

    result = pd.concat(frames, axis=1)
    result.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in result.columns]
    result = result.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # cleanup temp FASTA
    if fasta_path.endswith(".as_fasta.tmp"):
        try:
            os.remove(fasta_path)
        except Exception:
            pass

    return result


def esm_encode(sequences, model, tokenizer, device, max_length):
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

def generate_features_single_seq(fasta_path):
    return generate_features(fasta_path)
