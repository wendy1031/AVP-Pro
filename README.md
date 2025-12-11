# **AVP-Fusion**

### **Adaptive Multi-Modal Fusion and Contrastive Learning for Two-Stage Antiviral Peptide Identification**

---

## ** Introduction**

**AVP-Fusion** is a deep learning framework designed for the high-accuracy identification of antiviral peptides (AVPs).  
The core of this project is a robust model that classifies whether a given peptide sequence possesses antiviral properties.

While the full framework supports a two-stage process, this repository provides the complete implementation for:

### 🔹 **Stage 1 — General AVP Identification**

**Key features include:**
- **Adaptive Gating Mechanism** for intelligently fusing local (CNN) and global (BiLSTM) sequence features.  
- **OHEM-based Contrastive Learning** to enhance the model’s ability to distinguish difficult samples.  
- **Multi-Modal Feature Space** combining:
  - Deep evolutionary representations from **ESM-2**  
  - Ten groups of physicochemical descriptors  

---

## ** Environment Setup**

We recommend using Conda to manage the environment.

### **1. Create Conda Environment**
```bash
conda create -n avp_fusion python=3.8
conda activate avp_fusion
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download ESM-2 Model (for Offline Use)**

The official **esm2_t30_150M_UR50D** model has already been downloaded and included in this project.  
If you prefer to download it yourself, you can get it from:

👉 https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/main


Place the downloaded folder under:

```
esm2_t30_150M_UR50D/
```

---

## ** Dataset**

All Stage 1 training/testing data are stored in:

```
dataset/Set 1/
```

---

## **🔧 Usage**

---

# **1.  Training Stage 1 (General Model)**

Train the model from scratch:

```bash
python train_stage1.py
```

### **Output files (auto-generated):**

Saved under:

```
checkpoints/stage1/
```

- `best_model_stage1.pth` — Best model (based on MCC on the test set)  
- `scaler.pkl` — Fitted feature scaler (required for prediction)  

---

# **2. One-Click Prediction (Inference)**

After completing training, you can run prediction using `predict.py`.

This requires:

- `best_model_stage1.pth` — trained model weights  
- `scaler.pkl` — the feature scaler fitted during training  

### **Example:**
```bash
python predict.py --sequence "ANKFNQALGAMQTGFTTTNEAFRKVQDAVNNNAQALSKLASE" --ckpt checkpoints/stage1/best_model_stage1.pth
```

---

