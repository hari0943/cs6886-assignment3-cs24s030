# MobileNetV2 on CIFAR-10 — Training, Evaluation & Compression (Original Code)

This README documents the **original code layout** with a clear, step-by-step workflow. It uses **pip (no conda)** and covers training, compression (pruning + quantization + optional activation PTQ / HAWQ), and W&B sweeps.

---

## 0) Prerequisites
- Python **3.10** (recommended)
- NVIDIA GPU + CUDA 12.1 (optional but recommended). CPU works too (slower).
- Git installed; Weights & Biases account if you plan to run sweeps.

---

## 1) Setup (pip only)

```bash
# Create & activate a virtualenv
python3.10 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip
```

### Install dependencies — GPU (CUDA 12.1 wheels)
```bash
pip install -r requirements.txt 

---

## 2) Reproducibility: Seed configuration

In your shell:
```bash
export PYTHONHASHSEED=42
```

Then pass `--seed 42` to training. (For strict determinism, disable CuDNN benchmarking and enable deterministic algorithms in code if needed.)

---

## 3) Train the baseline

The training script handles dataset download and normalization.

```bash
python MobilenetV2_train.py \
  --data ./data \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.05 \
  --warmup-epochs 5 \
  --width-mult 1.0 \
  --out ./runs/mobilenetv2_cifar10 \
  --seed 42
```

**Outputs**
- Checkpoints: `./runs/mobilenetv2_cifar10/best.pth` and `last.pth`
- Console logs show train/val loss and Acc@1 per epoch

---

## 4) Evaluate & compress (uniform bits + optional activation PTQ)

The compression script loads your trained checkpoint, **prints baseline accuracy**, then applies pruning and quantization.

**Uniform bits (weights) + Activation PTQ (optional):**
```bash
python prune.py \
  --checkpoint ./runs/mobilenetv2_cifar10/best.pth \
  --sparsity 0.9 \
  --conv-bits 8 \
  --fc-bits 5 \
  --act-bits 8 \
  --act-calib-batches 30
```

**What you’ll see printed**
- Baseline Accuracy
- After Pruning / After Quantization (and **After Activation PTQ** if enabled)
- **MEMORY COMPRESSION SUMMARY** with:
  - Compressed weight/activation size (MB)
  - Original FP32 sizes (MB)
  - **Compression ratios** (weights / activations / total)
  - Approx. **Compressed Model size (MB)**

*Notes*
- `--sparsity` is the target global magnitude pruning ratio.
- `--conv-bits` and `--fc-bits` set K-means **weight sharing** (e.g., 8→256 clusters).
- `--act-bits` inserts post-training **activation quantizers** and runs calibration for the specified batches; set `--act-bits 0` to skip.

---

## 5) Mixed-precision compression (HAWQ)

To assign **per-layer** bit-widths via Hessian-based sensitivity:

```bash
python prune.py \
  --checkpoint ./runs/mobilenetv2_cifar10/best.pth \
  --quant-mode hawq \
  --bit-bins 8,6,4,3,2 \
  --bin-proportions 0.15,0.25,0.30,0.30 \
  --power-iters 8 \
  --eig-batches 2 \
  --sparsity 0.9 \
  --act-bits 8 \
  --act-calib-batches 30
```

**What you’ll see printed**
- Per-layer sensitivity and **bit assignment**
- The same accuracy and **MEMORY COMPRESSION SUMMARY** section as above

---

## 6) (Optional) W&B sweeps

Login once:
```bash
wandb login
```

### Uniform quantization sweep
```bash
python sweep.py \
  --checkpoint ./runs/mobilenetv2_cifar10/best.pth \
  --prune-script prune.py \
  --project mv2-compression \
  --count 20
```

### HAWQ sweep
```bash
python sweep_hawq.py \
  --checkpoint ./runs/mobilenetv2_cifar10/best.pth \
  --prune-script prune.py \
  --project mv2-hawq \
  --count 40
```

**Notes**
- Sweeps parse specific console lines from `prune.py` (e.g., “Weight bits: … MB”, “Total Compression: …×”). If you change those print formats, update the regex in the sweep scripts.

---

## 7) Tips & gotchas

- **First/last-layer pruning**: typically excluded for stability (default in script).
- **Activation PTQ**: Use a representative calibration set (`--act-calib-batches`) for better accuracy.
- **Model path**: Always use `best.pth` from training for compression.
- **GPU vs CPU**: Results/accuracy are identical; training/compression is just much faster on GPU.
- **File sizes vs RAM footprint**: Codebooks + indices reduce weight storage; runtime RAM also depends on activation sizes and framework overhead.

---

## 8) One-page quickstart

```bash
# 1) Env (pip)
python3.10 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
pip install -r requirements.txt

# 2) Train
export PYTHONHASHSEED=42
python MobilenetV2_train.py --data ./data --epochs 200 --batch-size 128 --lr 0.05 --warmup-epochs 5 --width-mult 1.0 --out ./runs/mobilenetv2_cifar10 --seed 42

# 3) Compress (uniform + PTQ)
python prune.py --checkpoint ./runs/mobilenetv2_cifar10/best.pth --sparsity 0.9 --conv-bits 8 --fc-bits 5 --act-bits 8 --act-calib-batches 30

# 4) (Alt) Compress with HAWQ
python prune.py --checkpoint ./runs/mobilenetv2_cifar10/best.pth --quant-mode hawq \
  --bit-bins 8,6,4,3,2 --bin-proportions 0.15,0.25,0.30,0.30 --power-iters 8 --eig-batches 2 \
  --sparsity 0.9 --act-bits 8 --act-calib-batches 30

# 5) (Optional) Sweeps
wandb login
python sweep.py --checkpoint ./runs/mobilenetv2_cifar10/best.pth --prune-script prune.py --project mv2-compression --count 20
python sweep_hawq.py --checkpoint ./runs/mobilenetv2_cifar10/best.pth --prune-script prune.py --project mv2-hawq --count 40
```

---

If anything is unclear or you want the README customized for your cluster (Slurm, multi-GPU, etc.), let me know.
