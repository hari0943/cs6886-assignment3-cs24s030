import os, re, sys, time, subprocess, argparse
from pathlib import Path
import wandb
# --- minimal stdout parser (matches your prints) ---
PATS = {
    "final_acc": re.compile(r"Final accuracy.*?:\s*([0-9.]+)%"),
    "ptq_acc":   re.compile(r"After Activation PTQ:\s*([0-9.]+)%"),
    "quant_acc": re.compile(r"After Quantization:\s*([0-9.]+)%"),
    "weight_mb": re.compile(r"Weight bits:\s*([0-9.]+)\s*MB"),
    "act_mb":    re.compile(r"Activation bits:\s*([0-9.]+)\s*MB"),
    "weight_comp": re.compile(r"Weight Compression:\s*([0-9.]+)×"),
    "act_comp":    re.compile(r"Activation Compression:\s*([0-9.]+)×"),
    "tot_comp":    re.compile(r"Total Compression:\s*([0-9.]+)×"),
    "total_model_size": re.compile(r"Compressed Model size:\s*([0-9.]+)\s*MB"),
}

def parse_metrics(text: str):
    m = {}
    for k, pat in PATS.items():
        hit = pat.search(text)
        if hit:
            m[k] = float(hit.group(1))

    # sweep metric
    val_acc = m.get("final_acc") or m.get("ptq_acc") or m.get("quant_acc")
    if val_acc is not None:
        m["val_acc"] = val_acc

    # compression ratios with names you want on the parallel plot
    if "tot_comp" in m:
        m["total_compression_ratio"] = m["tot_comp"]
    if "weight_comp" in m:
        m["weight_compression_ratio"] = m["weight_comp"]
    if "act_comp" in m:
        m["activation_compression_ratio"] = m["act_comp"]

    # total model size axis
    if "total_ram_mb" in m:
        m["total_model_size_mb"] = m["total_ram_mb"]
    elif "weight_mb" in m and "act_mb" in m:
        m["total_model_size_mb"] = m["weight_mb"] + m["act_mb"]

    return m

# --- the train function called by the agent ---
def sweep_train():
    run = wandb.init()
    cfg = wandb.config
    prune_py = cfg.prune_script
    checkpoint = cfg.checkpoint

    cmd = [
        sys.executable, prune_py,
        "--checkpoint", checkpoint,
        "--quant-mode", "uniform",
        "--conv-bits", str(cfg.conv_bits),
        "--fc-bits", str(cfg.fc_bits),
        "--sparsity", f"{cfg.sparsity}",
        "--act-bits", str(cfg.act_bits),
        "--act-calib-batches", str(cfg.act_calib_batches),
    ]
    print("Running:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0

    out = proc.stdout
    print(out)  # so you still get full logs in your console

    metrics = parse_metrics(out)
    metrics.update({
        "seconds": dt,
        "returncode": proc.returncode,
        # log the knobs too
        "conv_bits": cfg.conv_bits,
        "fc_bits": cfg.fc_bits,
        "sparsity": cfg.sparsity,
        "act_bits": cfg.act_bits,
    })
    wandb.log(metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default='./runs/wandb_mobilenetv2_cifar10/best.pth', help="runs/.../best.pth")
    ap.add_argument("--prune-script", default="prune.py")
    ap.add_argument("--project", default="mv2-compression")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--count", type=int, default=1, help="number of trials")
    args = ap.parse_args()

    # define the sweep (Bayesian like your example)
    sweep_config = {
        "name": "mv2-uniform-bayes",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "conv_bits":   {"values": [5]},
            "fc_bits":     {"values": [2]},
            "sparsity":    {"values": [0.9]},  # 0.0..0.9
            "act_bits":    {"values": [8]},
            "act_calib_batches": {"value": 30},
            # static paths
            "checkpoint":  {"value": str(Path(args.checkpoint).resolve())},
            "prune_script":{"value": str(Path(args.prune_script).resolve())},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)

    # wandb agent: will call sweep_train() with sampled configs
    wandb.agent(sweep_id, function=sweep_train, count=args.count)
    wandb.finish()
