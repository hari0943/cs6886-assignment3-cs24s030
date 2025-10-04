import os, re, sys, time, subprocess, argparse
from pathlib import Path
import wandb

# Parse key metrics from prune.py stdout
PATS = {
    "final_acc":     re.compile(r"Final accuracy.*?:\s*([0-9.]+)%"),
    "ptq_acc":       re.compile(r"After Activation PTQ:\s*([0-9.]+)%"),
    "quant_acc":     re.compile(r"After Quantization:\s*([0-9.]+)%"),
    "weight_mb":     re.compile(r"Weight bits:\s*([0-9.]+)\s*MB"),
    "act_mb":        re.compile(r"Activation bits:\s*([0-9.]+)\s*MB"),
    "weight_comp":   re.compile(r"Weight Compression:\s*([0-9.]+)×"),
    "act_comp":      re.compile(r"Activation Compression:\s*([0-9.]+)×"),
    "tot_comp":      re.compile(r"Total Compression:\s*([0-9.]+)×"),
    # Either of these may be present depending on what prune.py prints:
    "total_ram_mb":      re.compile(r"Total RAM:\s*([0-9.]+)\s*MB"),
    "total_model_size":  re.compile(r"Compressed Model size:\s*([0-9.]+)\s*MB"),
}

def parse_metrics(text: str):
    m = {}
    for k, pat in PATS.items():
        hit = pat.search(text)
        if hit:
            m[k] = float(hit.group(1))

    # sweep target
    val_acc = m.get("final_acc") or m.get("ptq_acc") or m.get("quant_acc")
    if val_acc is not None:
        m["val_acc"] = val_acc

    # friendly names for parallel-coordinates axes (ratios)
    if "tot_comp" in m:
        m["total_compression_ratio"] = m["tot_comp"]
    if "weight_comp" in m:
        m["weight_compression_ratio"] = m["weight_comp"]
    if "act_comp" in m:
        m["activation_compression_ratio"] = m["act_comp"]

    # total model size axis (prefer explicit prints, else fallback)
    if "total_model_size" in m:
        m["total_model_size_mb"] = m["total_model_size"]
    elif "total_ram_mb" in m:
        m["total_model_size_mb"] = m["total_ram_mb"]
    elif "weight_mb" in m and "act_mb" in m:
        m["total_model_size_mb"] = m["weight_mb"] + m["act_mb"]

    return m

def sweep_train():
    run = wandb.init()
    cfg = wandb.config

    # Build comma strings for prune.py CLI
    bit_bins_str = ",".join(str(x) for x in cfg.bit_bins)
    prop_str     = ",".join(str(x) for x in cfg.bin_proportions)

    cmd = [
        sys.executable, cfg.prune_script,
        "--checkpoint", cfg.checkpoint,
        "--quant-mode", "hawq",
        "--bit-bins", bit_bins_str,
        "--bin-proportions", prop_str,
        "--power-iters", str(cfg.power_iters),
        "--eig-batches", str(cfg.eig_batches),
        "--sparsity", str(cfg.sparsity),
        "--act-bits", str(cfg.act_bits),
        "--act-calib-batches", str(cfg.act_calib_batches),
    ]
    print("Running:", " ".join(cmd), flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    out = proc.stdout
    print(out)

    metrics = parse_metrics(out)
    # Log knobs so they appear as axes
    metrics.update({
        "returncode": proc.returncode,
        "seconds": dt,
        "sparsity": cfg.sparsity,
        "act_bits": cfg.act_bits,
        "act_calib_batches": cfg.act_calib_batches,
        # For parallel-coords, log string views + a numeric helper:
        "bit_bins_str": bit_bins_str,
        "bin_proportions_str": prop_str,
        "avg_weight_bits": sum(cfg.bit_bins)/len(cfg.bit_bins),
        "power_iters": cfg.power_iters,
        "eig_batches": cfg.eig_batches,
    })
    wandb.log(metrics)
    run.finish()

if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./runs/wandb_mobilenetv2_cifar10/best.pth", help="runs/.../best.pth")
    ap.add_argument("--prune-script", default="prune.py")
    ap.add_argument("--project", default="mv2-hawq")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--count", type=int, default=40)
    args = ap.parse_args()

    # Candidate sets to explore
    bit_bins_candidates = [
        [8,6,4,3,2],
        [8,7,6,5,4],
        [6,5,4,3,2],
        [8,6,5,4,2],
    ]
    prop_candidates = [
        [0.15,0.25,0.30,0.30],
        [0.20,0.30,0.40,0.10],
        [0.40,0.30,0.20,0.10],
        [0.10,0.20,0.30,0.40],
        [0.05,0.20,0.35,0.40],
    ]

    sweep_config = {
        "name": "hawq-bayes-mv2",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            # Sample one of several bin blueprints
            "bit_bins": {"values": bit_bins_candidates},
            "bin_proportions": {"values": prop_candidates},
            # Cheap-to-fair eigen configs
            "power_iters": {"values": [2]},
            "eig_batches": {"values": [2]},
            # Regular knobs
            "sparsity": {"values": [0.7, 0.8, 0.9]},
            "act_bits": {"values": [0,6,8]},  # 0 disables activation PTQ
            "act_calib_batches": {"values": [30]},
            # Static paths
            "checkpoint":  {"value": str(Path(args.checkpoint).resolve())},
            "prune_script":{"value": str(Path(args.prune_script).resolve())},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, function=sweep_train, count=args.count)
    wandb.finish()
