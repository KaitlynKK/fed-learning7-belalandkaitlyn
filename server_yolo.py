# server_yolo.py — Federated YOLOv8 Server with Evaluation Support
# Author: Belal Nur & Kaitlyn
# Updated: 2025-10
# Description:
# - Runs Flower federated rounds using YOLOv8 clients
# - Aggregates weights, saves global model, and performs evaluation on test videos
# - Supports --test_only mode for running post-training evaluation only

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from ultralytics import YOLO
import yaml

# ============================================================
# FEDERATED STRATEGY
# ============================================================

class FedAvgWithSave(fl.server.strategy.FedAvg):
    """Flower strategy with saved final parameters after last round."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
        return aggregated, metrics


# ============================================================
# SAVE & EVALUATION HELPERS
# ============================================================

def save_final_model(params, base_ckpt="model/my_model.pt", out_path="static/output/final_model.pt"):
    """Convert Flower aggregated params back into a YOLO model checkpoint."""
    print("[SERVER] Saving final aggregated model...")
    base_model = YOLO(base_ckpt)
    base_sd = base_model.model.state_dict()
    ndarrays = parameters_to_ndarrays(params)

    if len(ndarrays) != len(base_sd):
        raise RuntimeError(f"[ERROR] Param count mismatch: got {len(ndarrays)} expected {len(base_sd)}")

    new_sd: Dict[str, torch.Tensor] = {}
    for (k, v), arr in zip(base_sd.items(), ndarrays):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise RuntimeError(f"Shape mismatch at '{k}': got {t.shape}, expected {v.shape}")
        new_sd[k] = t

    base_model.model.load_state_dict(new_sd, strict=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base_model.save(out_path)
    print(f"[SERVER] ✅ Final model saved: {out_path}")
    return out_path


def make_eval_checkpoint(src_path, dst_path, fuse=True):
    """Create an eval/deploy-only checkpoint (optional Conv+BN fusing)."""
    mdl = YOLO(src_path)
    if fuse:
        try:
            mdl.model.fuse()
            print("[SERVER] Fused Conv+BN for eval checkpoint.")
        except Exception as e:
            print(f"[SERVER][WARN] Could not fuse (continuing unfused): {e}")

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    mdl.save(dst_path)
    print(f"[SERVER] ✅ Eval checkpoint created at: {dst_path}")
    return dst_path


# ============================================================
# TESTING UTILITIES
# ============================================================

VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]

def ensure_test_yaml(test_root: Path,
                     yaml_path: Path = Path("data/test_data.yaml"),
                     class_names=None) -> Path:
    """Generate YOLO data.yaml for labelled test videos using ABSOLUTE paths."""
    if class_names is None:
        class_names = ["object"]

    imgs: List[str] = []
    for root, _, files in os.walk(test_root):
        if os.path.basename(root).lower() == "images":
            for f in files:
                if Path(f).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    imgs.append(str(Path(root, f).resolve().as_posix()))

    if not imgs:
        raise FileNotFoundError(f"No labelled images found under {test_root}")

    # Write absolute list file alongside yaml (but absolute content inside)
    list_file = yaml_path.with_name("_auto_test_list.txt").resolve()
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(imgs)))

    # IMPORTANT: keep 'path' empty so Ultralytics won't prefix it.
    content = {
        "path": "",  # avoid prefixing; all entries are absolute
        "train": [],  # not used
        "val": list_file.as_posix(),  # absolute text file of image paths
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = yaml_path.resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)
    return yaml_path


def test_final_model(model_path="static/output/final_model_eval.pt",
                     test_videos_dir="data/test_videos",
                     test_yaml_path="data/test_data.yaml"):
    """Evaluate trained YOLO model on labelled test videos."""
    model = YOLO(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"static/output/test_results_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SERVER] 🔍 Evaluating model: {model_path}")

    # Inference on videos
    tvdir = Path(test_videos_dir)
    for root, _, files in os.walk(tvdir):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTS:
                src = str(Path(root, fname))
                print(f"[SERVER] Predicting on: {src}")
                try:
                    model.predict(source=src, save=True, save_dir=str(save_dir / Path(fname).stem), imgsz=640)
                except Exception as e:
                    print(f"[SERVER][WARN] Skipped {fname}: {e}")

    # Metrics on labelled data (absolute paths; no prefixing)
    yaml_abs = ensure_test_yaml(tvdir.resolve(), Path(test_yaml_path).resolve())
    metrics = model.val(data=yaml_abs.as_posix(), imgsz=640)

    summary_path = save_dir / "test_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== TEST PERFORMANCE SUMMARY ===\n")
        f.write("\n--- Overall Metrics (labelled test_videos) ---\n")
        f.write(f"Precision : {metrics.box.p.mean():.3f}\n")
        f.write(f"Recall    : {metrics.box.r.mean():.3f}\n")
        f.write(f"mAP50     : {metrics.box.map50.mean():.3f}\n")
        f.write(f"mAP50-95  : {metrics.box.map.mean():.3f}\n")

    print(f"\n📄 Results saved to: {summary_path}")
    return summary_path


# ============================================================
# MAIN EXECUTION
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Federated YOLOv8 Server with Test Support")
    ap.add_argument("--server_address", default="0.0.0.0:8080", help="Flower gRPC server address")
    ap.add_argument("--num_rounds", type=int, default=4)
    ap.add_argument("--base_ckpt", default="model/my_model.pt")
    ap.add_argument("--final_out", default="static/output/final_model.pt")
    ap.add_argument("--eval_ckpt", default="static/output/final_model_eval.pt")
    ap.add_argument("--eval_fuse", action="store_true", help="Fuse Conv+BN for eval checkpoint")
    ap.add_argument("--test_videos_dir", default="data/test_videos")
    ap.add_argument("--test_yaml", default="data/test_data.yaml")
    ap.add_argument("--test_only", action="store_true", help="Run test only on existing model")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # TEST ONLY MODE
    if args.test_only:
        print("[SERVER] Running in TEST-ONLY mode...")
        target = args.eval_ckpt if Path(args.eval_ckpt).exists() else args.final_out
        if not Path(target).exists():
            raise FileNotFoundError(f"No model found at {target}")
        # Optionally fuse if user asked for it and eval_ckpt missing
        if args.eval_fuse and not Path(args.eval_ckpt).exists():
            target = make_eval_checkpoint(target, args.eval_ckpt, fuse=True)
        test_final_model(model_path=target,
                         test_videos_dir=args.test_videos_dir,
                         test_yaml_path=args.test_yaml)
        raise SystemExit(0)

    # TRAINING MODE
    print("[SERVER] Starting Flower federated server...")
    strategy = FedAvgWithSave(min_fit_clients=1, min_available_clients=1)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # Save and (optionally) create an eval checkpoint after final aggregation
    if strategy.final_parameters is None:
        print("[SERVER][WARN] No final parameters found.")
        raise SystemExit(0)

    final_model = save_final_model(strategy.final_parameters, args.base_ckpt, args.final_out)

    if args.eval_fuse:
        make_eval_checkpoint(final_model, args.eval_ckpt, fuse=True)
    else:
        print("[SERVER] Skipping fusion during training to keep BN layers intact.")
