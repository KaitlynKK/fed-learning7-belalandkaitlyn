import argparse
import random
import time
import os
from pathlib import Path
from shutil import copy2, rmtree
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import yaml
from ultralytics import YOLO


# -----------------------------
# Windows-safe copy helper
# -----------------------------
def same_file(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    try:
        s1, s2 = src.stat(), dst.stat()
        return s1.st_size == s2.st_size and int(s1.st_mtime) == int(s2.st_mtime)
    except Exception:
        return False


def safe_copy2(src: Path, dst: Path, retries: int = 10, delay: float = 0.2) -> None:
    """Copy with retries; skip if identical; handle Windows file-in-use (WinError 1224)."""
    if same_file(src, dst):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(retries):
        try:
            # If destination exists but differs, try to remove to avoid overwrite-on-open issues.
            if dst.exists():
                try:
                    dst.unlink()
                except Exception:
                    pass
            copy2(src, dst)
            return
        except OSError as e:
            last_err = e
            # 1224: file in use / user-mapped section open
            if getattr(e, "winerror", None) == 1224 or e.errno in (13,):
                time.sleep(delay * (1 + i * 0.3))
                continue
            else:
                raise
    # Final attempt failed
    raise last_err if last_err else OSError(f"Failed to copy {src} -> {dst}")


# -----------------------------
# Dataset preparation helpers
# -----------------------------
def prepare_from_labeled(
    labeled_dir: Path,
    train_dir: Path,
    val_dir: Path,
    split: float = 0.8,
    seed: int | None = 42,
) -> Tuple[int, int]:
    """
    Copy images and labels from a labeled_dir structured as:
      labeled_dir/
        images/*.jpg|.png
        labels/*.txt     (YOLO format)
    into train_dir and val_dir, each with images/ and labels/.

    Returns: (num_train_images, num_val_images)
    """
    img_dir = labeled_dir / "images"
    lbl_dir = labeled_dir / "labels"

    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        raise RuntimeError(
            f"No images found in {img_dir}. Expecting labeled dataset with images/ and labels/."
        )

    # Shuffle before splitting (optionally seeded for reproducibility)
    if seed is not None:
        random.seed(seed)
    random.shuffle(imgs)

    split_idx = max(1, int(len(imgs) * split)) if len(imgs) > 1 else 1
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:] or imgs[:1]  # ensure at least one val image

    for dest_root, subset in [(train_dir, train_imgs), (val_dir, val_imgs)]:
        (dest_root / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / "labels").mkdir(parents=True, exist_ok=True)
        for img in subset:
            # copy image (Windows-safe)
            safe_copy2(img, dest_root / "images" / img.name)
            # matching label
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                safe_copy2(lbl, dest_root / "labels" / lbl.name)
            else:
                # IMPORTANT: do NOT create empty labels (it tanks recall if objects are present)
                raise FileNotFoundError(f"Missing label for image {img.name} at {lbl}")

    return len(train_imgs), len(val_imgs)


def write_data_yaml(
    root: Path, train_dir: Path, val_dir: Path, nc: int, names: List[str]
) -> Path:
    """Write a minimal Ultralytics data.yaml pointing to train/val folders."""
    data = {
        "path": "",
        "train": str((train_dir / "images").resolve()).replace("\\", "/"),
        "val": str((val_dir / "images").resolve()).replace("\\", "/"),
        "nc": nc,
        "names": names,
    }
    yaml_path = root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


# -----------------------------
# Flower <-> YOLO param bridge
# -----------------------------
def ndarrays_to_state_dict(
    keys: List[str], nds: List[np.ndarray], ref_sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    if len(keys) != len(nds):
        raise RuntimeError(f"Param length mismatch: keys={len(keys)} vs nds={len(nds)}")
    out: Dict[str, torch.Tensor] = {}
    for k, arr in zip(keys, nds):
        if k not in ref_sd:
            raise KeyError(f"Missing key in reference state_dict: {k}")
        ref_t = ref_sd[k]
        t = torch.from_numpy(arr).to(ref_t.device).to(ref_t.dtype)
        if t.shape != ref_t.shape:
            raise RuntimeError(f"Shape mismatch for '{k}': got {t.shape}, expected {ref_t.shape}")
        out[k] = t
    return out


# -----------------------------
# Flower client
# -----------------------------
class YOLOFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        base_ckpt: Path,
        work_root: Path,
        labeled_dir: Path,
        split: float,
        epochs: int,
        imgsz: int,
        nc: int,
        names: List[str],
        seed: int | None,
        fresh_workdir: bool,
    ):
        self.base_ckpt = base_ckpt
        self.work_root = work_root
        self.labeled_dir = labeled_dir
        self.split = split
        self.epochs = epochs
        self.imgsz = imgsz
        self.nc = nc
        self.names = names
        self.seed = seed
        self.fresh_workdir = fresh_workdir

        # Fresh workdir avoids overwriting files that might be locked by Windows/Explorer/antivirus
        self.train_dir = work_root / "train"
        self.val_dir = work_root / "val"
        if self.fresh_workdir and work_root.exists():
            # Try to wipe, but be tolerant to transient locks
            for _ in range(5):
                try:
                    if self.train_dir.exists():
                        rmtree(self.train_dir, ignore_errors=True)
                    if self.val_dir.exists():
                        rmtree(self.val_dir, ignore_errors=True)
                    break
                except Exception:
                    time.sleep(0.2)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset (shuffled 80/20)
        ntr, nval = prepare_from_labeled(
            labeled_dir=self.labeled_dir,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            split=self.split,
            seed=self.seed,
        )
        print(f"[CLIENT] Prepared dataset: train={ntr}, val={nval}")

        # data.yaml
        self.data_yaml = write_data_yaml(
            root=self.work_root,
            train_dir=self.train_dir,
            val_dir=self.val_dir,
            nc=self.nc,
            names=self.names,
        )
        print(f"[CLIENT] data.yaml -> {self.data_yaml}")

        # Load YOLO (unfused architecture)
        self.model = YOLO(str(self.base_ckpt))
        print("[CLIENT] First 10 model keys:", list(self.model.model.state_dict().keys())[:10])

        # Capture a stable list of state_dict keys/ordering
        self._keys = list(self.model.model.state_dict().keys())

    # --- Flower API ---
    def get_parameters(self, config):
        sd = self.model.model.state_dict()
        return [sd[k].detach().cpu().numpy() for k in self._keys]

    def set_parameters(self, parameters: List[np.ndarray]):
        # Ensure our model has the expected (unfused) architecture
        ref_sd = self.model.model.state_dict()
        if any(k not in ref_sd for k in self._keys):
            print("[CLIENT][WARN] Detected fused/mismatched model; reloading base architecture.")
            self.model = YOLO(str(self.base_ckpt))
            ref_sd = self.model.model.state_dict()
        new_sd = ndarrays_to_state_dict(self._keys, parameters, ref_sd)
        self.model.model.load_state_dict(new_sd, strict=True)

    def fit(self, parameters, config):
        # Receive global params
        self.set_parameters(parameters)

        # Train
        print(f"[CLIENT] Training for {self.epochs} epochs @ {self.imgsz}px")
        self.model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            imgsz=self.imgsz,
            verbose=True,
        )

        # Do NOT swap to a new YOLO(best.pt) object here (avoids BN fusion key drift)

        # Return updated params
        sd = self.model.model.state_dict()
        metrics = {"train_epochs": float(self.epochs)}
        num_train_imgs = len(list((self.train_dir / "images").glob("*")))
        return [sd[k].detach().cpu().numpy() for k in self._keys], num_train_imgs, metrics

    def evaluate(self, parameters, config):
        # Receive latest global params
        self.set_parameters(parameters)

        # Evaluate on our local validation split
        results = self.model.val(data=str(self.data_yaml), imgsz=self.imgsz)

        # Extract main metrics
        p = float(results.box.p.mean().item())
        r = float(results.box.r.mean().item())
        map50 = float(results.box.map50.mean().item())
        map5095 = float(results.box.map.mean().item())

        metrics = {
            "precision": p,
            "recall": r,
            "map50": map50,
            "map50_95": map5095,
        }
        num_val = len(list((self.val_dir / "images").glob("*")))
        print(f"[CLIENT] Eval -> P:{p:.3f} R:{r:.3f} mAP50:{map50:.3f} mAP50-95:{map5095:.3f}")
        return 0.0, num_val, metrics


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="localhost:8080", help="Flower server host:port")
    ap.add_argument("--model", default="model/my_model.pt", help="Base YOLO checkpoint path")
    ap.add_argument("--labeled_dir", default="data/labelfront2", help="Labeled dataset root (images/, labels/)")
    ap.add_argument("--work_root", default="output/client1", help="Client working dir")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--names", nargs="*", default=["object"])
    ap.add_argument("--seed", type=int, default=42)  # set to -1 to make it None (fully random)
    ap.add_argument(
        "--fresh_workdir",
        action="store_true",
        help="Wipe and recreate train/val inside work_root before preparing dataset (avoids locked-file overwrites on Windows).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    seed = None if (args.seed is not None and args.seed < 0) else args.seed

    client = YOLOFlowerClient(
        base_ckpt=Path(args.model),
        work_root=Path(args.work_root),
        labeled_dir=Path(args.labeled_dir),
        split=args.split,
        epochs=args.epochs,
        imgsz=args.imgsz,
        nc=args.nc,
        names=args.names,
        seed=seed,
        fresh_workdir=args.fresh_workdir,
    )

    print(f"[CLIENT] Connecting to server at {args.server}")
    # Prefer modern Flower API, but fall back to numpy_client for older installs.
    try:
        fl.client.start_client(server_address=args.server, client=client.to_client())
    except Exception as e:
        print(f"[CLIENT][WARN] start_client failed ({e}); falling back to start_numpy_client.")
        fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()