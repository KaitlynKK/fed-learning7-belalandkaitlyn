import os
import cv2
import argparse
from pathlib import Path
from shutil import copy2
import yaml
import numpy as np
import torch
import flwr as fl
from ultralytics import YOLO

# -------- dataset prep from labeled_dir --------
def prepare_from_labeled(labeled_dir: Path, train_folder: Path, val_folder: Path, split: float = 0.8):
    img_dir = labeled_dir / "images"
    lbl_dir = labeled_dir / "labels"
    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}. Expecting labeled dataset with images/ and labels/.")

    split_idx = max(1, int(len(imgs) * split)) if len(imgs) > 1 else 1
    sets = [("train", imgs[:split_idx]), ("val", imgs[split_idx:] or imgs[:1])]

    for split_name, items in sets:
        dest_root = train_folder if split_name == "train" else val_folder
        (dest_root / "images").mkdir(parents=True, exist_ok=True)
        (dest_root / "labels").mkdir(parents=True, exist_ok=True)
        for img in items:
            lbl = lbl_dir / (img.stem + ".txt")
            copy2(img, dest_root / "images" / img.name)
            if lbl.exists():
                copy2(lbl, dest_root / "labels" / lbl.name)

# -------- helpers to move weights between Flower <-> PyTorch --------
def state_dict_to_ndarrays(sd):
    # Stable order by iterating over items()
    return [v.detach().cpu().numpy() for _, v in sd.items()]

def ndarrays_to_state_dict_like(model, nds):
    base_sd = model.model.state_dict()
    if len(nds) != len(base_sd):
        raise ValueError("Mismatched parameter count.")
    new_sd = {}
    for (k, v), arr in zip(base_sd.items(), nds):
        t = torch.from_numpy(arr).to(v.device).to(v.dtype)
        if t.shape != v.shape:
            raise ValueError(f"Shape mismatch for {k}: got {t.shape}, expected {v.shape}")
        new_sd[k] = t
    return new_sd

# ---------------- Flower client ----------------
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model)  # MUST be same arch across all clients
        self.labeled_dir = Path(args.labeled_dir)  # e.g., data/labelfront1
        self.work_root = Path(args.work_root)      # e.g., output/client1
        self.work_root.mkdir(parents=True, exist_ok=True)

    def get_parameters(self, config):
        return state_dict_to_ndarrays(self.model.model.state_dict())

    def set_parameters(self, parameters):
        new_sd = ndarrays_to_state_dict_like(self.model, parameters)
        self.model.model.load_state_dict(new_sd, strict=True)

    def fit(self, parameters, config):
        print("[CLIENT] FIT STARTED")
        if parameters:
            self.set_parameters(parameters)

        train_folder = self.work_root / "train"
        val_folder = self.work_root / "val"
        out_root = self.work_root / "runs"
        for d in [train_folder, val_folder, out_root]:
            d.mkdir(parents=True, exist_ok=True)

        # Build train/val from labeled dir
        prepare_from_labeled(self.labeled_dir, train_folder, val_folder, split=self.args.split)

        # Write data.yaml
        data_yaml_path = out_root / "data.yaml"
        data_yaml = {
            "train": str(train_folder.resolve()).replace("\\", "/"),
            "val": str(val_folder.resolve()).replace("\\", "/"),
            "nc": self.args.nc,
            "names": self.args.names.split(","),
        }
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Train YOLO locally on this client's data
        self.model.train(
            data=str(data_yaml_path.resolve()),
            epochs=self.args.epochs,
            imgsz=self.args.imgsz,
            project=str(out_root),
            name="train_result",
            exist_ok=True,
            plots=True,
            save=True,
        )

        # Return updated weights for aggregation
        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters, config):
        # optional: implement a proper val here
        if parameters:
            self.set_parameters(parameters)
        return 0.0, 1, {}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="localhost:8080")
    p.add_argument("--model", default="model/my_model.pt", help="Base YOLO checkpoint")
    p.add_argument("--labeled_dir", required=True, help="Path to labeled dataset (images/ & labels/)")
    p.add_argument("--work_root", required=True, help="Client working dir for outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--names", default="object")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    fl.client.start_numpy_client(server_address=args.server, client=YOLOClient(args))
