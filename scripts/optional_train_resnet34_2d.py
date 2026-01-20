from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from common import ensure_dir, load_labels_csv, save_json


PARAM_NAMES = ["lat", "lon"]


class Geo2DDataset(Dataset):
    def __init__(self, images_dir: Path, labels_csv: Path, img_size: int = 224):
        self.images_dir = images_dir
        self.df = load_labels_csv(labels_csv)

        missing = [c for c in ["filename"] + PARAM_NAMES if c not in self.df.columns]
        if missing:
            raise ValueError(f"Labels CSV missing columns: {missing}")

        self.df = self.df.dropna(subset=PARAM_NAMES).reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        fname = str(row["filename"])
        img = Image.open(self.images_dir / fname).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([row["lat"], row["lon"]], dtype=torch.float32)
        return x, y


class ResNet34Regressor(nn.Module):
    def __init__(self, num_params: int = 2, pretrained: bool = False):
        super().__init__()
        if pretrained:
            # May download weights (internet required)
            try:
                self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                self.base_model = models.resnet34(pretrained=True)
        else:
            try:
                self.base_model = models.resnet34(weights=None)
            except TypeError:
                self.base_model = models.resnet34(pretrained=False)

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.base_model(x)
        return self.regressor(feats)


def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            errs.append(torch.abs(pred - yb).detach().cpu().numpy())
    e = np.vstack(errs)
    return float(e.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Optional: train ResNet34 2D regressor (lat/lon).")
    ap.add_argument("--dataset_root", type=str, default="dataset_v1")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone (may download).")
    ap.add_argument("--outdir", type=str, default="results/train_resnet34_2d")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_csv = dataset_root / "labels" / "train" / "_train_annotations.csv"
    val_csv = dataset_root / "labels" / "val" / "_val_annotations.csv"

    outdir = ensure_dir(Path(args.outdir))
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Geo2DDataset(train_images, train_csv, img_size=args.img_size)
    val_ds = Geo2DDataset(val_images, val_csv, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResNet34Regressor(num_params=2, pretrained=args.pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    patience = 0
    best_path = outdir / "best_resnet34_2d.pth"

    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses):.5f}"})

        val_mae = evaluate_mae(model, val_loader, device)
        print(f"Val MAE: {val_mae:.6f}")

        if val_mae < best_val:
            best_val = val_mae
            patience = 0
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best weights: {best_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print("🛑 Early stopping.")
                break

    save_json({
        "model_name": "ResNet34Regressor",
        "task": "2D regression (lat,lon)",
        "best_val_mae": best_val,
        "weights": str(best_path),
        "dataset_root": str(dataset_root),
        "epochs_ran": epoch + 1,
        "device": str(device),
    }, outdir / "train_summary.json")

    print(f"Done. Best val MAE: {best_val:.6f}")
    print(f"Next: evaluate with scripts/2_eval_resnet34_2d.py --weights {best_path}")


if __name__ == "__main__":
    main()
