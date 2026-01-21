from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _infer_model_name_from_path(p: Path) -> str:
    s = str(p).lower()
    if "resnet18" in s:
        return "ResNet18"
    if "resnet34" in s:
        return "ResNet34"
    if "resnext" in s:
        return "ResNeXt50_32x4d"
    return p.parent.name


def _infer_out_dim_from_flat_keys(d: Dict[str, Any]) -> Optional[int]:
    # Detect by presence of MAE_* keys
    if "MAE_w" in d and "MAE_z" in d and "MAE_x" in d and "MAE_y" in d:
        return 7
    if "MAE_alt" in d and "MAE_lon" in d and "MAE_lat" in d and "MAE_w" not in d:
        return 3
    if "MAE_lat" in d and "MAE_lon" in d and "MAE_alt" not in d:
        return 2
    return None


def _outputs_by_dim(dim: int) -> str:
    if dim == 2:
        return "lat, lon"
    if dim == 3:
        return "lat, lon, alt"
    if dim == 7:
        return "lat, lon, alt, x, y, z, w"
    return ""


def _family_from_name(name: str) -> str:
    if name.startswith("ResNet"):
        return "ResNet"
    if name.startswith("ResNeXt"):
        return "ResNeXt"
    if "YOLO" in name:
        return "YOLO"
    if "MobileNet" in name:
        return "MobileNet"
    return ""


def _pos_stats_from_series(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=float)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "min": float(np.min(err)),
        "max": float(np.max(err)),
    }


def _try_compute_pos_error_from_report(report_csv: Path) -> Optional[Dict[str, float]]:
    if not report_csv.exists():
        return None
    df = pd.read_csv(report_csv)

    # Two common layouts:
    # A) lat, lon, pred_lat, pred_lon
    if {"lat", "lon", "pred_lat", "pred_lon"}.issubset(df.columns):
        lat_true = df["lat"].to_numpy(dtype=float)
        lon_true = df["lon"].to_numpy(dtype=float)
        lat_pred = df["pred_lat"].to_numpy(dtype=float)
        lon_pred = df["pred_lon"].to_numpy(dtype=float)
        err = np.sqrt((lat_pred - lat_true) ** 2 + (lon_pred - lon_true) ** 2)
        return _pos_stats_from_series(err)

    # B) lat_true, lon_true, lat_pred, lon_pred
    if {"lat_true", "lon_true", "lat_pred", "lon_pred"}.issubset(df.columns):
        lat_true = df["lat_true"].to_numpy(dtype=float)
        lon_true = df["lon_true"].to_numpy(dtype=float)
        lat_pred = df["lat_pred"].to_numpy(dtype=float)
        lon_pred = df["lon_pred"].to_numpy(dtype=float)
        err = np.sqrt((lat_pred - lat_true) ** 2 + (lon_pred - lon_true) ** 2)
        return _pos_stats_from_series(err)

    return None


def _parse_metrics_file(metrics_path: Path) -> Dict[str, Any]:
    d = _read_json(metrics_path)
    row: Dict[str, Any] = {}
    row["source"] = str(metrics_path)

    # Base fields
    row["model_name"] = d.get("model_name") or _infer_model_name_from_path(metrics_path)
    row["task"] = d.get("task") or ""
    row["split"] = d.get("split") or "test"
    row["weights"] = d.get("weights") or ""
    row["device"] = d.get("device") or ""
    row["scaler_used"] = d.get("scaler_used") if "scaler_used" in d else ""

    # n_images
    n = d.get("n_samples")
    if n is None:
        n = d.get("n_images")
    if n is None:
        n = d.get("n_matched")
    row["n_images"] = int(n) if n is not None else ""

    # Determine out_dim
    out_dim = None
    if "per_param" in d and isinstance(d["per_param"], dict):
        # ResNeXt-like structure
        mae = d["per_param"].get("MAE", {})
        if isinstance(mae, dict):
            if set(mae.keys()) >= {"lat", "lon", "alt"}:
                out_dim = 3
    if out_dim is None:
        out_dim = _infer_out_dim_from_flat_keys(d)
    if out_dim is None:
        # fallback by presence of total/per_param
        out_dim = 3 if "per_param" in d else 2

    row["outputs"] = _outputs_by_dim(out_dim)
    row["family"] = _family_from_name(row["model_name"])

    # totals
    if "total" in d and isinstance(d["total"], dict):
        row["total_MAE"] = d["total"].get("MAE", "")
        row["total_MSE"] = d["total"].get("MSE", "")
        row["total_RMSE"] = d["total"].get("RMSE", "")
    else:
        # flat format: MAE_mean / RMSE_mean exist
        row["total_MAE"] = d.get("MAE_mean", "")
        row["total_RMSE"] = d.get("RMSE_mean", "")
        # total_MSE: average of all MSE_* we have
        mse_keys = [k for k in d.keys() if k.startswith("MSE_") and k not in ("MSE_mean",)]
        if mse_keys:
            row["total_MSE"] = float(np.mean([float(d[k]) for k in mse_keys]))
        else:
            row["total_MSE"] = ""

    # per-param MAE
    def _get_mae(name: str) -> Any:
        if "per_param" in d and isinstance(d["per_param"], dict):
            mae = d["per_param"].get("MAE", {})
            if isinstance(mae, dict) and name in mae:
                return mae[name]
        return d.get(f"MAE_{name}", "")

    row["MAE_lat"] = _get_mae("lat")
    row["MAE_lon"] = _get_mae("lon")
    row["MAE_alt"] = _get_mae("alt") if out_dim >= 3 else ""

    # pos_error
    if "pos_error_2d" in d and isinstance(d["pos_error_2d"], dict):
        pe = d["pos_error_2d"]
        row["pos_error_mean"] = pe.get("mean", "")
        row["pos_error_median"] = pe.get("median", "")
        row["pos_error_rmse"] = pe.get("rmse", "")
        row["pos_error_p90"] = pe.get("p90", "")
        row["pos_error_p95"] = pe.get("p95", "")
        row["pos_error_p99"] = pe.get("p99", "")
        row["pos_error_min"] = pe.get("min", "")
        row["pos_error_max"] = pe.get("max", "")
    elif "pos_error_2d_mean" in d:
        # flat format from ResNet34 auto
        row["pos_error_mean"] = d.get("pos_error_2d_mean", "")
        row["pos_error_median"] = d.get("pos_error_2d_median", "")
        row["pos_error_rmse"] = d.get("pos_error_2d_rmse", "")
        row["pos_error_p90"] = d.get("pos_error_2d_p90", "")
        row["pos_error_p95"] = d.get("pos_error_2d_p95", "")
        row["pos_error_p99"] = d.get("pos_error_2d_p99", "")
        row["pos_error_min"] = ""
        row["pos_error_max"] = ""
    else:
        # Try compute from report CSV (useful for ResNet18 2D auto)
        report_csv = metrics_path.parent / f"report_{row['split']}.csv"
        pe = _try_compute_pos_error_from_report(report_csv)
        if pe is not None:
            row["pos_error_mean"] = pe["mean"]
            row["pos_error_median"] = pe["median"]
            row["pos_error_rmse"] = pe["rmse"]
            row["pos_error_p90"] = pe["p90"]
            row["pos_error_p95"] = pe["p95"]
            row["pos_error_p99"] = pe["p99"]
            row["pos_error_min"] = pe["min"]
            row["pos_error_max"] = pe["max"]
        else:
            row["pos_error_mean"] = ""
            row["pos_error_median"] = ""
            row["pos_error_rmse"] = ""
            row["pos_error_p90"] = ""
            row["pos_error_p95"] = ""
            row["pos_error_p99"] = ""
            row["pos_error_min"] = ""
            row["pos_error_max"] = ""

    # if task empty — fill from out_dim
    if not row["task"]:
        if out_dim == 2:
            row["task"] = "2D regression (lat,lon)"
        elif out_dim == 3:
            row["task"] = "3D regression (lat,lon,alt)"
        else:
            row["task"] = "7D regression (lat,lon,alt,x,y,z,w)"

    return row


def _load_manual(manual_path: Path) -> pd.DataFrame:
    if not manual_path.exists():
        return pd.DataFrame()
    d = _read_json(manual_path)
    if isinstance(d, list):
        return pd.DataFrame(d)
    if isinstance(d, dict):
        for key in ["rows", "manual_results", "manual_entries", "data", "items", "entries"]:
            if key in d and isinstance(d[key], list):
                return pd.DataFrame(d[key])
        # fallback: single dict
        return pd.DataFrame([d])
    return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--manual", type=str, default="")
    ap.add_argument("--out_xlsx", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    metrics_files = sorted(results_root.rglob("metrics_test.json"))

    rows: List[Dict[str, Any]] = []
    for mp in metrics_files:
        try:
            rows.append(_parse_metrics_file(mp))
        except Exception as e:
            print(f"[warn] skip {mp}: {e}")

    auto_df = pd.DataFrame(rows)

    # Stable column order
    cols = [
        "model_name","task","split","weights","device","scaler_used","n_images",
        "total_MAE","total_MSE","total_RMSE","source",
        "MAE_lat","MAE_lon","MAE_alt",
        "pos_error_mean","pos_error_median","pos_error_rmse","pos_error_p90","pos_error_p95","pos_error_p99","pos_error_min","pos_error_max",
        "family","outputs",
    ]
    for c in cols:
        if c not in auto_df.columns:
            auto_df[c] = ""
    auto_df = auto_df[cols].sort_values(["model_name","task"]).reset_index(drop=True)

    manual_df = _load_manual(Path(args.manual)) if args.manual else pd.DataFrame()

    # Combined: just concat (align columns where possible)
    combined_df = pd.concat([auto_df, manual_df], ignore_index=True, sort=False)

    out_xlsx = Path(args.out_xlsx)
    out_csv = Path(args.out_csv)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        auto_df.to_excel(w, index=False, sheet_name="auto_results")
        if not manual_df.empty:
            manual_df.to_excel(w, index=False, sheet_name="manual_results")
        combined_df.to_excel(w, index=False, sheet_name="combined")

    combined_df.to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_xlsx}")
    print(f"✅ Saved: {out_csv}")
    print(f"[info] auto rows: {len(auto_df)}  manual rows: {len(manual_df)}  combined: {len(combined_df)}")


if __name__ == "__main__":
    main()
