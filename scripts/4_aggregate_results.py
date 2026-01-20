from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_metrics_files(results_root: Path) -> List[Path]:
    return sorted(results_root.rglob("metrics_*.json"))


def flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model_name": m.get("model_name"),
        "task": m.get("task"),
        "split": m.get("split"),
        "weights": m.get("weights"),
        "device": m.get("device"),
        "scaler_used": m.get("scaler_used", None),
    }

    # sample counts
    for key in ["n_samples", "n_matched"]:
        if key in m:
            row["n_images"] = m[key]
            break
    row.setdefault("n_images", None)

    # totals
    total = m.get("total", {})
    row["total_MAE"] = total.get("MAE")
    row["total_MSE"] = total.get("MSE")
    row["total_RMSE"] = total.get("RMSE")

    # per-param MAE
    per = (m.get("per_param", {}) or {}).get("MAE", {})
    for k, v in per.items():
        row[f"MAE_{k}"] = v

    # pos error
    pe = m.get("pos_error_2d", {})
    for k in ["mean", "median", "rmse", "p90", "p95", "p99", "min", "max"]:
        if k in pe:
            row[f"pos_error_{k}"] = pe[k]

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate metrics_*.json into a single Excel/CSV comparison table.")
    ap.add_argument("--results_root", type=str, default="results", help="Root folder containing model subfolders.")
    ap.add_argument("--manual", type=str, default="manual_results.json", help="Manual results JSON (e.g., YOLO).")
    ap.add_argument("--out_xlsx", type=str, default="results/model_comparison.xlsx")
    ap.add_argument("--out_csv", type=str, default="results/model_comparison.csv")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    metrics_files = find_metrics_files(results_root)

    auto_rows: List[Dict[str, Any]] = []
    for p in metrics_files:
        try:
            m = load_json(p)
            row = flatten_metrics(m)
            row["source"] = str(p)
            auto_rows.append(row)
        except Exception as e:
            print(f"[warn] Could not parse {p}: {e}")

    df_auto = pd.DataFrame(auto_rows).sort_values(by=["model_name", "split"], ignore_index=True) if auto_rows else pd.DataFrame()

    manual_path = Path(args.manual)
    df_manual = pd.DataFrame()
    if manual_path.exists():
        try:
            mj = load_json(manual_path)
            entries = mj.get("manual_entries", [])
            df_manual = pd.DataFrame(entries)
        except Exception as e:
            print(f"[warn] Could not read manual results: {e}")

    df_combined = pd.concat([df_auto, df_manual], ignore_index=True, sort=False)

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        if not df_auto.empty:
            df_auto.to_excel(writer, sheet_name="auto_results", index=False)
        if not df_manual.empty:
            df_manual.to_excel(writer, sheet_name="manual_results", index=False)
        df_combined.to_excel(writer, sheet_name="combined", index=False)

    out_csv = Path(args.out_csv)
    df_combined.to_csv(out_csv, index=False)

    print(f"✅ Saved Excel: {out_xlsx.resolve()}")
    print(f"✅ Saved CSV:  {out_csv.resolve()}")
    print(f"Auto metrics files found: {len(metrics_files)}")


if __name__ == "__main__":
    main()
