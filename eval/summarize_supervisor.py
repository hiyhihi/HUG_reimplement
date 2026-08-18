"""Build the supervisor-requested CSV tables from experiment JSON artifacts."""

import argparse
import csv
import glob
import json
import re
import statistics
from pathlib import Path


CLEAN_RE = re.compile(r"(?P<model>.+)_(?P<category>dress|shirt|toptee)_seed(?P<seed>\d+)_(?P<mode>mean|probabilistic)\.json$")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    severity_rows = []
    output = Path(args.output_dir)
    clean_rows, modality_rows, robustness_rows, uncertainty_rows, query_rows = [], [], [], [], []

    for filename in glob.glob(str(Path(args.result_root) / "**" / "*.json"), recursive=True):
        try:
            with open(filename, encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        kind = value.get("kind")
        match = CLEAN_RE.search(Path(filename).name)
        if match and kind is None and "recall@10" in value:
            clean_rows.append({**match.groupdict(), "seed": int(match.group("seed")),
                               "recall@10": value["recall@10"], "recall@50": value["recall@50"],
                               "source": filename})
        if kind == "modality_reliance":
            for condition in value["conditions"]:
                metric = condition["metrics"]
                modality_rows.append({"model_id": value["model_id"], "category": value["category"],
                                      "seed": value["seed"], "condition": condition["condition"],
                                      "distance_mode": condition["distance_mode"],
                                      "recall@10": metric["recall@10"], "recall@50": metric["recall@50"]})
        if kind == "retrieval_evaluation":
            corrupt = value["corruption"]
            metric = value["metrics"]
            calibration = value.get("calibration") or {}
            base = {"model_id": value["model_id"], "category": value.get("category"),
                    "seed": corrupt["seed"], "distance_mode": value["distance_mode"],
                    "corruption_type": corrupt["name"], "modality": corrupt["modality"],
                    "severity": corrupt["severity"]}
            robustness_rows.append({**base, "recall@10": metric["recall@10"],
                                    "recall@50": metric["recall@50"]})
            uncertainty_rows.append({**base, "failure_auroc": calibration.get("failure_auroc"),
                                     "failure_auprc": calibration.get("failure_auprc"),
                                     "spearman_rank": calibration.get("spearman_rank"),
                                     "aurc": calibration.get("aurc")})
            for row in value.get("per_query", []):
                uncertainty = row.get("uncertainty", {})
                query_rows.append({**base, "query_id": row.get("query_id", row["query_index"]),
                                   "candidate_id": row.get("candidate_id"), "target_id": row.get("target_id"),
                                   "rank_target": row["rank"], "hit@10": row.get("hit@10", int(row["rank"] <= 10)),
                                   "hit@50": row.get("hit@50", int(row["rank"] <= 50)),
                                   "uncertainty_image": uncertainty.get("r"),
                                   "uncertainty_text": uncertainty.get("t"),
                                   "uncertainty_crossmodal": uncertainty.get("m"),
                                   "uncertainty_overall": uncertainty.get("q")})
        if kind == "robustness_summary":
            for curve in value.get("curves", []):
                severity_rows.append({"model_id": curve["model_id"], "distance_mode": curve["distance_mode"],
                                      "corruption_type": curve["corruption"], "modality": curve["modality"],
                                      "severity_monotonicity": curve.get("severity_monotonicity"),
                                      "severity_spearman": curve.get("severity_spearman"),
                                      "recall@10_auc": curve.get("recall@10_auc")})

    grouped, summary = {}, []
    for row in clean_rows:
        grouped.setdefault((row["model"], row["category"], row["mode"]), []).append(row)
    for (model, category, mode), rows in sorted(grouped.items()):
        r10, r50 = mean_std([row["recall@10"] for row in rows]), mean_std([row["recall@50"] for row in rows])
        summary.append({"model": model, "category": category, "distance_mode": mode, "n_seeds": len(rows),
                        "recall@10_mean": r10[0], "recall@10_std": r10[1],
                        "recall@50_mean": r50[0], "recall@50_std": r50[1]})
    by_run = {(row["model"], row["category"], row["seed"], row["mode"]): row for row in clean_rows}
    delta_groups, delta_rows = {}, []
    for row in clean_rows:
        if row["model"] not in {"point", "point_matched"} or row["mode"] != "mean":
            continue
        for hug_mode in ("mean", "probabilistic"):
            candidate = by_run.get(("hug_e2e", row["category"], row["seed"], hug_mode))
            if candidate:
                key = (row["model"], row["category"], hug_mode)
                delta_groups.setdefault(key, []).append((candidate["recall@10"] - row["recall@10"], candidate["recall@50"] - row["recall@50"]))
    for (baseline, category, mode), pairs in sorted(delta_groups.items()):
        d10, d50 = mean_std([pair[0] for pair in pairs]), mean_std([pair[1] for pair in pairs])
        delta_rows.append({"baseline_model": baseline, "category": category, "hug_distance_mode": mode, "n_paired_seeds": len(pairs),
                           "delta_recall@10_mean": d10[0], "delta_recall@10_std": d10[1],
                           "delta_recall@50_mean": d50[0], "delta_recall@50_std": d50[1]})


    write_csv(output / "01_clean_runs.csv", clean_rows,
              ["model", "category", "seed", "mode", "recall@10", "recall@50", "source"])
    write_csv(output / "01_clean_mean_std.csv", summary,
              ["model", "category", "distance_mode", "n_seeds", "recall@10_mean", "recall@10_std",
               "recall@50_mean", "recall@50_std"])
    write_csv(output / "01_clean_delta.csv", delta_rows,
              ["baseline_model", "category", "hug_distance_mode", "n_paired_seeds", "delta_recall@10_mean",
               "delta_recall@10_std", "delta_recall@50_mean", "delta_recall@50_std"])
    write_csv(output / "02_modality_reliance.csv", modality_rows,
              ["model_id", "category", "seed", "condition", "distance_mode", "recall@10", "recall@50"])
    write_csv(output / "03_robustness_severity.csv", robustness_rows,
              ["model_id", "category", "seed", "distance_mode", "corruption_type", "modality", "severity",
               "recall@10", "recall@50"])
    write_csv(output / "04_uncertainty_metrics.csv", uncertainty_rows,
              ["model_id", "category", "seed", "distance_mode", "corruption_type", "modality", "severity",
               "failure_auroc", "failure_auprc", "spearman_rank", "aurc"])
    write_csv(output / "04_uncertainty_severity.csv", severity_rows,
              ["model_id", "distance_mode", "corruption_type", "modality", "severity_monotonicity",
               "severity_spearman", "recall@10_auc"])
    write_csv(output / "per_query.csv", query_rows,
              ["query_id", "candidate_id", "target_id", "category", "seed", "model_id", "distance_mode",
               "corruption_type", "modality", "severity", "rank_target", "hit@10", "hit@50",
               "uncertainty_image", "uncertainty_text", "uncertainty_crossmodal", "uncertainty_overall"])
    print(f"Wrote summary tables to {output}")


if __name__ == "__main__":
    main()
