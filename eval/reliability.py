"""Build Fashion-IQ-C failure labels and evaluate a reliability checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import collate_fn_gallery, get_transform
from eval import compute_pairwise_distance_matrix, extract_gallery_features, load_checkpoint
from robustness_legacy import build_gallery_dataset, build_query_dataset, get_tokenizer_and_processor
from models.reliability_model import load_reliability_checkpoint
from modules.fiqc import (
    FIQC_SCHEMA_VERSION, FIQC_SUITE, FIQCCorruptionGenerator, audit_text_corruptions, make_query_id,
)
from modules.reliability import reliability_metrics
from utils.artifact_paths import resolve_artifact_path


def _write_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _write_jsonl(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _write_csv(rows, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_texts(dataset):
    if not hasattr(dataset, "queries"):
        raise ValueError("Fashion-IQ-C requires a Fashion-IQ query dataset")
    return [", ".join(query["captions"]) for query in dataset.queries]


def audit_corruptions(args):
    raw = build_query_dataset(
        "fashion-iq", args.data_root, args.split, args.category, None, None, None,
    )
    rows = audit_text_corruptions(raw_texts(raw), args.seed, args.severities)
    synonym = [row for row in rows if row["corruption_type"] == "synonym_replacement"]
    minimum_rate = min((row["changed_rate"] for row in synonym), default=0.0)
    result = {
        "schema_version": FIQC_SCHEMA_VERSION,
        "kind": "fashion_iq_c_corruption_audit",
        "split": args.split,
        "category": args.category,
        "seed": args.seed,
        "minimum_synonym_changed_rate": minimum_rate,
        "required_synonym_changed_rate": args.min_synonym_changed_rate,
        "passed": minimum_rate >= args.min_synonym_changed_rate,
        "conditions": rows,
    }
    _write_json(result, args.output_file)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit("Fashion-IQ-C corruption audit failed")
    return result


class FIQCQueryDataset(Dataset):
    def __init__(self, raw, processor, tokenizer, transform, modality, name, severity,
                 seed, category, limit=None):
        self.raw, self.processor, self.tokenizer, self.transform = raw, processor, tokenizer, transform
        self.modality, self.name, self.severity, self.seed = modality, name, severity, seed
        self.category, self.generator = category, FIQCCorruptionGenerator(seed)
        self.texts = raw_texts(raw)
        self.limit = min(limit or len(raw), len(raw))

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        sample, clean_text = self.raw[index], self.texts[index]
        image, corrupted_text = sample["ref_image"], clean_text
        if self.modality == "image":
            image, metadata = self.generator.image(image, self.name, self.severity, index)
        elif self.modality == "text":
            corrupted_text, metadata = self.generator.text(clean_text, self.name, self.severity, index)
        else:
            metadata = {"type": "identity", "changed": False}
        processed = self.processor(corrupted_text)
        tokens = self.tokenizer(
            processed, return_tensors="pt", padding="max_length", truncation=True, max_length=77,
        )
        candidate_id, target_id = sample["candidate_id"], sample["target_id"]
        return {
            "ref_image": self.transform(image),
            "text_input_ids": tokens["input_ids"].squeeze(0),
            "text_attention_mask": tokens["attention_mask"].squeeze(0),
            "target_id": target_id, "candidate_id": candidate_id,
            "query_index": index,
            "query_id": make_query_id(self.category, candidate_id, target_id, index),
            "clean_text": clean_text, "corrupted_text": corrupted_text,
            "corruption_metadata": metadata,
        }


def collate_query(batch):
    return {
        "ref_images": torch.stack([item["ref_image"] for item in batch]),
        "text_input_ids": torch.stack([item["text_input_ids"] for item in batch]),
        "text_attention_mask": torch.stack([item["text_attention_mask"] for item in batch]),
        "target_ids": [item["target_id"] for item in batch],
        "candidate_ids": [item["candidate_id"] for item in batch],
        "query_indices": [item["query_index"] for item in batch],
        "query_ids": [item["query_id"] for item in batch],
        "clean_texts": [item["clean_text"] for item in batch],
        "corrupted_texts": [item["corrupted_text"] for item in batch],
        "corruption_metadata": [item["corruption_metadata"] for item in batch],
    }


@torch.no_grad()
def _extract_point_queries(model, loader, device):
    means, metadata = [], []
    for batch in loader:
        mu_q, _ = model.encode_query(
            batch["ref_images"].to(device), batch["text_input_ids"].to(device),
            batch["text_attention_mask"].to(device), compute_uncertainty=False,
        )
        means.append(mu_q.cpu())
        for position in range(len(batch["query_ids"])):
            metadata.append({key: batch[key][position] for key in (
                "target_ids", "candidate_ids", "query_indices", "query_ids", "clean_texts",
                "corrupted_texts", "corruption_metadata",
            )})
    return torch.cat(means), metadata


def _rank_rows(query_mu, query_meta, gallery, gallery_map, device, condition, args):
    distances = compute_pairwise_distance_matrix(
        query_mu, None, gallery[0], None, device, include_uncertainty=False,
    )
    for index, item in enumerate(query_meta):
        candidate_id = item["candidate_ids"]
        if candidate_id in gallery_map:
            distances[index, gallery_map[candidate_id]] = float("inf")
    rankings = torch.argsort(distances, dim=1)
    rows = []
    for index, item in enumerate(query_meta):
        target_id = item["target_ids"]
        rank = int((rankings[index] == gallery_map[target_id]).nonzero(as_tuple=True)[0].item()) + 1
        rows.append({
            "schema_version": FIQC_SCHEMA_VERSION, "dataset": "fashion-iq-c",
            "source_dataset": "fashion-iq", "split": args.split, "category": args.category,
            "query_id": item["query_ids"], "query_index": item["query_indices"],
            "candidate_id": item["candidate_ids"], "target_id": target_id, "target_image": target_id,
            "clean_text": item["clean_texts"], "corrupted_text": item["corrupted_texts"],
            "modality": condition[0], "corruption_type": condition[1],
            "severity": condition[2], "corruption_seed": args.seed,
            "corruption_metadata": item["corruption_metadata"],
            "label_checkpoint": str(Path(args.checkpoint).resolve()),
            "retrieval_rank": rank, "hit@10": int(rank <= 10), "hit@50": int(rank <= 50),
            "failure_k": args.failure_k, "failure_label": int(rank > args.failure_k),
        })
    return rows


def _condition_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["modality"], row["corruption_type"], row["severity"])].append(row)
    output = []
    for (modality, name, severity), values in sorted(groups.items()):
        count = len(values)
        output.append({
            "modality": modality, "corruption_type": name, "severity": severity, "count": count,
            "recall@10": 100 * sum(row["hit@10"] for row in values) / count,
            "recall@50": 100 * sum(row["hit@50"] for row in values) / count,
            "failure_rate": sum(row["failure_label"] for row in values) / count,
            "changed_rate": sum(bool(row["corruption_metadata"].get("changed")) for row in values) / count,
        })
    return output


def build_labels(args):
    output = Path(args.output_dir)
    manifest_path = output / f"fiqc_{args.category}_{args.split}_seed{args.seed}.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    processor, tokenizer = get_tokenizer_and_processor()
    transform = get_transform(224, False)
    raw = build_query_dataset("fashion-iq", args.data_root, args.split, args.category, None, None, None)
    gallery_data = build_gallery_dataset("fashion-iq", args.data_root, args.split, args.category, transform)
    gallery_loader = DataLoader(
        gallery_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn_gallery,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    gallery_mu, _, gallery_ids = extract_gallery_features(
        model, gallery_loader, device, compute_uncertainty=False,
    )
    conditions = [("none", "identity", 0)] + [
        (modality, name, severity) for modality, name in FIQC_SUITE for severity in args.severities
    ]
    rows = []
    for modality, name, severity in conditions:
        dataset = FIQCQueryDataset(
            raw, processor, tokenizer, transform, modality, name, severity,
            args.seed, args.category, args.max_queries,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_query,
        )
        query_mu, query_meta = _extract_point_queries(model, loader, device)
        rows.extend(_rank_rows(
            query_mu, query_meta, (gallery_mu, gallery_ids), gallery_data.id_to_idx,
            device, (modality, name, severity), args,
        ))
    summaries = _condition_summary(rows)
    synonym = [row for row in summaries if row["corruption_type"] == "synonym_replacement"]
    if synonym and min(row["changed_rate"] for row in synonym) < args.min_synonym_changed_rate:
        rates = [row["changed_rate"] for row in synonym]
        raise ValueError(
            f"synonym changed_rate below {args.min_synonym_changed_rate:.2%}: {rates}"
        )
    _write_jsonl(rows, manifest_path)
    _write_csv(rows, manifest_path.with_suffix(".csv"))
    _write_json({
        "schema_version": FIQC_SCHEMA_VERSION, "kind": "fashion_iq_c_summary",
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "category": args.category, "seed": args.seed, "failure_k": args.failure_k,
        "conditions": summaries,
    }, output / f"fiqc_{args.category}_{args.split}_seed{args.seed}_summary.json")
    _write_csv(summaries, output / f"fiqc_{args.category}_{args.split}_seed{args.seed}_robustness.csv")
    print(f"Wrote {len(rows)} per-query rows to {manifest_path}")


class ManifestDataset(Dataset):
    def __init__(self, rows, data_root, processor, tokenizer, transform):
        self.rows, self.image_dir = rows, Path(data_root) / "images"
        self.processor, self.tokenizer, self.transform = processor, tokenizer, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        from PIL import Image
        image = Image.open(self.image_dir / f"{row['candidate_id']}.png").convert("RGB")
        if row["modality"] == "image" and row["severity"] > 0:
            image, _ = FIQCCorruptionGenerator(row["corruption_seed"]).image(
                image, row["corruption_type"], row["severity"], row["query_index"],
            )
        text = self.processor(row["corrupted_text"])
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            "ref_image": self.transform(image), "text_input_ids": tokens["input_ids"].squeeze(0),
            "text_attention_mask": tokens["attention_mask"].squeeze(0), "row": row,
        }


def collate_manifest(batch):
    return {
        "ref_images": torch.stack([item["ref_image"] for item in batch]),
        "text_input_ids": torch.stack([item["text_input_ids"] for item in batch]),
        "text_attention_mask": torch.stack([item["text_attention_mask"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }



def _subset_metrics(labels, probabilities, indices, ece_bins):
    return reliability_metrics(
        [labels[index] for index in indices],
        [probabilities[index] for index in indices],
        ece_bins,
    )
@torch.no_grad()
def evaluate(args):
    rows = load_manifest(args.manifest)
    train_rows = load_manifest(args.train_manifest)
    if not rows or not train_rows:
        raise ValueError("train and evaluation manifests must be non-empty")
    for name, values in (("train", train_rows), ("evaluation", rows)):
        schemas = {row.get("schema_version") for row in values}
        if schemas != {FIQC_SCHEMA_VERSION}:
            raise ValueError(
                f"{name} manifest schema {schemas} does not match v{FIQC_SCHEMA_VERSION}"
            )
    if args.max_queries:
        rows = rows[:args.max_queries]
    processor, tokenizer = get_tokenizer_and_processor(); transform = get_transform(224, False)
    loader = DataLoader(
        ManifestDataset(rows, args.data_root, processor, tokenizer, transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_manifest,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_reliability_checkpoint(args.checkpoint, device)
    probabilities, output_rows = [], []
    for batch in loader:
        _, logits = model.encode_query_with_reliability(
            batch["ref_images"].to(device), batch["text_input_ids"].to(device),
            batch["text_attention_mask"].to(device),
        )
        batch_probabilities = torch.sigmoid(logits).cpu().tolist()
        probabilities.extend(batch_probabilities)
        output_rows.extend([
            {**row, "failure_probability": probability}
            for row, probability in zip(batch["rows"], batch_probabilities)
        ])
    labels = [row["failure_label"] for row in rows]
    overall = reliability_metrics(labels, probabilities, args.ece_bins)
    grouped = {}
    for name in sorted({row["corruption_type"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["corruption_type"] == name]
        grouped[name] = reliability_metrics(
            [labels[index] for index in indices], [probabilities[index] for index in indices], args.ece_bins,
        )
    by_condition = {}
    for name, severity in sorted({(row["corruption_type"], row["severity"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["corruption_type"] == name and row["severity"] == severity]
        by_condition[f"{name}/s{severity}"] = reliability_metrics(
            [labels[index] for index in indices], [probabilities[index] for index in indices], args.ece_bins,
        )
    by_modality = {}
    for modality in sorted({row["modality"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["modality"] == modality]
        by_modality[modality] = reliability_metrics(
            [labels[index] for index in indices], [probabilities[index] for index in indices], args.ece_bins,
        )
    subset_indices = {
        "clean_only": [i for i, row in enumerate(rows) if row["corruption_type"] == "identity"],
        "nuisance_only": [i for i, row in enumerate(rows) if row["corruption_type"] != "semantic_contradiction"],
        "corrupted_nuisance_only": [i for i, row in enumerate(rows) if row["corruption_type"] not in {"identity", "semantic_contradiction"}],
        "semantic_contradiction": [i for i, row in enumerate(rows) if row["corruption_type"] == "semantic_contradiction"],
    }
    subsets = {
        name: _subset_metrics(labels, probabilities, indices, args.ece_bins)
        for name, indices in subset_indices.items()
    }
    train_condition_labels = defaultdict(list)
    for row in train_rows:
        key = (row["corruption_type"], row["severity"])
        train_condition_labels[key].append(row["failure_label"])
    condition_prevalence = {
        key: sum(values) / len(values)
        for key, values in train_condition_labels.items()
    }
    missing_conditions = {
        (row["corruption_type"], row["severity"]) for row in rows
    } - set(condition_prevalence)
    if missing_conditions:
        raise ValueError(f"train manifest lacks conditions: {sorted(missing_conditions)}")
    condition_scores = [
        condition_prevalence[(row["corruption_type"], row["severity"])]
        for row in rows
    ]
    condition_only = reliability_metrics(labels, condition_scores, args.ece_bins)
    clean_scores = {
        row["query_id"]: probability
        for row, probability in zip(rows, probabilities)
        if row["corruption_type"] == "identity"
    }
    if set(row["query_id"] for row in rows) - set(clean_scores):
        raise ValueError("every evaluation query must have an identity row")
    clean_repeated = reliability_metrics(
        labels, [clean_scores[row["query_id"]] for row in rows], args.ece_bins,
    )
    risk_at_half = min(overall["risk_coverage"], key=lambda point: abs(point["coverage"] - 0.5))["risk"]
    relative_risk_reduction = (overall["failure_prevalence"] - risk_at_half) / overall["failure_prevalence"] if overall["failure_prevalence"] else 0.0
    condition_values = [metric for metric in by_condition.values() if metric["auroc"] is not None]
    within_condition = {
        "macro_auroc": sum(metric["auroc"] for metric in condition_values) / len(condition_values),
        "weighted_auroc": sum(metric["auroc"] * metric["count"] for metric in condition_values) / sum(metric["count"] for metric in condition_values),
        "minimum_auroc": min(metric["auroc"] for metric in condition_values),
    }
    synonym_rows = [row for row in rows if row["corruption_type"] == "synonym_replacement"]
    synonym_rate = sum(bool(row["corruption_metadata"].get("changed")) for row in synonym_rows) / len(synonym_rows)
    label_checkpoints = {str(resolve_artifact_path(value)) for value in
                         {row["label_checkpoint"] for row in rows + train_rows}}
    backbone_match = label_checkpoints == {str(Path(model.point_checkpoint).resolve())}
    condition_criteria = {
        "clean_auroc_ge_0_60": subsets["clean_only"]["auroc"] >= 0.60,
        "image_auroc_ge_0_60": by_modality["image"]["auroc"] >= 0.60,
        "text_auroc_ge_0_60": by_modality["text"]["auroc"] >= 0.60,
        "nuisance_auroc_gt_0_70": subsets["nuisance_only"]["auroc"] > 0.70,
        "within_condition_macro_auroc_ge_0_65": within_condition["macro_auroc"] >= 0.65,
        "beats_condition_only_by_ge_0_05": overall["auroc"] - condition_only["auroc"] >= 0.05,
    }
    criteria = {
        "auroc_gt_0_70": overall["auroc"] is not None and overall["auroc"] > 0.70,
        "auprc_margin_ge_0_10": overall["auprc"] is not None and overall["auprc"] - overall["failure_prevalence"] >= 0.10,
        "ece_le_0_10": overall["ece"] <= 0.10,
        "risk_reduction_at_50pct_ge_0_20": relative_risk_reduction >= 0.20,
    }
    gate_passed = (
        all(criteria.values())
        and all(condition_criteria.values())
        and synonym_rate >= args.min_synonym_changed_rate
        and backbone_match
    )
    result = {
        "schema_version": FIQC_SCHEMA_VERSION, "kind": "reliability_evaluation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "train_manifest": str(Path(args.train_manifest).resolve()), "failure_k": rows[0]["failure_k"],
        "metrics": overall,
        "subsets": subsets,
        "by_modality": by_modality,
        "by_corruption": grouped,
        "by_condition": by_condition,
        "shortcut_baselines": {"condition_only": condition_only, "clean_score_repeated": clean_repeated},
        "within_condition": within_condition,
        "data_audit": {"synonym_changed_rate": synonym_rate, "backbone_match": backbone_match},
        "adaptive_fusion_gate": {
            "numeric_criteria": criteria,
            "condition_criteria": condition_criteria,
            "risk_at_50pct_coverage": risk_at_half,
            "relative_risk_reduction_at_50pct": relative_risk_reduction,
            "automatic_numeric_pass": all(criteria.values()),
            "condition_audit_pass": all(condition_criteria.values()),
            "corruption_audit_pass": synonym_rate >= args.min_synonym_changed_rate,
            "manual_review_recommended": True,
            "passed": gate_passed,
        },
    }
    _write_json(result, args.output_file)
    _write_csv(output_rows, Path(args.output_file).with_suffix(".per_query.csv"))
    _write_csv(
        overall["risk_coverage"], Path(args.output_file).with_suffix(".risk_coverage.csv"),
    )
    print(json.dumps(result["metrics"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-labels")
    build.add_argument("--data_root", required=True); build.add_argument("--checkpoint", required=True)
    build.add_argument("--split", choices=["train", "val"], required=True)
    build.add_argument("--category", choices=["dress", "shirt", "toptee"], default="dress")
    build.add_argument("--seed", type=int, default=42); build.add_argument("--failure_k", type=int, default=10)
    build.add_argument("--severities", nargs="+", type=int, choices=range(1, 5), default=[1, 2, 3, 4])
    build.add_argument("--batch_size", type=int, default=32); build.add_argument("--num_workers", type=int, default=4)
    build.add_argument("--max_queries", type=int); build.add_argument("--output_dir", required=True)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--min_synonym_changed_rate", type=float, default=0.90)
    audit = subparsers.add_parser("audit-corruptions")
    audit.add_argument("--data_root", required=True); audit.add_argument("--output_file", required=True)
    audit.add_argument("--split", choices=["train", "val"], required=True)
    audit.add_argument("--category", choices=["dress", "shirt", "toptee"], default="dress")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--severities", nargs="+", type=int, choices=range(1, 5), default=[1, 2, 3, 4])
    audit.add_argument("--min_synonym_changed_rate", type=float, default=0.90)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data_root", required=True); evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--train_manifest", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True); evaluate_parser.add_argument("--output_file", required=True)
    evaluate_parser.add_argument("--min_synonym_changed_rate", type=float, default=0.90)
    evaluate_parser.add_argument("--batch_size", type=int, default=32); evaluate_parser.add_argument("--num_workers", type=int, default=4)
    evaluate_parser.add_argument("--ece_bins", type=int, default=10); evaluate_parser.add_argument("--max_queries", type=int)
    args = parser.parse_args()
    {
        "audit-corruptions": audit_corruptions,
        "build-labels": build_labels,
        "evaluate": evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
