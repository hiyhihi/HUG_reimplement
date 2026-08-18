"""Deterministic, modality-isolated robustness/calibration evaluation for HUG."""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import collate_fn_gallery, collate_fn_query, get_transform
from eval import compute_pairwise_distance_matrix, extract_gallery_features, load_checkpoint
from robustness_legacy import build_gallery_dataset, build_query_dataset, get_tokenizer_and_processor


SUITE = (
    ("A", "image", "gaussian_blur"),
    ("A", "text", "typo"),
    ("B", "image", "occlusion"),
    ("B", "text", "token_dropout"),
)


def save_json(value, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


class Generator:
    """Nested severity with a stable RNG per query (independent of DataLoader workers)."""

    def __init__(self, seed):
        self.seed = seed

    def rng(self, index, name):
        digest = hashlib.sha256(f"{self.seed}|{index}|{name}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def image(self, image, group, name, severity, index):
        if (group, name) not in {("A", "gaussian_blur"), ("B", "occlusion")}:
            raise ValueError(f"invalid image corruption {group}/{name}")
        if severity == 0:
            return image.copy(), {"type": "identity", "changed": False}
        if name == "gaussian_blur":
            radius = (0, .5, 1, 2, 4)[severity]
            return image.filter(ImageFilter.GaussianBlur(radius)), {"type": name, "radius": radius, "changed": True}
        ratio = (0, .1, .2, .35, .5)[severity]
        rng, width, height = self.rng(index, name), *image.size
        side = math.sqrt(ratio)
        block_w, block_h = max(1, round(width * side)), max(1, round(height * side))
        left, top = rng.randrange(width - block_w + 1), rng.randrange(height - block_h + 1)
        result = image.copy()
        result.paste((127, 127, 127), (left, top, left + block_w, top + block_h))
        return result, {"type": name, "area_ratio": ratio, "changed": True}

    def text(self, text, group, name, severity, index, all_texts):
        if (group, name) not in {("A", "typo"), ("B", "token_dropout"), ("C", "caption_swap")}:
            raise ValueError(f"invalid text corruption {group}/{name}")
        if severity == 0:
            return text, {"type": "identity", "changed": False}
        rng = self.rng(index, name)
        if name == "caption_swap":
            offset = rng.randrange(1, len(all_texts))
            result = all_texts[(index + offset) % len(all_texts)]
            return result, {"type": name, "changed": result != text, "source_offset": offset}
        if name == "token_dropout":
            tokens, rate = text.split(), (0, .1, .2, .35, .5)[severity]
            priorities = sorted((rng.random(), i) for i in range(len(tokens)))
            count = min(max(0, len(tokens) - 1), max(1, round(rate * len(tokens))))
            removed = {i for _, i in priorities[:count]}
            result = " ".join(token for i, token in enumerate(tokens) if i not in removed)
            return result, {"type": name, "rate": rate, "changed": result != text}
        chars, rate = list(text), (0, .05, .1, .2, .3)[severity]
        candidates = [i for i in range(len(chars) - 1) if chars[i].isalpha() and chars[i + 1].isalpha()]
        count = min(len(candidates), max(1, round(rate * len(candidates))))
        for i in sorted(rng.sample(candidates, count), reverse=True):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        result = "".join(chars)
        return result, {"type": name, "rate": rate, "changed": result != text}


def raw_texts(dataset):
    """Read annotations without opening every reference image."""
    if hasattr(dataset, "queries"):
        values = []
        for query in dataset.queries:
            values.append(", ".join(query["captions"]) if "captions" in query else query["caption"])
        return values
    return [dataset[index]["text_input_ids"] for index in range(len(dataset))]


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, raw, processor, tokenizer, transform, group, modality, name, severity, seed, limit=None):
        self.raw, self.processor, self.tokenizer, self.transform = raw, processor, tokenizer, transform
        self.group, self.modality, self.name, self.severity = group, modality, name, severity
        self.generator, self.limit = Generator(seed), min(limit or len(raw), len(raw))
        self.texts = raw_texts(raw)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        sample, text = self.raw[index], self.texts[index]
        image = sample["ref_image"]
        if self.modality == "image":
            image, _ = self.generator.image(image, self.group, self.name, self.severity, index)
        else:
            text, _ = self.generator.text(text, self.group, self.name, self.severity, index, self.texts)
        text = self.processor(text)
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {"ref_image": self.transform(image), "text_input_ids": tokens["input_ids"].squeeze(0),
                "text_attention_mask": tokens["attention_mask"].squeeze(0), "target_id": sample["target_id"],
                "candidate_id": sample["candidate_id"]}


def average_ranks(values):
    order, output, start = sorted(range(len(values)), key=lambda i: values[i]), [0.] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for position in range(start, end):
            output[order[position]] = (start + 1 + end) / 2
        start = end
    return output


def auroc(labels, scores):
    positive, negative = sum(labels), len(labels) - sum(labels)
    if not positive or not negative:
        return None
    ranked = average_ranks(scores)
    return (sum(rank for rank, label in zip(ranked, labels) if label) - positive * (positive + 1) / 2) / (positive * negative)


def auprc(labels, scores):
    if not sum(labels):
        return None
    found, total = 0, 0.
    for position, (_, label) in enumerate(sorted(zip(scores, labels), reverse=True), 1):
        if label:
            found += 1
            total += found / position
    return total / sum(labels)


def spearman(left, right):
    left, right = average_ranks(left), average_ranks(right)
    mean_l, mean_r = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - mean_l) ** 2 for a in left) * sum((b - mean_r) ** 2 for b in right))
    return numerator / denominator if denominator else None


@torch.no_grad()
def extract_queries(model, loader, device, uncertainty):
    means, variances, targets, candidates = [], [], [], []
    scores = {key: [] for key in ("r", "t", "m", "q", "w_r", "w_t", "w_m")} if uncertainty else None
    for batch in loader:
        images = batch["ref_images"].to(device)
        ids, masks = batch["text_input_ids"].to(device), batch["text_attention_mask"].to(device)
        if uncertainty:
            parts = model.extract_query_components(images, ids, masks)
            means.append(parts["mu_m"].cpu()); variances.append(parts["sigma_q"].cpu())
            for short, key in (("r", "sigma_r"), ("t", "sigma_t"), ("m", "sigma_m"), ("q", "sigma_q")):
                value = parts[key] if model.uncertainty_is_variance else parts[key].square()
                scores[short].extend(value.mean((-2, -1)).cpu().tolist())
            for key, value in zip(("w_r", "w_t", "w_m"), model.dynamic_weighting.get_weights(parts["sigma_r"], parts["sigma_t"], parts["sigma_m"])):
                scores[key].extend(value.mean((-2, -1)).cpu().tolist())
        else:
            mean, _ = model.encode_query(images, ids, masks, compute_uncertainty=False)
            means.append(mean.cpu())
        targets.extend(batch["target_ids"]); candidates.extend(batch["candidate_ids"])
    return {"mu": torch.cat(means), "sigma": torch.cat(variances) if variances else None,
            "targets": targets, "candidates": candidates, "scores": scores}


def evaluate_bundle(model, query, gallery, gallery_map, device, mode):
    probabilistic = mode == "probabilistic"
    if probabilistic and query["sigma"] is None:
        raise ValueError("point checkpoints must use mean mode with --skip_uncertainty")
    distances = compute_pairwise_distance_matrix(query["mu"], query["sigma"], gallery[0], gallery[1], device,
        uncertainty_is_variance=model.uncertainty_is_variance, include_uncertainty=probabilistic)
    for index, candidate in enumerate(query["candidates"]):
        if candidate in gallery_map:
            distances[index, gallery_map[candidate]] = float("inf")
    order, target_ranks = torch.argsort(distances, 1), []
    for index, target in enumerate(query["targets"]):
        target_ranks.append(int((order[index] == gallery_map[target]).nonzero(as_tuple=True)[0].item()) + 1)
    count = len(target_ranks)
    metrics = {f"recall@{k}": 100 * sum(rank <= k for rank in target_ranks) / count for k in (1, 5, 10, 50)}
    metrics.update(mrr=sum(1 / rank for rank in target_ranks) / count, median_rank=sorted(target_ranks)[count // 2])
    calibration = None
    if query["scores"]:
        uncertainty, failures = query["scores"]["q"], [int(rank > 10) for rank in target_ranks]
        order_by_confidence = sorted(range(count), key=lambda index: uncertainty[index])
        cumulative, risks, risk_curve = 0, [], []
        for coverage, index in enumerate(order_by_confidence, 1):
            cumulative += failures[index]; risks.append(cumulative / coverage)
            if coverage == count or coverage % max(1, count // 10) == 0:
                risk_curve.append({"coverage": coverage / count, "risk": risks[-1]})
        bins = []
        for bin_index in range(10):
            start, end = round(bin_index * count / 10), round((bin_index + 1) * count / 10)
            selected = order_by_confidence[start:end]
            if selected:
                bins.append({"bin": bin_index, "count": len(selected),
                             "mean_uncertainty": sum(uncertainty[i] for i in selected) / len(selected),
                             "failure@10": sum(failures[i] for i in selected) / len(selected)})
        calibration = {"spearman_rank": spearman(uncertainty, [math.log1p(rank) for rank in target_ranks]),
                       "failure_auroc": auroc(failures, uncertainty), "failure_auprc": auprc(failures, uncertainty),
                       "aurc": sum(risks) / len(risks), "risk_coverage": risk_curve,
                       "bins": bins, "failure_k": 10}
    rows = []
    for index, rank in enumerate(target_ranks):
        row = {"query_index": index,
               "query_id": f"{query['candidates'][index]}->{query['targets'][index]}",
               "candidate_id": query["candidates"][index], "target_id": query["targets"][index],
               "rank": rank, "hit@10": int(rank <= 10), "hit@50": int(rank <= 50),
               "failure@10": int(rank > 10)}
        if query["scores"]:
            row["uncertainty"] = {key: query["scores"][key][index] for key in ("r", "t", "m", "q")}
            row["weights"] = {key: query["scores"][key][index] for key in ("w_r", "w_t", "w_m")}
        rows.append(row)
    return {"metrics": metrics, "calibration": calibration, "per_query": rows}


def prepare(args):
    processor, tokenizer = get_tokenizer_and_processor(); transform = get_transform(224, False)
    raw = build_query_dataset(args.dataset, args.data_root, args.split, args.category, None, None, None)
    gallery_data = build_gallery_dataset(args.dataset, args.data_root, args.split, args.category, transform)
    loader = DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_gallery)
    device, model = torch.device("cuda" if torch.cuda.is_available() else "cpu"), None
    model = load_checkpoint(args.checkpoint, device)
    gallery = extract_gallery_features(model, loader, device, compute_uncertainty="probabilistic" in args.distance_modes)
    return processor, tokenizer, transform, raw, gallery_data, device, model, gallery


def make_query(args, state, group, modality, name, severity, force_uncertainty=False):
    processor, tokenizer, transform, raw, _, device, model, _ = state
    data = QueryDataset(raw, processor, tokenizer, transform, group, modality, name, severity, args.seed, args.max_queries)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_query)
    return extract_queries(model, loader, device, force_uncertainty or not args.skip_uncertainty)


def result(args, condition, evaluated):
    return {"schema_version": 2, "kind": "retrieval_evaluation", "model_id": args.model_id,
            "dataset": args.dataset, "split": args.split, "category": args.category if args.dataset == "fashion-iq" else None,
            "checkpoint": args.checkpoint, "distance_mode": condition[-1],
            "corruption": dict(zip(("group", "modality", "name", "severity", "seed"), condition[:-1])), **evaluated}


def sweep(args):
    state, output = prepare(args), Path(args.output_dir)
    conditions = [("A", "image", "gaussian_blur", 0)] + [(*item, severity) for item in SUITE for severity in args.severities]
    for group, modality, name, severity in conditions:
        query = make_query(args, state, group, modality, name, severity)
        _, _, _, _, gallery_data, device, model, gallery = state
        for mode in args.distance_modes:
            evaluated = evaluate_bundle(model, query, gallery, gallery_data.id_to_idx, device, mode)
            clean = severity == 0
            condition = ("clean" if clean else group, "none" if clean else modality, "identity" if clean else name, severity, args.seed, mode)
            value = result(args, condition, evaluated)
            label = "clean_s0" if clean else f"{group}_{modality}_{name}_s{severity}"
            save_json(value, output / f"{args.model_id}_{args.category}_{mode}_{label}_seed{args.seed}.json")
    if args.include_mismatch and not args.skip_uncertainty:
        matched = make_query(args, state, "A", "text", "typo", 0, True)["scores"]["m"]
        mismatched = make_query(args, state, "C", "text", "caption_swap", 1, True)["scores"]["m"]
        labels, scores = [0] * len(matched) + [1] * len(mismatched), matched + mismatched
        save_json({"schema_version": 2, "kind": "mismatch_detection", "model_id": args.model_id,
                   "metrics": {"mismatch_auroc": auroc(labels, scores), "mismatch_auprc": auprc(labels, scores)},
                   "per_query": [{"query_index": i, "matched": matched[i], "mismatched": mismatched[i]} for i in range(len(matched))]},
                  output / f"{args.model_id}_{args.category}_mismatch_seed{args.seed}.json")


def generate(args):
    raw = build_query_dataset(args.dataset, args.data_root, args.split, args.category, None, None, None)
    texts, generator = raw_texts(raw), Generator(args.seed)
    samples = []
    for index in range(min(args.max_samples, len(raw))):
        sample, text = raw[index], texts[index]
        text_meta, image_meta, corrupted = {"type": "identity", "changed": False}, {"type": "identity", "changed": False}, text
        if args.modality == "text":
            corrupted, text_meta = generator.text(text, args.group, args.corruption, args.severity, index, texts)
        else:
            _, image_meta = generator.image(sample["ref_image"], args.group, args.corruption, args.severity, index)
        samples.append({"query_index": index, "original_text": text, "corrupted_text": corrupted, "text": text_meta, "image": image_meta})
    config = {"group": args.group, "modality": args.modality, "name": args.corruption,
              "severity": args.severity, "seed": args.seed}
    save_json({"schema_version": 2, "kind": "corruption_manifest", "corruption": config, "samples": samples}, args.output_file)


def aggregate(args):
    values = []
    for filename in glob.glob(args.input_glob, recursive=True):
        with open(filename, encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("kind") == "retrieval_evaluation": values.append(value)
    clean, grouped = {}, {}
    for value in values:
        base = (value["model_id"], value["dataset"], value.get("category"), value["distance_mode"], value["corruption"]["seed"])
        if value["corruption"]["severity"] == 0: clean[base] = value
        else: grouped.setdefault(base + (value["corruption"]["group"], value["corruption"]["modality"], value["corruption"]["name"]), []).append(value)
    curves = []
    for key, items in grouped.items():
        ordered = [clean[key[:5]]] + sorted(items, key=lambda x: x["corruption"]["severity"])
        severity_values = [x["corruption"]["severity"] for x in ordered]
        points = [{"severity": x["corruption"]["severity"], "recall@10": x["metrics"]["recall@10"], "recall@50": x["metrics"]["recall@50"],
                   "failure_auroc": (x["calibration"] or {}).get("failure_auroc"), "aurc": (x["calibration"] or {}).get("aurc")} for x in ordered]
        component, checks, severity_correlations = ("r" if key[6] == "image" else "t"), [], []
        maps = [{row["query_index"]: row for row in x["per_query"]} for x in ordered]
        for index in set.intersection(*(set(rows) for rows in maps)):
            sequence = [rows[index].get("uncertainty", {}).get(component) for rows in maps]
            if all(x is not None for x in sequence):
                checks.append(all(b >= a for a, b in zip(sequence, sequence[1:])))
                correlation = spearman(severity_values, sequence)
                if correlation is not None: severity_correlations.append(correlation)
        width = points[-1]["severity"]
        auc10 = sum((b["severity"] - a["severity"]) * (a["recall@10"] + b["recall@10"]) / 2 for a, b in zip(points, points[1:])) / width
        curves.append({"model_id": key[0], "distance_mode": key[3], "group": key[5], "modality": key[6], "corruption": key[7],
                       "points": points, "recall@10_auc": auc10, "worst_recall@10": min(x["recall@10"] for x in points),
                       "worst_retention@10": min(x["recall@10"] / points[0]["recall@10"] for x in points[1:]),
                       "severity_monotonicity": sum(checks) / len(checks) if checks else None,
                       "severity_spearman": sum(severity_correlations) / len(severity_correlations) if severity_correlations else None})
    save_json({"schema_version": 2, "kind": "robustness_summary", "curves": curves}, args.output_file)


def compare(args):
    with open(args.baseline_file) as handle: left = {x["query_index"]: x for x in json.load(handle)["per_query"]}
    with open(args.candidate_file) as handle: right = {x["query_index"]: x for x in json.load(handle)["per_query"]}
    indices, rng, output = sorted(set(left) & set(right)), random.Random(args.seed), {}
    for k in (10, 50):
        estimates = []
        for _ in range(args.iterations):
            sample = [indices[rng.randrange(len(indices))] for _ in indices]
            estimates.append(100 * sum((right[i]["rank"] <= k) - (left[i]["rank"] <= k) for i in sample) / len(sample))
        estimates.sort(); observed = 100 * sum((right[i]["rank"] <= k) - (left[i]["rank"] <= k) for i in indices) / len(indices)
        low, high = round(.025 * (len(estimates) - 1)), round(.975 * (len(estimates) - 1))
        output[f"delta_recall@{k}"] = {"estimate": observed, "ci95": [estimates[low], estimates[high]]}
    save_json({"kind": "paired_bootstrap", "metrics": output}, args.output_file)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    def data(p):
        p.add_argument("--dataset", choices=["fashion-iq", "cirr"], default="fashion-iq"); p.add_argument("--data_root", required=True)
        p.add_argument("--split", default="val"); p.add_argument("--category", default="dress"); p.add_argument("--seed", type=int, default=42)
    g = sub.add_parser("generate"); data(g); g.add_argument("--group", required=True); g.add_argument("--modality", required=True)
    g.add_argument("--corruption", required=True); g.add_argument("--severity", type=int, choices=range(5), required=True)
    g.add_argument("--max_samples", type=int, default=50); g.add_argument("--output_file", required=True)
    s = sub.add_parser("sweep"); data(s); s.add_argument("--checkpoint", required=True); s.add_argument("--model_id", required=True)
    s.add_argument("--batch_size", type=int, default=32); s.add_argument("--num_workers", type=int, default=4); s.add_argument("--max_queries", type=int)
    s.add_argument("--skip_uncertainty", action="store_true"); s.add_argument("--distance_modes", nargs="+", default=["probabilistic"])
    s.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4]); s.add_argument("--include_mismatch", action="store_true"); s.add_argument("--output_dir", required=True)
    a = sub.add_parser("aggregate"); a.add_argument("--input_glob", required=True); a.add_argument("--output_file", required=True)
    c = sub.add_parser("compare"); c.add_argument("--baseline_file", required=True); c.add_argument("--candidate_file", required=True)
    c.add_argument("--iterations", type=int, default=1000); c.add_argument("--seed", type=int, default=42); c.add_argument("--output_file", required=True)
    args = parser.parse_args(); {"generate": generate, "sweep": sweep, "aggregate": aggregate, "compare": compare}[args.command](args)


if __name__ == "__main__": main()
