"""Evaluate image/text reliance under five controlled query conditions."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import collate_fn_query, get_transform
from robustness import evaluate_bundle, extract_queries, prepare, raw_texts, save_json

CONDITIONS = ("image_text", "image_only", "text_only", "image_shuffled_text", "shuffled_image_text")


def deranged_index(index, size, seed, label):
    """Return a stable non-self index shared by all compared models."""
    digest = hashlib.sha256(f"{seed}|{index}|{label}".encode()).digest()
    return (index + 1 + int.from_bytes(digest[:8], "big") % (size - 1)) % size


class ModalityDataset(torch.utils.data.Dataset):
    def __init__(self, raw, processor, tokenizer, transform, condition, seed, limit=None):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.raw, self.processor, self.tokenizer, self.transform = raw, processor, tokenizer, transform
        self.condition, self.seed = condition, seed
        self.limit = min(limit or len(raw), len(raw))
        self.texts = raw_texts(raw)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        sample, text = self.raw[index], self.texts[index]
        image = sample["ref_image"]
        if self.condition == "image_only":
            text = ""
        elif self.condition == "text_only":
            image = Image.new("RGB", image.size, (127, 127, 127))
        elif self.condition == "image_shuffled_text":
            text = self.texts[deranged_index(index, len(self.raw), self.seed, "text")]
        elif self.condition == "shuffled_image_text":
            image = self.raw[deranged_index(index, len(self.raw), self.seed, "image")]["ref_image"]
        tokens = self.tokenizer(self.processor(text), return_tensors="pt", padding="max_length",
                                truncation=True, max_length=77)
        return {"ref_image": self.transform(image),
                "text_input_ids": tokens["input_ids"].squeeze(0),
                "text_attention_mask": tokens["attention_mask"].squeeze(0),
                "target_id": sample["target_id"], "candidate_id": sample["candidate_id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fashion-iq", "cirr"], default="fashion-iq")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--category", default="dress")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_queries", type=int)
    parser.add_argument("--distance_modes", nargs="+", choices=["mean", "probabilistic"], default=["mean"])
    parser.add_argument("--skip_uncertainty", action="store_true")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    state = prepare(args)
    processor, tokenizer, _, raw, gallery_data, device, model, gallery = state
    transform, rows = get_transform(224, False), []
    for condition in CONDITIONS:
        dataset = ModalityDataset(raw, processor, tokenizer, transform, condition, args.seed, args.max_queries)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn_query)
        query = extract_queries(model, loader, device, not args.skip_uncertainty)
        for mode in args.distance_modes:
            evaluated = evaluate_bundle(model, query, gallery, gallery_data.id_to_idx, device, mode)
            rows.append({"condition": condition, "distance_mode": mode, **evaluated})
    save_json({"schema_version": 1, "kind": "modality_reliance", "model_id": args.model_id,
               "dataset": args.dataset, "category": args.category, "seed": args.seed,
               "checkpoint": args.checkpoint,
               "ablation_definition": {
                   "image_only": "empty modification text",
                   "text_only": "constant mid-gray reference image",
                   "shuffling": "deterministic non-self pairing within the evaluation split",
               }, "conditions": rows}, args.output_file)


if __name__ == "__main__":
    main()
