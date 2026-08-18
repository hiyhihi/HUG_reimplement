import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

# Ensure root path imports work when script is run from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from eval import evaluate_retrieval, load_checkpoint
from data import (
    FashionIQQueryDataset,
    FashionIQGalleryDataset,
    CIRRQueryDataset,
    CIRRGalleryDataset,
    collate_fn_query,
    collate_fn_gallery,
    get_transform,
)


@dataclass
class CorruptionSample:
    query_index: int
    group: str
    severity: int
    seed: int
    candidate_id: str
    target_id: str
    original_text: str
    corrupted_text: Optional[str]
    metadata: Dict

    def to_dict(self):
        return asdict(self)


class CorruptionGenerator:
    IMAGE_GROUPS = ['A', 'B', 'C']
    TEXT_GROUPS = ['A', 'B', 'C']

    COLOR_SUBSTITUTIONS = {
        'black': 'dark',
        'white': 'light',
        'red': 'maroon',
        'blue': 'navy',
        'green': 'olive',
        'yellow': 'gold',
        'pink': 'rose',
        'brown': 'tan',
        'gray': 'silver',
        'grey': 'silver',
        'purple': 'plum',
        'orange': 'coral',
    }

    NEGATION_PHRASES = [
        'without the',
        'not',
        'no longer',
        'instead of',
        'but not',
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

    def corrupt_image(self, image: Image.Image, group: str, severity: int) -> Tuple[Image.Image, Dict]:
        if severity == 0:
            return image, {'type': 'identity'}

        if group == 'A':
            return self._image_group_A(image, severity)
        if group == 'B':
            return self._image_group_B(image, severity)
        if group == 'C':
            return self._image_group_C(image, severity)

        raise ValueError(f'Unknown image corruption group: {group}')

    def corrupt_text(self, text: str, group: str, severity: int, other_texts: Optional[List[str]] = None) -> Tuple[str, Dict]:
        if severity == 0:
            return text, {'type': 'identity'}

        if group == 'A':
            return self._text_group_A(text, severity)
        if group == 'B':
            return self._text_group_B(text, severity)
        if group == 'C':
            return self._text_group_C(text, severity, other_texts)

        raise ValueError(f'Unknown text corruption group: {group}')

    def _image_group_A(self, image: Image.Image, severity: int) -> Tuple[Image.Image, Dict]:
        # Mild semantic-preserving perturbations
        if severity == 1:
            corrupted = image.filter(ImageFilter.GaussianBlur(radius=1))
            return corrupted, {'type': 'gaussian_blur', 'radius': 1}
        if severity == 2:
            corrupted = ImageOps.autocontrast(image)
            corrupted = corrupted.filter(ImageFilter.GaussianBlur(radius=1.5))
            return corrupted, {'type': 'blur_autocontrast', 'radius': 1.5}
        if severity == 3:
            corrupted = image.filter(ImageFilter.GaussianBlur(radius=2.0))
            corrupted = self._apply_random_occlusion(corrupted, area_ratio=0.08)
            return corrupted, {'type': 'blur_occlusion', 'radius': 2.0, 'occlusion_area': 0.08}
        if severity >= 4:
            corrupted = image.filter(ImageFilter.GaussianBlur(radius=3.0))
            corrupted = self._apply_random_occlusion(corrupted, area_ratio=0.15)
            corrupted = self._adjust_color(corrupted, contrast=0.7, brightness=1.1)
            return corrupted, {'type': 'blur_occlusion_color', 'radius': 3.0, 'occlusion_area': 0.15}
        return image, {'type': 'identity'}

    def _image_group_B(self, image: Image.Image, severity: int) -> Tuple[Image.Image, Dict]:
        # Stronger spatial and color corruptions
        if severity == 1:
            return ImageOps.equalize(image), {'type': 'equalize'}
        if severity == 2:
            return self._apply_random_occlusion(image, area_ratio=0.06), {'type': 'occlusion', 'area': 0.06}
        if severity == 3:
            corrupted = self._apply_random_crop(image, crop_ratio=0.85)
            corrupted = self._apply_random_occlusion(corrupted, area_ratio=0.1)
            return corrupted, {'type': 'crop_occlusion', 'crop_ratio': 0.85, 'occlusion_area': 0.10}
        if severity >= 4:
            corrupted = self._apply_random_crop(image, crop_ratio=0.75)
            corrupted = self._apply_random_occlusion(corrupted, area_ratio=0.15)
            corrupted = self._apply_random_noise(corrupted, intensity=0.12)
            return corrupted, {'type': 'crop_occlusion_noise', 'crop_ratio': 0.75, 'occlusion_area': 0.15}
        return image, {'type': 'identity'}

    def _image_group_C(self, image: Image.Image, severity: int) -> Tuple[Image.Image, Dict]:
        # Mismatch / contrastive corruption: heavy degradation or misalignment
        if severity == 1:
            return image.convert('L').convert('RGB'), {'type': 'grayscale'}
        if severity == 2:
            corrupted = self._apply_random_crop(image, crop_ratio=0.75)
            return corrupted, {'type': 'crop', 'crop_ratio': 0.75}
        if severity == 3:
            corrupted = self._apply_random_occlusion(image, area_ratio=0.18)
            corrupted = self._apply_random_noise(corrupted, intensity=0.14)
            return corrupted, {'type': 'occlusion_noise', 'occlusion_area': 0.18}
        if severity >= 4:
            corrupted = self._apply_random_crop(image, crop_ratio=0.6)
            corrupted = self._apply_random_occlusion(corrupted, area_ratio=0.20)
            corrupted = self._apply_random_noise(corrupted, intensity=0.20)
            return corrupted, {'type': 'extreme_crop_occlusion_noise', 'crop_ratio': 0.60, 'occlusion_area': 0.20}
        return image, {'type': 'identity'}

    def _text_group_A(self, text: str, severity: int) -> Tuple[str, Dict]:
        if severity == 1:
            return self._drop_adjective(text), {'type': 'drop_adjective'}
        if severity == 2:
            return self._replace_color_word(text), {'type': 'replace_color'}
        if severity == 3:
            return self._shuffle_adjacent(text), {'type': 'shuffle_adjacent'}
        if severity >= 4:
            return self._add_negation(text), {'type': 'add_negation'}
        return text, {'type': 'identity'}

    def _text_group_B(self, text: str, severity: int) -> Tuple[str, Dict]:
        if severity == 1:
            return self._drop_adjective(text), {'type': 'drop_adjective'}
        if severity == 2:
            return self._replace_color_word(text), {'type': 'replace_color'}
        if severity == 3:
            return self._replace_adverb(text), {'type': 'replace_adverb'}
        if severity >= 4:
            return self._scramble_tokens(text), {'type': 'scramble'}
        return text, {'type': 'identity'}

    def _text_group_C(self, text: str, severity: int, other_texts: Optional[List[str]] = None) -> Tuple[str, Dict]:
        if other_texts is None or len(other_texts) == 0:
            return self._add_negation(text), {'type': 'negation'}
        if severity == 1:
            other = self.rng.choice(other_texts)
            return other, {'type': 'swap_with_other'}
        if severity == 2:
            return self._add_negation(text), {'type': 'add_negation'}
        if severity == 3:
            other = self.rng.choice(other_texts)
            return f"{other} but not {text}", {'type': 'swap_and_negate'}
        if severity >= 4:
            other = self.rng.choice(other_texts)
            return f"{other} instead of {text}", {'type': 'strong_mismatch'}
        return text, {'type': 'identity'}

    def _drop_adjective(self, text: str) -> str:
        tokens = text.split()
        adjectives = [i for i, tok in enumerate(tokens) if tok.endswith('y') or tok.endswith('ful') or tok.endswith('less')]
        if adjectives:
            idx = self.rng.choice(adjectives)
            tokens.pop(idx)
        elif len(tokens) > 1:
            idx = self.rng.randrange(len(tokens))
            tokens.pop(idx)
        return ' '.join(tokens)

    def _replace_color_word(self, text: str) -> str:
        tokens = text.split()
        for i, tok in enumerate(tokens):
            key = tok.lower().strip('.,')
            if key in self.COLOR_SUBSTITUTIONS:
                tokens[i] = tokens[i].replace(tok, self.COLOR_SUBSTITUTIONS[key])
                return ' '.join(tokens)
        # fallback: swap a random token with a neighboring token
        return self._shuffle_adjacent(text)

    def _replace_adverb(self, text: str) -> str:
        tokens = text.split()
        adverbs = [i for i, tok in enumerate(tokens) if tok.endswith('ly')]
        if adverbs:
            idx = self.rng.choice(adverbs)
            tokens[idx] = 'slightly'
            return ' '.join(tokens)
        return self._shuffle_adjacent(text)

    def _shuffle_adjacent(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) < 3:
            return text
        idx = self.rng.randrange(len(tokens) - 1)
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
        return ' '.join(tokens)

    def _add_negation(self, text: str) -> str:
        phrase = self.rng.choice(self.NEGATION_PHRASES)
        return f"{phrase} {text}"

    def _apply_random_occlusion(self, image: Image.Image, area_ratio: float) -> Image.Image:
        width, height = image.size
        occlusion_width = int(width * self.rng.uniform(area_ratio * 0.8, area_ratio * 1.2))
        occlusion_height = int(height * self.rng.uniform(area_ratio * 0.8, area_ratio * 1.2))
        x0 = self.rng.randrange(0, max(1, width - occlusion_width))
        y0 = self.rng.randrange(0, max(1, height - occlusion_height))
        occlusion = Image.new('RGB', (occlusion_width, occlusion_height), (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255)))
        image = image.copy()
        image.paste(occlusion, (x0, y0))
        return image

    def _apply_random_crop(self, image: Image.Image, crop_ratio: float) -> Image.Image:
        width, height = image.size
        new_w = int(width * crop_ratio)
        new_h = int(height * crop_ratio)
        left = self.rng.randrange(0, width - new_w + 1)
        top = self.rng.randrange(0, height - new_h + 1)
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((width, height), Image.BILINEAR)

    def _apply_random_noise(self, image: Image.Image, intensity: float) -> Image.Image:
        pixels = image.convert('RGB').load()
        width, height = image.size
        noisy = Image.new('RGB', (width, height))
        noisy_pixels = noisy.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                nr = min(255, max(0, int(r + self.rng.uniform(-1, 1) * 255 * intensity)))
                ng = min(255, max(0, int(g + self.rng.uniform(-1, 1) * 255 * intensity)))
                nb = min(255, max(0, int(b + self.rng.uniform(-1, 1) * 255 * intensity)))
                noisy_pixels[x, y] = (nr, ng, nb)
        return noisy

    def _adjust_color(self, image: Image.Image, contrast: float = 1.0, brightness: float = 1.0) -> Image.Image:
        image = ImageOps.autocontrast(image)
        enhancer = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        return enhancer(image)


class CorruptedQueryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        image_transform,
        group: str,
        severity: int,
        split: str,
        seed: int,
        save_images: bool = False,
        saved_images_dir: Optional[Path] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.group = group
        self.severity = severity
        self.split = split
        self.generator = CorruptionGenerator(seed)
        self.save_images = save_images
        self.saved_images_dir = saved_images_dir
        self.other_texts = [self._get_raw_text(i) for i in range(len(dataset))]
        if save_images and saved_images_dir is not None:
            saved_images_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def _get_raw_text(self, idx: int) -> str:
        sample = self.dataset[idx]
        if isinstance(sample['text_input_ids'], str):
            return sample['text_input_ids']
        if isinstance(sample['text_input_ids'], list):
            return ' '.join(sample['text_input_ids'])
        # fallback: decode token ids if tokenizer known
        if hasattr(self.dataset, 'tokenizer') and self.dataset.tokenizer is not None:
            return self.dataset.tokenizer.decode(sample['text_input_ids'], skip_special_tokens=True)
        return ''

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        ref_image = sample['ref_image']
        text = sample['text_input_ids'] if isinstance(sample['text_input_ids'], str) else self._get_raw_text(idx)

        corrupted_text, text_meta = self.generator.corrupt_text(
            text, self.group, self.severity, other_texts=self.other_texts
        )
        corrupted_image = ref_image
        image_meta = {'type': 'identity'}
        if self.group in ['A', 'B']:
            corrupted_image, image_meta = self.generator.corrupt_image(ref_image, self.group, self.severity)

        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                corrupted_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77,
            )
            text_input_ids = tokenized['input_ids'].squeeze(0)
            text_attention_mask = tokenized['attention_mask'].squeeze(0)
        else:
            text_input_ids = corrupted_text
            text_attention_mask = None

        if self.image_transform is not None:
            corrupted_image = self.image_transform(corrupted_image)

        if self.save_images and self.saved_images_dir is not None and idx < 20:
            sample_path = self.saved_images_dir / f'{idx}_{self.group}_{self.severity}.png'
            corrupted_image_to_save = corrupted_image
            if isinstance(corrupted_image_to_save, torch.Tensor):
                corrupted_image_to_save = transforms.ToPILImage()(corrupted_image_to_save)
            corrupted_image_to_save.save(sample_path)

        return {
            'ref_image': corrupted_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'target_id': sample['target_id'],
            'candidate_id': sample['candidate_id'],
            'corruption_metadata': {
                'text': text_meta,
                'image': image_meta,
                'group': self.group,
                'severity': self.severity,
                'seed': self.generator.seed,
            }
        }


def build_query_dataset(dataset_name: str, data_root: str, split: str, category: Optional[str], processor, tokenizer, image_transform):
    if dataset_name == 'fashion-iq':
        return FashionIQQueryDataset(
            data_root=data_root,
            split=split,
            category=category,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )
    if dataset_name == 'cirr':
        return CIRRQueryDataset(
            data_root=data_root,
            split=split,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def build_gallery_dataset(dataset_name: str, data_root: str, split: str, category: Optional[str], image_transform):
    if dataset_name == 'fashion-iq':
        return FashionIQGalleryDataset(
            data_root=data_root,
            split=split,
            category=category,
            image_transform=image_transform,
        )
    if dataset_name == 'cirr':
        return CIRRGalleryDataset(
            data_root=data_root,
            split=split,
            image_transform=image_transform,
        )
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def get_tokenizer_and_processor():
    from lavis.processors import load_processor
    from transformers import BertTokenizer

    processor = load_processor('blip_caption').build()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    return processor, tokenizer


def compute_spearman_correlation(ranks: List[int], scores: List[float]) -> float:
    if len(ranks) != len(scores):
        raise ValueError('Lengths must match')
    n = len(ranks)
    rank_r = rankdata(ranks)
    rank_s = rankdata(scores)
    mean_r = sum(rank_r) / n
    mean_s = sum(rank_s) / n
    cov = sum((xr - mean_r) * (xs - mean_s) for xr, xs in zip(rank_r, rank_s))
    var_r = sum((xr - mean_r) ** 2 for xr in rank_r)
    var_s = sum((xs - mean_s) ** 2 for xs in rank_s)
    if var_r == 0 or var_s == 0:
        return 0.0
    return cov / (var_r ** 0.5) / (var_s ** 0.5)


def rankdata(values: List[float]) -> List[float]:
    sorted_values = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0] * len(values)
    cur_rank = 1
    for i, (_, idx) in enumerate(sorted_values):
        if i > 0 and sorted_values[i][0] != sorted_values[i - 1][0]:
            cur_rank = i + 1
        ranks[idx] = cur_rank
    return ranks


def generate_corrupted_metadata(
    dataset_name: str,
    data_root: str,
    split: str,
    category: Optional[str],
    group: str,
    severity: int,
    seed: int,
    output_file: str,
    max_samples: Optional[int] = None,
    save_images: bool = False,
):
    processor, tokenizer = get_tokenizer_and_processor()
    image_transform = get_transform(image_size=224, is_train=False)
    raw_dataset = build_query_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        category=category,
        processor=None,
        tokenizer=None,
        image_transform=None,
    )

    generator = CorruptionGenerator(seed)
    samples = []
    total = len(raw_dataset) if max_samples is None else min(max_samples, len(raw_dataset))
    save_dir = Path(output_file).resolve().parent / 'corrupted_images'
    if save_images:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(total), desc='Generating corrupted metadata'):
        sample = raw_dataset[idx]
        raw_text = sample['text_input_ids'] if isinstance(sample['text_input_ids'], str) else str(sample['text_input_ids'])
        corrupted_text, text_meta = generator.corrupt_text(raw_text, group, severity, other_texts=None)
        corrupted_image_meta = {'group': group, 'severity': severity, 'seed': seed}
        samples.append(CorruptionSample(
            query_index=idx,
            group=group,
            severity=severity,
            seed=seed,
            candidate_id=sample['candidate_id'],
            target_id=sample['target_id'],
            original_text=raw_text,
            corrupted_text=corrupted_text,
            metadata={'text': text_meta, 'image': corrupted_image_meta},
        ).to_dict())

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f'Wrote corrupted metadata for {len(samples)} samples to {output_file}')


def evaluate_corruption(
    dataset_name: str,
    data_root: str,
    split: str,
    category: Optional[str],
    checkpoint: str,
    group: str,
    severity: int,
    seed: int,
    distance_mode: str,
    output_file: Optional[str],
    batch_size: int,
    num_workers: int,
):
    processor, tokenizer = get_tokenizer_and_processor()
    image_transform = get_transform(image_size=224, is_train=False)

    raw_dataset = build_query_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        category=category,
        processor=None,
        tokenizer=None,
        image_transform=None,
    )

    corrupted_query_dataset = CorruptedQueryDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        image_transform=image_transform,
        group=group,
        severity=severity,
        split=split,
        seed=seed,
        save_images=False,
        saved_images_dir=None,
    )

    query_loader = torch.utils.data.DataLoader(
        corrupted_query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_query,
        pin_memory=True,
    )

    gallery_dataset = build_gallery_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        category=category,
        image_transform=image_transform,
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_gallery,
        pin_memory=True,
    )

    gallery_id_to_idx = gallery_dataset.id_to_idx

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(checkpoint, device)

    metrics = evaluate_retrieval(
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        gallery_id_to_idx=gallery_id_to_idx,
        device=device,
        distance_mode=distance_mode,
    )

    results = {
        'dataset': dataset_name,
        'split': split,
        'category': category,
        'checkpoint': checkpoint,
        'group': group,
        'severity': severity,
        'seed': seed,
        'distance_mode': distance_mode,
        'metrics': metrics,
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'Saved evaluation results to {output_file}')

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Corruption-based robustness evaluation for HUG')
    subparsers = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--dataset', type=str, default='fashion-iq', choices=['fashion-iq', 'cirr'])
    common.add_argument('--data_root', type=str, required=True)
    common.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    common.add_argument('--group', type=str, default='A', choices=['A', 'B', 'C'])
    common.add_argument('--severity', type=int, default=1, choices=[0, 1, 2, 3, 4])
    common.add_argument('--seed', type=int, default=42)
    common.add_argument('--category', type=str, default='dress', choices=['dress', 'shirt', 'toptee'])

    generate_parser = subparsers.add_parser('generate', parents=[common], help='Generate corrupted query metadata')
    generate_parser.add_argument('--output_file', type=str, required=True)
    generate_parser.add_argument('--max_samples', type=int, default=None)
    generate_parser.add_argument('--save_images', action='store_true')

    eval_parser = subparsers.add_parser('evaluate', parents=[common], help='Evaluate model on corrupted queries')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--distance_mode', type=str, default='probabilistic', choices=['probabilistic', 'mean'])
    eval_parser.add_argument('--output_file', type=str, default=None)
    eval_parser.add_argument('--batch_size', type=int, default=32)
    eval_parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'generate':
        generate_corrupted_metadata(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split=args.split,
            category=args.category,
            group=args.group,
            severity=args.severity,
            seed=args.seed,
            output_file=args.output_file,
            max_samples=args.max_samples,
            save_images=args.save_images,
        )
    elif args.command == 'evaluate':
        evaluate_corruption(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split=args.split,
            category=args.category,
            checkpoint=args.checkpoint,
            group=args.group,
            severity=args.severity,
            seed=args.seed,
            distance_mode=args.distance_mode,
            output_file=args.output_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == '__main__':
    main()
