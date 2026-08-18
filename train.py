# """
# Training Script for HUG Model

# This script implements the training procedure for HUG following the paper specifications:
# - Learning rate: 3e-5
# - Optimizer: AdamW (²1=0.9, ²2=0.999)
# - Batch size: 32
# - Loss weights: lambda_FC=0.5, lambda_Cord=0.1
# """

# import os
# import sys

# # Add LAVIS to path for processor
# lavis_path = os.path.expanduser("~/AAAI26-HUG/ref/LAVIS")
# if lavis_path not in sys.path:
#     sys.path.insert(0, lavis_path)

# import argparse
# import yaml
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm
# try:
#     import wandb
#     HAS_WANDB = True
# except ImportError:
#     HAS_WANDB = False
#     print("wandb not installed, logging to file only")

# from models import HUGModel
# from modules import HUGLoss
# from data import FashionIQDataset, CIRRDataset, collate_fn, get_transform

# # Import LAVIS processor (instead of HuggingFace Blip2Processor)
# from lavis.processors import load_processor
# from transformers import BertTokenizer


# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Train HUG model for Composed Image Retrieval')

#     # Dataset arguments
#     parser.add_argument('--dataset', type=str, default='fashion-iq',
#                        choices=['fashion-iq', 'cirr'],
#                        help='Dataset to use')
#     parser.add_argument('--data_root', type=str, required=True,
#                        help='Root directory of the dataset')
#     parser.add_argument('--category', type=str, default='dress',
#                        choices=['dress', 'shirt', 'toptee'],
#                        help='Fashion-IQ category (only for Fashion-IQ)')

#     # Model arguments
#     parser.add_argument('--num_queries', type=int, default=32,
#                        help='Number of learnable query tokens (K)')
#     parser.add_argument('--hidden_dim', type=int, default=768,
#                        help='Hidden dimension')
#     parser.add_argument('--blip_model', type=str, default='pretrain',
#                        choices=['pretrain', 'pretrain_vitL', 'coco'],
#                        help='BLIP-2 LAVIS model type')

#     # Training arguments
#     parser.add_argument('--batch_size', type=int, default=32,
#                        help='Batch size per GPU')
#     parser.add_argument('--num_epochs', type=int, default=30,
#                        help='Number of training epochs')
#     parser.add_argument('--lr', type=float, default=3e-5,
#                        help='Learning rate')
#     parser.add_argument('--weight_decay', type=float, default=0.01,
#                        help='Weight decay')
#     parser.add_argument('--warmup_epochs', type=int, default=2,
#                        help='Number of warmup epochs')

#     # Loss arguments
#     parser.add_argument('--lambda_fc', type=float, default=0.5,
#                        help='Weight for fine-grained contrastive loss')
#     parser.add_argument('--lambda_cord', type=float, default=0.1,
#                        help='Weight for coordination loss')

#     # Other arguments
#     parser.add_argument('--num_workers', type=int, default=4,
#                        help='Number of data loading workers')
#     parser.add_argument('--output_dir', type=str, default='./checkpoints',
#                        help='Directory to save checkpoints')
#     parser.add_argument('--log_interval', type=int, default=10,
#                        help='Log every N steps')
#     parser.add_argument('--save_interval', type=int, default=1,
#                        help='Save checkpoint every N epochs')
#     parser.add_argument('--use_wandb', action='store_true',
#                        help='Use Weights & Biases for logging')
#     parser.add_argument('--seed', type=int, default=42,
#                        help='Random seed')

#     return parser.parse_args()


# def set_seed(seed: int):
#     """Set random seed for reproducibility"""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     import numpy as np
#     import random
#     np.random.seed(seed)
#     random.seed(seed)


# def create_dataloader(args, split='train'):
#     """Create dataloader for training or validation"""
#     # Get image transform
#     image_transform = get_transform(image_size=224, is_train=(split == 'train'))

#     # Get text processor (LAVIS for text preprocessing)
#     processor = load_processor('blip_caption').build()

#     # Get BLIP2 tokenizer (for actual tokenization)
#     # BLIP2 uses BertTokenizer with special tokens
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
#     tokenizer.add_special_tokens({'bos_token': '[DEC]'})

#     # Create dataset
#     if args.dataset == 'fashion-iq':
#         dataset = FashionIQDataset(
#             data_root=args.data_root,
#             split=split,
#             category=args.category,
#             processor=processor,
#             tokenizer=tokenizer,
#             image_transform=image_transform
#         )
#     elif args.dataset == 'cirr':
#         dataset = CIRRDataset(
#             data_root=args.data_root,
#             split=split,
#             processor=processor,
#             tokenizer=tokenizer,
#             image_transform=image_transform
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {args.dataset}")

#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=(split == 'train'),
#         num_workers=args.num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True
#     )

#     return dataloader


# def train_one_epoch(
#     model: nn.Module,
#     dataloader: DataLoader,
#     criterion: HUGLoss,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     epoch: int,
#     args
# ):
#     """Train for one epoch"""
#     model.train()

#     total_loss = 0
#     total_loss_hc = 0
#     total_loss_fc = 0
#     total_loss_cord = 0

#     pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

#     for step, batch in enumerate(pbar):
#         # Move batch to device
#         ref_images = batch['ref_images'].to(device)
#         target_images = batch['target_images'].to(device)
#         text_input_ids = batch['text_input_ids'].to(device)
#         text_attention_mask = batch['text_attention_mask'].to(device)

#         # Forward pass
#         outputs = model(
#             ref_pixel_values=ref_images,
#             text_input_ids=text_input_ids,
#             text_attention_mask=text_attention_mask,
#             target_pixel_values=target_images
#         )
#         print("\n===== FEATURE DEBUG =====")

#         print("mu_q requires_grad:", outputs['mu_q'].requires_grad)
#         print("mu_c requires_grad:", outputs['mu_c'].requires_grad)

#         print("sigma_q requires_grad:", outputs['sigma_q'].requires_grad)
#         print("sigma_c requires_grad:", outputs['sigma_c'].requires_grad)

#         # For L_Cord, we need mismatched pairs
#         # Simple approach: shuffle text within batch
#         shuffled_indices = torch.randperm(text_input_ids.size(0))
#         mismatched_text_ids = text_input_ids[shuffled_indices]
#         mismatched_text_mask = text_attention_mask[shuffled_indices]

#         # Get sigma_m for mismatched pairs
#         mismatched_components = model.extract_query_components(
#             ref_pixel_values=ref_images,
#             text_input_ids=mismatched_text_ids,
#             text_attention_mask=mismatched_text_mask
#         )
#         sigma_m_mismatched = mismatched_components['sigma_m']

#         # Compute loss
#         loss_dict = criterion(
#             mu_q=outputs['mu_q'],
#             sigma_q=outputs['sigma_q'],
#             mu_c=outputs['mu_c'],
#             sigma_c=outputs['sigma_c'],
#             sigma_m_matched=outputs['sigma_m'],
#             sigma_m_mismatched=sigma_m_mismatched
#         )

#         loss = loss_dict['total_loss']

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         qformer = model.blip_backbone.blip_model.Qformer

#         grad = (
#             qformer.bert.encoder.layer[0]
#             .attention.self.query.weight.grad
#         )

#         if grad is None:
#             print("\n===== GRAD DEBUG =====")
#             print("QFormer grad is NONE")
#         else:
#             print("\n===== GRAD DEBUG =====")
#             print(f"QFormer grad norm: {grad.norm().item():.6f}")

#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         # Update statistics
#         total_loss += loss.item()
#         total_loss_hc += loss_dict['loss_hc'].item()
#         total_loss_fc += loss_dict['loss_fc'].item()
#         total_loss_cord += loss_dict['loss_cord'].item()

#         # Update progress bar
#         pbar.set_postfix({
#             'loss': loss.item(),
#             'loss_hc': loss_dict['loss_hc'].item(),
#             'loss_fc': loss_dict['loss_fc'].item(),
#             'loss_cord': loss_dict['loss_cord'].item()
#         })

#         # Log to wandb
#         if args.use_wandb and step % args.log_interval == 0:
#             wandb.log({
#                 'train/loss': loss.item(),
#                 'train/loss_hc': loss_dict['loss_hc'].item(),
#                 'train/loss_fc': loss_dict['loss_fc'].item(),
#                 'train/loss_cord': loss_dict['loss_cord'].item(),
#                 'train/step': epoch * len(dataloader) + step
#             })

#     # Return average losses
#     num_batches = len(dataloader)
#     return {
#         'loss': total_loss / num_batches,
#         'loss_hc': total_loss_hc / num_batches,
#         'loss_fc': total_loss_fc / num_batches,
#         'loss_cord': total_loss_cord / num_batches
#     }


# def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
#     """Save model checkpoint"""
#     os.makedirs(args.output_dir, exist_ok=True)
#     checkpoint_path = os.path.join(args.output_dir, filename)

#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'args': vars(args)
#     }, checkpoint_path)

#     print(f"Checkpoint saved to {checkpoint_path}")


# def main():
#     """Main training function"""
#     args = parse_args()
#     set_seed(args.seed)

#     # Initialize wandb
#     if args.use_wandb:
#         wandb.init(project='hug-cir', config=vars(args))

#     # Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Create dataloaders
#     print("Creating dataloaders...")
#     train_loader = create_dataloader(args, split='train')
#     print(f"Training samples: {len(train_loader.dataset)}")

#     # Create model
#     print("Creating model...")
#     model = HUGModel(
#         num_queries=args.num_queries,
#         hidden_dim=args.hidden_dim,
#         blip_model_name=args.blip_model,
#         freeze_vision_encoder=True
#     ).to(device)

#     print("\n===== REQUIRES GRAD CHECK =====")
#     for name, param in model.named_parameters():

#         if "Qformer" in name:

#             print(name, param.requires_grad)
#             break

#     # Create loss function
#     criterion = HUGLoss(
#         lambda_fc=args.lambda_fc,
#         lambda_cord=args.lambda_cord
#     ).to(device)

#     # Create optimizer
#     optimizer = AdamW(
#         model.parameters(),
#         lr=args.lr,
#         betas=(0.9, 0.999),
#         weight_decay=args.weight_decay
#     )

#     num_qformer = 0

#     for group in optimizer.param_groups:
#         for p in group['params']:
#             num_qformer += p.numel()

#     print("optimizer params:", num_qformer)

#     for name, param in model.named_parameters():

#         if "Qformer" in name:

#             found = any(
#                 param is p
#                 for group in optimizer.param_groups
#                 for p in group['params']
#             )

#             print(name, "in optimizer:", found)
#             break

#     # Create learning rate scheduler
#     scheduler = CosineAnnealingLR(
#         optimizer,
#         T_max=args.num_epochs,
#         eta_min=1e-6
#     )

#     # Training loop
#     print("Starting training...")
#     for epoch in range(1, args.num_epochs + 1):
#         # Train one epoch
#         train_metrics = train_one_epoch(
#             model, train_loader, criterion, optimizer, device, epoch, args
#         )

#         # Print epoch summary
#         print(f"\nEpoch {epoch}/{args.num_epochs} Summary:")
#         print(f"  Loss: {train_metrics['loss']:.4f}")
#         print(f"  Loss HC: {train_metrics['loss_hc']:.4f}")
#         print(f"  Loss FC: {train_metrics['loss_fc']:.4f}")
#         print(f"  Loss Cord: {train_metrics['loss_cord']:.4f}")

#         # Log epoch metrics
#         if args.use_wandb:
#             wandb.log({
#                 'train/epoch_loss': train_metrics['loss'],
#                 'train/epoch_loss_hc': train_metrics['loss_hc'],
#                 'train/epoch_loss_fc': train_metrics['loss_fc'],
#                 'train/epoch_loss_cord': train_metrics['loss_cord'],
#                 'train/epoch': epoch,
#                 'train/lr': optimizer.param_groups[0]['lr']
#             })

#         # Update learning rate
#         scheduler.step()

#         # Save checkpoint
#         if epoch % args.save_interval == 0:
#             save_checkpoint(model, optimizer, epoch, args, f'checkpoint_epoch_{epoch}.pth')

#     # Save final checkpoint
#     save_checkpoint(model, optimizer, args.num_epochs, args, 'checkpoint_final.pth')

#     print("Training completed!")

#     if args.use_wandb:
#         wandb.finish()


# if __name__ == '__main__':
#     main()


"""
Training Script for HUG Model

This script implements the training procedure for HUG following the paper specifications:
- Learning rate: 3e-5 (Heads) / 3e-6 (Backbone)
- Optimizer: AdamW (b1=0.9, b2=0.999)
- Batch size: 32
- Loss weights: lambda_FC=0.5, lambda_Cord=0.1
"""

import os
import sys
import math

# Add LAVIS to path for processor
lavis_path = os.path.expanduser("~/AAAI26-HUG/ref/LAVIS")
if lavis_path not in sys.path:
    sys.path.insert(0, lavis_path)

import argparse
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed, logging to file only")

from models import HUGModel
from modules import HUGLoss
from modules.robustness_training import (
    apply_modality_dropout, calibration_pair, clean_teacher_kl,
    monotonic_ranking_loss, uncertainty_scalar,
)
from data import FashionIQDataset, CIRRDataset, collate_fn, get_transform

# Import LAVIS processor (instead of HuggingFace Blip2Processor)
from lavis.processors import load_processor
from transformers import BertTokenizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train HUG model for Composed Image Retrieval')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='fashion-iq',
                       choices=['fashion-iq', 'cirr'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--category', type=str, default='dress',
                       choices=['dress', 'shirt', 'toptee'],
                       help='Fashion-IQ category (only for Fashion-IQ)')

    # Model arguments
    parser.add_argument('--num_queries', type=int, default=32,
                       help='Number of learnable query tokens (K)')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension')
    parser.add_argument('--blip_model', type=str, default='pretrain',
                       choices=['pretrain', 'pretrain_vitL', 'coco'],
                       help='BLIP-2 LAVIS model type')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Learning rate (Max LR for new heads)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Number of warmup epochs')

    # Loss arguments
    parser.add_argument('--lambda_fc', type=float, default=0.5,
                       help='Weight for fine-grained contrastive loss')
    parser.add_argument('--lambda_cord', type=float, default=0.1,
                       help='Weight for coordination loss')
    parser.add_argument('--recipe', choices=['legacy', 'point', 'paper'], default='paper',
                       help='Training objective and representation recipe')
    parser.add_argument('--loss_lr_multiplier', type=float, default=100.0,
                       help='LR multiplier for learnable loss scalars')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                       help='Initialize model weights only (for point -> paper stage 2)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Resume a complete training state from checkpoint_last.pth')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze the pretrained BLIP/Q-Former mean encoder')

    # RC-HUG U1/U2/U3: disabled by default, so baseline recipes stay unchanged.
    parser.add_argument('--lambda_monotonic', type=float, default=0.0,
                       help='U1 weight for paired severity-monotonic uncertainty calibration')
    parser.add_argument('--monotonic_margin', type=float, default=0.005,
                       help='Required uncertainty increase from low to high severity')
    parser.add_argument('--monotonic_low_severity', type=float, default=0.10,
                       help='U1 low image-blur/text-dropout severity')
    parser.add_argument('--monotonic_high_severity', type=float, default=0.30,
                       help='U1 high image-blur/text-dropout severity')
    parser.add_argument('--monotonic_modality', choices=['alternate', 'image', 'text'], default='alternate',
                       help='Modality calibrated by U1 on each minibatch')
    parser.add_argument('--monotonic_query_weight', type=float, default=1.0,
                       help='Relative U1 weight on fused query uncertainty')
    parser.add_argument('--modality_dropout_prob', type=float, default=0.0,
                       help='U2 probability of dropping image or text information per training sample')
    parser.add_argument('--modality_dropout_text_rate', type=float, default=0.30,
                       help='U2 token-drop rate after text modality is selected')
    parser.add_argument('--lambda_kd', type=float, default=0.0,
                       help='U3 clean Point-teacher KL consistency weight')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                       help='Frozen Point checkpoint required when --lambda_kd is positive')
    parser.add_argument('--teacher_device', choices=['cpu', 'cuda'], default='cpu',
                       help='U3 teacher device; CPU avoids a second BLIP/Q-Former consuming GPU VRAM')
    parser.add_argument('--kd_temperature', type=float, default=0.07,
                       help='Temperature for U3 in-batch retrieval-distribution KL')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Archive every N epochs; 0 disables numbered archives')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Run validation every N epochs; 0 disables validation')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--tqdm_mininterval', type=float, default=1.0,
                       help='Minimum seconds between progress-bar refreshes')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(args, split='train'):
    """Create dataloader for training or validation"""
    # Get image transform
    image_transform = get_transform(image_size=224, is_train=(split == 'train'))

    # Get text processor (LAVIS for text preprocessing)
    processor = load_processor('blip_caption').build()

    # Get BLIP2 tokenizer (for actual tokenization)
    # BLIP2 uses BertTokenizer with special tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})

    # Create dataset
    if args.dataset == 'fashion-iq':
        dataset = FashionIQDataset(
            data_root=args.data_root,
            split=split,
            category=args.category,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )
    elif args.dataset == 'cirr':
        dataset = CIRRDataset(
            data_root=args.data_root,
            split=split,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: HUGLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    teacher=None,
    teacher_device=None,
):
    """Train one epoch, optionally adding RC-HUG U1/U2/U3 objectives."""
    model.train()
    if args.freeze_backbone:
        # Keep dropout disabled in frozen mean encoder while uncertainty heads train.
        model.blip_backbone.eval()

    totals = {key: 0.0 for key in ('loss', 'loss_hc', 'loss_fc', 'loss_cord', 'loss_mono', 'loss_kd', 'drop_image', 'drop_text')}
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', mininterval=args.tqdm_mininterval)

    for step, batch in enumerate(pbar):
        clean_ref_images = batch['ref_images'].to(device)
        target_images = batch['target_images'].to(device)
        clean_text_ids = batch['text_input_ids'].to(device)
        clean_text_mask = batch['text_attention_mask'].to(device)

        # U2: corruption is applied in-memory only; source data remain untouched.
        ref_images, text_input_ids, text_attention_mask, dropout_stats = apply_modality_dropout(
            clean_ref_images, clean_text_ids, clean_text_mask,
            args.modality_dropout_prob, args.modality_dropout_text_rate,
        )
        outputs = model(
            ref_pixel_values=ref_images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            target_pixel_values=target_images,
            compute_uncertainty=args.recipe != 'point',
        )

        sigma_m_mismatched = None
        if criterion.needs_mismatched:
            # A cyclic shift is a derangement, so no matched pair is reused.
            batch_size = text_input_ids.size(0)
            shift = torch.randint(1, batch_size, (), device=text_input_ids.device)
            shuffled_indices = torch.arange(batch_size, device=text_input_ids.device).roll(shift.item())
            sigma_m_mismatched = model.extract_coordination_uncertainty(
                ref_pixel_values=ref_images,
                text_input_ids=text_input_ids[shuffled_indices],
                text_attention_mask=text_attention_mask[shuffled_indices],
            )

        loss_dict = criterion(
            mu_q=outputs['mu_q'], sigma_q=outputs['sigma_q'],
            mu_c=outputs['mu_c'], sigma_c=outputs['sigma_c'],
            sigma_m_matched=outputs['sigma_m'], sigma_m_mismatched=sigma_m_mismatched,
        )
        loss = loss_dict['total_loss']
        loss_mono = loss.detach().new_zeros(())
        loss_kd = loss.detach().new_zeros(())

        if args.lambda_monotonic > 0:
            modality = args.monotonic_modality
            if modality == 'alternate':
                modality = 'image' if ((epoch - 1) * len(dataloader) + step) % 2 == 0 else 'text'
            low_view, high_view = calibration_pair(
                clean_ref_images, clean_text_ids, clean_text_mask, modality,
                args.monotonic_low_severity, args.monotonic_high_severity,
            )
            low_parts = model.extract_query_components(*low_view)
            high_parts = model.extract_query_components(*high_view)
            component_key = 'sigma_r' if modality == 'image' else 'sigma_t'
            low_component = uncertainty_scalar(low_parts[component_key], model.uncertainty_is_variance)
            high_component = uncertainty_scalar(high_parts[component_key], model.uncertainty_is_variance)
            low_query = uncertainty_scalar(low_parts['sigma_q'], model.uncertainty_is_variance)
            high_query = uncertainty_scalar(high_parts['sigma_q'], model.uncertainty_is_variance)
            loss_mono = monotonic_ranking_loss(low_component, high_component, args.monotonic_margin)
            loss_mono = loss_mono + args.monotonic_query_weight * monotonic_ranking_loss(
                low_query, high_query, args.monotonic_margin
            )
            loss = loss + args.lambda_monotonic * loss_mono

        if teacher is not None:
            # U3: clean Point teacher, corrupted/current student; gallery targets stay clean.
            with torch.no_grad():
                teacher_outputs = teacher(
                    clean_ref_images.to(teacher_device),
                    clean_text_ids.to(teacher_device),
                    clean_text_mask.to(teacher_device),
                    target_images.to(teacher_device),
                    compute_uncertainty=False,
                )
            loss_kd = clean_teacher_kl(
                outputs['mu_q'], outputs['mu_c'],
                teacher_outputs['mu_q'].to(device), teacher_outputs['mu_c'].to(device), args.kd_temperature,
            )
            loss = loss + args.lambda_kd * loss_kd

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), max_norm=1.0)
        optimizer.step()

        values = {
            'loss': loss.item(), 'loss_hc': loss_dict['loss_hc'].item(),
            'loss_fc': loss_dict['loss_fc'].item(), 'loss_cord': loss_dict['loss_cord'].item(),
            'loss_mono': loss_mono.item(), 'loss_kd': loss_kd.item(),
            'drop_image': dropout_stats['image_fraction'], 'drop_text': dropout_stats['text_fraction'],
        }
        for key, value in values.items():
            totals[key] += value
        pbar.set_postfix(loss=f"{values['loss']:.4f}", mono=f"{values['loss_mono']:.4f}", kd=f"{values['loss_kd']:.4f}")

        if args.use_wandb and step % args.log_interval == 0:
            wandb.log({
                'train/loss': values['loss'], 'train/loss_hc': values['loss_hc'],
                'train/loss_fc': values['loss_fc'], 'train/loss_cord': values['loss_cord'],
                'train/loss_mono': values['loss_mono'], 'train/loss_kd': values['loss_kd'],
                'train/drop_image_fraction': values['drop_image'], 'train/drop_text_fraction': values['drop_text'],
                'train/step': epoch * len(dataloader) + step,
                'train/lr_head': optimizer.param_groups[1]['lr'], 'train/lr_backbone': optimizer.param_groups[0]['lr'],
            })

    num_batches = len(dataloader)
    return {key: value / num_batches for key, value in totals.items()}


def build_point_teacher(args, device: torch.device):
    """Load a frozen Point model for U3 without adding it to the optimizer."""
    if not args.teacher_checkpoint or not os.path.isfile(args.teacher_checkpoint):
        raise ValueError('--lambda_kd requires an existing --teacher_checkpoint')
    if args.teacher_device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('--teacher_device cuda requested but CUDA is unavailable')
    teacher_device = torch.device(args.teacher_device)
    teacher = HUGModel(
        num_queries=args.num_queries, hidden_dim=args.hidden_dim,
        blip_model_name=args.blip_model, freeze_vision_encoder=True,
        text_feature_mode='query_tokens', uncertainty_is_variance=True,
        backbone_device=str(teacher_device),
    ).to(teacher_device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=teacher_device)
    teacher.load_state_dict(checkpoint['model_state_dict'], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f'Loaded frozen Point teacher from {args.teacher_checkpoint} on {teacher_device}')
    return teacher, teacher_device


def _rng_state():
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_checkpoint(model, optimizer, criterion, scheduler, epoch, args,
                    filename='checkpoint.pth', best_score=float('-inf'),
                    include_training_state=True):
    """Save a model-only evaluation checkpoint or resumable training state."""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'best_score': best_score,
    }
    if include_training_state:
        checkpoint.update({
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rng_state': _rng_state(),
            'resumable': True,
        })
    temporary_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function"""
    args = parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError('--init_checkpoint and --resume_checkpoint are mutually exclusive')
    if args.save_interval < 0:
        raise ValueError('--save_interval must be >= 0')
    set_seed(args.seed)
    robust_enabled = args.lambda_monotonic > 0 or args.modality_dropout_prob > 0 or args.lambda_kd > 0
    if robust_enabled and args.recipe != 'paper':
        raise ValueError('U1/U2/U3 require --recipe paper; legacy and point baselines stay unchanged.')
    if args.lambda_monotonic > 0 and not (0 <= args.monotonic_low_severity < args.monotonic_high_severity <= 1):
        raise ValueError('Require 0 <= --monotonic_low_severity < --monotonic_high_severity <= 1')
    if args.lambda_kd > 0 and args.freeze_backbone:
        raise ValueError('U3 KD needs a trainable mean encoder; remove --freeze_backbone.')

    # Initialize wandb
    if args.use_wandb and not HAS_WANDB:
        raise RuntimeError("--use_wandb was set but wandb is not installed")
    if args.use_wandb:
        wandb.init(project='hug-cir', config=vars(args))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(args, split='train')
    print(f"Training samples: {len(train_loader.dataset)}")

    val_loaders = None
    if args.eval_every > 0:
        from eval import create_retrieval_dataloaders, evaluate_retrieval
        val_loaders = create_retrieval_dataloaders(args, split='val')
        print(f"Validation queries: {len(val_loaders[0].dataset)}")

    # Create model
    print("Creating model...")
    model = HUGModel(
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        blip_model_name=args.blip_model,
        freeze_vision_encoder=True,
        text_feature_mode='legacy_tokens' if args.recipe == 'legacy' else 'query_tokens',
        uncertainty_is_variance=args.recipe != 'legacy'
    ).to(device)

    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(initial['model_state_dict'], strict=True)
        print(f"Initialized model weights from {args.init_checkpoint}")

    if args.freeze_backbone:
        for param in model.blip_backbone.parameters():
            param.requires_grad = False
        model.query_tokens.requires_grad = False
        print("Frozen BLIP/Q-Former backbone and query tokens")

    teacher, teacher_device = build_point_teacher(args, device) if args.lambda_kd > 0 else (None, None)
    if robust_enabled:
        print(
            'RC-HUG objectives: '
            f'U1(lambda={args.lambda_monotonic:g}), '
            f'U2(prob={args.modality_dropout_prob:g}), '
            f'U3(lambda={args.lambda_kd:g})'
        )

    # Create loss function
    criterion = HUGLoss(
        lambda_fc=args.lambda_fc,
        lambda_cord=args.lambda_cord,
        recipe=args.recipe,
        uncertainty_is_variance=args.recipe != 'legacy'
    ).to(device)
    if args.recipe == 'point':
        print(
            "Loss recipe: point (symmetric mean-only InfoNCE); "
            "Loss FC and Loss Cord are intentionally disabled."
        )
    else:
        print(f"Loss recipe: {args.recipe} (HC + {args.lambda_fc}*FC + {args.lambda_cord}*Cord)")

    backbone_params = []
    head_params = []
    loss_params = []

    # 1. Phân loại tham số từ MODEL
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'blip_backbone' in name:
            backbone_params.append(param)
        else:
            # Các head mới như Uncertainty Estimators, Dynamic Weighting...
            head_params.append(param)

    # Learnable sigmoid scale/bias parameters belong to the criterion.
    loss_params = [p for p in criterion.parameters() if p.requires_grad]

    # 3. Khởi tạo Optimizer
    optimizer = AdamW(
        [
            {'params': backbone_params, 'lr': args.lr},      # LR cho backbone (hoặc args.lr * 0.1 nếu cần)
            {'params': head_params, 'lr': args.lr},                # LR gốc cho các head
            {'params': loss_params, 'lr': args.lr * args.loss_lr_multiplier, 'weight_decay': 0.0}           # LR x100 cho a và b
        ],
        betas=(0.9, 0.999),
        eps=1e-7,
        weight_decay=args.weight_decay
    )
    # =========================================================================

    # Create learning rate scheduler
    # CosineAnnealingLR sẽ tự động áp dụng decay cho TỪNG parameter group
    if args.warmup_epochs > 0:
        if args.warmup_epochs >= args.num_epochs:
            raise ValueError("warmup_epochs must be smaller than num_epochs")
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], [args.warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # Training loop
    best_score = float("-inf")
    start_epoch = 1
    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise ValueError(f'Resume checkpoint not found: {args.resume_checkpoint}')
        resume = torch.load(args.resume_checkpoint, map_location=device)
        required = {
            'model_state_dict', 'optimizer_state_dict', 'criterion_state_dict',
            'scheduler_state_dict', 'rng_state', 'epoch', 'resumable',
        }
        missing = sorted(required.difference(resume))
        if missing:
            raise ValueError(
                f'Checkpoint is not safely resumable ({", ".join(missing)} missing): '
                f'{args.resume_checkpoint}'
            )
        previous_args = resume.get('args', {})
        for name in ('dataset', 'category', 'batch_size', 'num_epochs', 'lr',
                     'weight_decay', 'warmup_epochs', 'num_queries', 'hidden_dim',
                     'blip_model', 'recipe', 'lambda_fc', 'lambda_cord',
                     'loss_lr_multiplier', 'freeze_backbone', 'seed'):
            if name in previous_args and previous_args[name] != getattr(args, name):
                raise ValueError(
                    f'Resume argument mismatch for {name}: '
                    f'{previous_args[name]!r} != {getattr(args, name)!r}'
                )
        model.load_state_dict(resume['model_state_dict'], strict=True)
        criterion.load_state_dict(resume['criterion_state_dict'], strict=True)
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        _restore_rng_state(resume['rng_state'])
        start_epoch = int(resume['epoch']) + 1
        best_score = float(resume.get('best_score', float('-inf')))
        print(f'Resumed training from {args.resume_checkpoint} at epoch {start_epoch}')
        if start_epoch > args.num_epochs:
            raise ValueError(
                f'Resume checkpoint already reached epoch {resume["epoch"]}; '
                f'num_epochs={args.num_epochs}'
            )
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs + 1):
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args, teacher, teacher_device
        )

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Loss HC: {train_metrics['loss_hc']:.4f}")
        print(f"  Loss FC: {train_metrics['loss_fc']:.4f}")
        print(f"  Loss Cord: {train_metrics['loss_cord']:.4f}")
        print(f"  Loss Mono: {train_metrics['loss_mono']:.4f}")
        print(f"  Loss KD: {train_metrics['loss_kd']:.4f}")

        # Log epoch metrics
        if args.use_wandb:
            wandb.log({
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_loss_hc': train_metrics['loss_hc'],
                'train/epoch_loss_fc': train_metrics['loss_fc'],
                'train/epoch_loss_cord': train_metrics['loss_cord'],
                'train/epoch_loss_mono': train_metrics['loss_mono'],
                'train/epoch_loss_kd': train_metrics['loss_kd'],
                'train/epoch': epoch
            })

        if val_loaders is not None and epoch % args.eval_every == 0:
            query_loader, gallery_loader, gallery_id_to_idx = val_loaders
            val_metrics = evaluate_retrieval(
                model, query_loader, gallery_loader, gallery_id_to_idx,
                device, distance_mode="mean" if args.recipe == "point" else "probabilistic"
            )
            val_score = 0.5 * (val_metrics['recall@10'] + val_metrics['recall@50'])
            print(
                f"  Val R@10: {val_metrics['recall@10']:.2f} | "
                f"R@50: {val_metrics['recall@50']:.2f} | Avg: {val_score:.2f}"
            )
            if args.use_wandb:
                wandb.log({
                    'val/recall@10': val_metrics['recall@10'],
                    'val/recall@50': val_metrics['recall@50'],
                    'val/avg_recall': val_score,
                    'val/epoch': epoch,
                })
            if val_score > best_score:
                best_score = val_score
                save_checkpoint(
                    model, optimizer, criterion, scheduler, epoch, args,
                    'checkpoint_best.pth', best_score, include_training_state=False,
                )
                print(f"  New best validation checkpoint (Avg={best_score:.2f})")
            model.train()

        # Update learning rate
        scheduler.step()

        # Keep one resumable checkpoint instead of one multi-GB file per epoch.
        save_checkpoint(
            model, optimizer, criterion, scheduler, epoch, args,
            'checkpoint_last.pth', best_score, include_training_state=True,
        )
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, criterion, scheduler, epoch, args,
                f'checkpoint_epoch_{epoch}.pth', best_score, include_training_state=True,
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, criterion, scheduler, args.num_epochs, args,
        'checkpoint_final.pth', best_score, include_training_state=False,
    )

    print("Training completed!")

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
