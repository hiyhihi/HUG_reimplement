"""CPU tests for U1/U2/U3 tensor utilities; run with `python tests/test_robustness_training.py`."""
import torch

from modules.robustness_training import (
    apply_modality_dropout,
    blur_images,
    monotonic_ranking_loss,
    text_dropout_pair,
    text_dropout_view,
)


def test_text_dropout_view_preserves_special_tokens():
    ids = torch.tensor([[101, 10, 11, 102, 0], [101, 20, 21, 102, 0]])
    masks = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])

    dropped_ids, dropped_masks = text_dropout_view(ids, masks, probability=1.0)

    assert torch.equal(dropped_ids[:, [0, 3, 4]], ids[:, [0, 3, 4]])
    assert torch.equal(dropped_masks[:, [0, 3, 4]], masks[:, [0, 3, 4]])
    assert torch.count_nonzero(dropped_ids[:, 1:3]) == 0
    assert torch.count_nonzero(dropped_masks[:, 1:3]) == 0


def main():
    torch.manual_seed(7)
    images = torch.rand(4, 3, 12, 12)
    ids = torch.tensor([[101, 10, 11, 102, 0], [101, 20, 21, 102, 0]]).repeat(2, 1)
    masks = torch.tensor([[1, 1, 1, 1, 0]]).repeat(4, 1)

    assert torch.equal(blur_images(images, 0.0), images)
    assert not torch.equal(blur_images(images, 0.75), images)
    (low_ids, low_masks), (high_ids, high_masks) = text_dropout_pair(ids, masks, 0.2, 0.8)
    assert torch.all(high_masks <= low_masks)  # High corruption is a superset.
    assert torch.equal(low_ids[:, 0], ids[:, 0]) and torch.equal(high_ids[:, 3], ids[:, 3])
    assert monotonic_ranking_loss(torch.tensor([0.2]), torch.tensor([0.4]), 0.1).item() == 0.0

    _, _, _, stats = apply_modality_dropout(images, ids, masks, 1.0, 1.0)
    assert abs(stats["image_fraction"] + stats["text_fraction"] - 1.0) < 1e-6
    print("robustness_training tests passed")


if __name__ == "__main__":
    main()
