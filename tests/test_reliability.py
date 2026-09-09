import torch
from PIL import Image

from modules.fiqc import FIQC_SCHEMA_VERSION, FIQCCorruptionGenerator, audit_text_corruptions
from modules.reliability import ReliabilityHead, binary_auprc, reliability_metrics


def test_fiqc_severity_zero_is_identity():
    generator = FIQCCorruptionGenerator(seed=42)
    text, text_meta = generator.text("a long black dress", "word_deletion", 0, 3)
    image = Image.new("RGB", (32, 24), (255, 255, 255))
    corrupted, image_meta = generator.image(image, "occlusion", 0, 3)
    assert text == "a long black dress"
    assert text_meta["changed"] is False
    assert corrupted.tobytes() == image.tobytes()
    assert image_meta["changed"] is False


def test_fiqc_is_deterministic_and_word_deletion_is_nested():
    text = "make the dress longer darker and more formal"
    left = FIQCCorruptionGenerator(seed=7)
    right = FIQCCorruptionGenerator(seed=7)
    output_left, meta_left = left.text(text, "word_deletion", 3, 11)
    output_right, meta_right = right.text(text, "word_deletion", 3, 11)
    _, mild = left.text(text, "word_deletion", 1, 11)
    assert output_left == output_right
    assert meta_left == meta_right
    assert set(mild["removed_indices"]).issubset(meta_left["removed_indices"])


def test_lexical_corruptions_change_fashion_caption():
    generator = FIQCCorruptionGenerator(seed=42)
    synonym, synonym_meta = generator.text(
        "a long black dress with striped sleeves", "synonym_replacement", 2, 0,
    )
    contradiction, contradiction_meta = generator.text(
        "a long black dress with striped sleeves", "semantic_contradiction", 2, 0,
    )
    assert synonym != "a long black dress with striped sleeves"
    assert contradiction != "a long black dress with striped sleeves"
    assert synonym_meta["changed"] and contradiction_meta["changed"]


def test_occlusion_is_deterministic_and_increases_area():
    image = Image.new("RGB", (40, 30), (255, 255, 255))
    generator = FIQCCorruptionGenerator(seed=123)
    mild, mild_meta = generator.image(image, "occlusion", 1, 9)
    severe, severe_meta = generator.image(image, "occlusion", 4, 9)
    repeated, repeated_meta = generator.image(image, "occlusion", 4, 9)
    assert severe.tobytes() == repeated.tobytes()
    assert severe_meta == repeated_meta
    assert mild_meta["area_ratio"] < severe_meta["area_ratio"]
    assert mild.tobytes() != image.tobytes()


def test_reliability_head_shape_and_gradient():
    head = ReliabilityHead(hidden_dim=8, bottleneck_dim=4, dropout=0.0)
    features = torch.randn(5, 3, 8, requires_grad=True)
    logits = head(features)
    assert logits.shape == (5,)
    logits.sum().backward()
    assert features.grad is not None


def test_reliability_metrics_for_perfect_ordering():
    metrics = reliability_metrics([0, 0, 1, 1], [0.05, 0.10, 0.90, 0.95], num_bins=4)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["risk_coverage"][0]["risk"] == 0.0


def test_schema_v2_synonym_audit_covers_common_fashion_constructions():
    assert FIQC_SCHEMA_VERSION == 2
    captions = [
        "is black with straps",
        "has a sleeveless floral print",
        "is a lighter solid color with a belt",
    ]
    rows = audit_text_corruptions(captions, seed=42, severities=[1, 4])
    synonym = [row for row in rows if row["corruption_type"] == "synonym_replacement"]
    assert all(row["changed_rate"] == 1.0 for row in synonym)


def test_auprc_ties_are_threshold_grouped():
    assert binary_auprc([0, 1], [0.5, 0.5]) == 0.5
