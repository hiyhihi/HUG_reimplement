import torch

from modules.losses import HolisticContrastiveLoss


def _inputs():
    torch.manual_seed(0)
    mu_q = torch.randn(3, 2, 4)
    mu_c = torch.randn(3, 2, 4)
    sigma_q = torch.full_like(mu_q, 0.1)
    sigma_c = torch.full_like(mu_c, 0.2)
    return mu_q, sigma_q, mu_c, sigma_c


def test_hc_mean_distance_ignores_variance():
    mu_q, sigma_q, mu_c, sigma_c = _inputs()
    loss = HolisticContrastiveLoss(
        1.0, 0.0, symmetric=True, uncertainty_is_variance=True,
        include_uncertainty=False,
    )
    baseline = loss(mu_q, sigma_q, mu_c, sigma_c)
    changed = loss(mu_q, sigma_q * 100, mu_c, sigma_c * 100)
    assert torch.allclose(baseline, changed)


def test_hc_average_negatives_changes_only_reduction():
    mu_q, sigma_q, mu_c, sigma_c = _inputs()
    summed = HolisticContrastiveLoss(1.0, 0.0, symmetric=True, uncertainty_is_variance=True)
    averaged = HolisticContrastiveLoss(
        1.0, 0.0, symmetric=True, uncertainty_is_variance=True,
        average_negatives=True,
    )
    averaged.load_state_dict(summed.state_dict())
    distances = averaged.pairwise_distances(mu_q, sigma_q, mu_c, sigma_c)
    mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
    positive = torch.nn.functional.softplus(
        averaged.a * distances.diagonal() + averaged.b
    ).mean()
    negative = torch.nn.functional.softplus(-averaged.a * distances - averaged.b)
    expected = positive + 2 * negative[mask].mean()
    actual = averaged(mu_q, sigma_q, mu_c, sigma_c)
    assert torch.allclose(actual, expected)
    assert actual < summed(mu_q, sigma_q, mu_c, sigma_c)
