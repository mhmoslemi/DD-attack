"""
Robust aggregation of a batch of representations.

This module implements the mean estimators that replace `mean(real_reps)` in
the distribution-matching loss:

  * mean_calibration  -> Algorithm 4 + 5 (RDM-DC, the paper's proposed defense)
  * truncated_mean    -> drop top-k reps by L2 distance to the mean
  * trimmed_mean      -> coordinate-wise drop k/2 high + k/2 low
  * coordinate_median -> coordinate-wise median
  * (mean / none)     -> plain mean (vanilla DM, no defense)

All functions take `reps` of shape [N, d] and return a vector of shape [d].

Mean calibration (Algorithm 4):
  1. mu = mean(reps); Sigma = cov(reps)            (cov along feature dim d)
  2. v = top eigenvector of Sigma  (power method, Algorithm 5)
  3. score_i = |<reps_i - mu, v>|
  4. drop the `drop_count` = floor(3 * eps * N) reps with the largest scores
  5. return the mean of the remaining reps

We expose two equivalent ways to get the top eigenvector: a matrix-free power
iteration over the data (default; never forms the d x d covariance, which is
2048x2048 for the CIFAR ConvNet) and `power_method`, the literal Algorithm 5 on
an explicit covariance matrix, kept for faithfulness / testing.
"""
import torch


def power_method(sigma, num_iters, generator=None):
    """Algorithm 5, literal. sigma: [d, d] covariance. Returns unit vector [d]."""
    d = sigma.shape[0]
    v = torch.randn(d, generator=generator, device=sigma.device, dtype=sigma.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(num_iters):
        v = sigma @ v
        v = v / (v.norm() + 1e-12)
    return v


def top_eigenvector_via_data(centered, num_iters, generator=None):
    """Matrix-free top eigenvector of the covariance of `centered` (N x d,
    already mean-subtracted). Avoids materialising the d x d covariance:
        Sigma v = centered^T (centered v) / (N - 1).
    """
    n, d = centered.shape
    denom = max(n - 1, 1)
    v = torch.randn(d, generator=generator, device=centered.device,
                    dtype=centered.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(num_iters):
        v = centered.t() @ (centered @ v) / denom
        v = v / (v.norm() + 1e-12)
    return v


def mean_calibration(reps, drop_count, power_iters=10, generator=None):
    n = reps.shape[0]
    mu = reps.mean(dim=0)
    if drop_count <= 0 or drop_count >= n:
        return mu
    centered = reps - mu
    v = top_eigenvector_via_data(centered, power_iters, generator=generator)
    scores = (centered @ v).abs()                       # |<r_i - mu, v>|
    keep = torch.topk(scores, n - drop_count, largest=False).indices
    return reps[keep].mean(dim=0)


def truncated_mean(reps, drop_count):
    n = reps.shape[0]
    mu = reps.mean(dim=0)
    if drop_count <= 0 or drop_count >= n:
        return mu
    dist = (reps - mu).norm(dim=1)                      # spherical distance
    keep = torch.topk(dist, n - drop_count, largest=False).indices
    return reps[keep].mean(dim=0)


def trimmed_mean(reps, drop_count):
    """Coordinate-wise: drop k/2 largest and k/2 smallest per dimension."""
    n = reps.shape[0]
    half = drop_count // 2
    if half <= 0 or 2 * half >= n:
        return reps.mean(dim=0)
    sorted_reps, _ = torch.sort(reps, dim=0)            # ascending per column
    trimmed = sorted_reps[half:n - half]
    return trimmed.mean(dim=0)


def coordinate_median(reps):
    return reps.median(dim=0).values


_SYNONYMS = {
    'none': 'mean', 'mean': 'mean', 'dm': 'mean',
    'rdmdc': 'calibration', 'rdm-dc': 'calibration',
    'calibration': 'calibration', 'calibrate': 'calibration',
    'truncated': 'truncated', 'truncated_mean': 'truncated',
    'trimmed': 'trimmed', 'trimmed_mean': 'trimmed', 'trim': 'trimmed',
    'median': 'median', 'coordinate_median': 'median',
}


def aggregate(reps, method, drop_count=0, power_iters=10, generator=None):
    """Dispatch to the chosen estimator. `reps`: [N, d] -> [d]."""
    key = _SYNONYMS.get(method.lower())
    if key is None:
        raise ValueError('unknown aggregation method: %s' % method)
    if key == 'mean':
        return reps.mean(dim=0)
    if key == 'calibration':
        return mean_calibration(reps, drop_count, power_iters, generator)
    if key == 'truncated':
        return truncated_mean(reps, drop_count)
    if key == 'trimmed':
        return trimmed_mean(reps, drop_count)
    if key == 'median':
        return coordinate_median(reps)
    raise AssertionError  # unreachable
