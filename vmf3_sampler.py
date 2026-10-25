# Copyright © MERL 2026
# Description: Samples and estimates entropy for 3D von Mises-Fisher action distributions.
import math
from typing import Optional, Sequence

import torch
from torch import Tensor


@torch.no_grad()
def sample_vmf3(
    mu: Optional[Tensor] = None,
    kappa: Optional[Tensor] = None,
    *,
    eta: Optional[Tensor] = None,
    sample_shape: Sequence[int] = (),
    eps: float = 1e-7,
) -> Tensor:
    """
    Sample exactly from a von Mises-Fisher distribution on S^2.

    The distribution is

        p(x | mu, kappa) ∝ exp(kappa * mu^T x),

    where:
        x ∈ R^3 and ||x|| = 1
        mu ∈ R^3 and ||mu|| = 1
        kappa >= 0

    Parameters
    ----------
    mu:
        Mean direction with shape (..., 3).

    kappa:
        Concentration with shape (...) or (..., 1).

    eta:
        Pyro-style natural parameter eta = kappa * mu,
        with shape (..., 3). Supply either eta or (mu, kappa),
        but not both.

    sample_shape:
        Leading sample dimensions. For example:
            sample_shape=(100,)
        generates 100 samples from each batch distribution.

    eps:
        Numerical tolerance.

    Returns
    -------
    Tensor
        Samples with shape:

            sample_shape + batch_shape + (3,)

    Notes
    -----
    This sampler is not reparameterized. It is suitable for PPO and
    likelihood-ratio policy gradients, but gradients do not flow through
    the sampled direction.
    """
    if eta is not None:
        if mu is not None or kappa is not None:
            raise ValueError(
                "Provide either eta or (mu, kappa), not both."
            )

        eta = torch.as_tensor(eta)

        if eta.shape[-1] != 3:
            raise ValueError("eta must have final dimension 3.")

        if not eta.is_floating_point():
            eta = eta.float()

        kappa = torch.linalg.vector_norm(eta, dim=-1)
        mu = eta / kappa.unsqueeze(-1).clamp_min(eps)

        # At kappa = 0, the mean direction is irrelevant.
        default_mu = torch.zeros_like(mu)
        default_mu[..., 2] = 1.0

        mu = torch.where(
            (kappa > eps).unsqueeze(-1),
            mu,
            default_mu,
        )

    else:
        if mu is None or kappa is None:
            raise ValueError(
                "Provide either eta or both mu and kappa."
            )

        mu = torch.as_tensor(mu)

        if mu.shape[-1] != 3:
            raise ValueError("mu must have final dimension 3.")

        if not mu.is_floating_point():
            mu = mu.float()

        kappa = torch.as_tensor(
            kappa,
            dtype=mu.dtype,
            device=mu.device,
        )

        # Allow kappa shape (..., 1).
        if kappa.ndim > 0 and kappa.shape[-1] == 1:
            kappa = kappa.squeeze(-1)

        if torch.any(kappa < 0):
            raise ValueError("kappa must be nonnegative.")

        mu_norm = torch.linalg.vector_norm(
            mu,
            dim=-1,
            keepdim=True,
        )

        if torch.any(mu_norm <= eps):
            raise ValueError("mu must contain nonzero vectors.")

        mu = mu / mu_norm

    # Broadcast mu and kappa to a common batch shape.
    batch_shape = torch.broadcast_shapes(
        mu.shape[:-1],
        kappa.shape,
    )

    mu = mu.expand(batch_shape + (3,))
    kappa = kappa.expand(batch_shape)

    sample_shape = torch.Size(sample_shape)
    output_shape = sample_shape + batch_shape

    mu = mu.expand(sample_shape + batch_shape + (3,))
    kappa = kappa.expand(output_shape)

    # ---------------------------------------------------------------
    # Step 1: sample w = cos(theta).
    #
    # On S^2:
    #
    #     p(w) ∝ exp(kappa * w),  w ∈ [-1, 1].
    #
    # Its inverse CDF is:
    #
    #     w = 1 + log(u + (1-u)e^(-2kappa)) / kappa.
    #
    # We evaluate the logarithm using logaddexp for stability.
    # ---------------------------------------------------------------
    u = torch.rand(
        output_shape,
        dtype=mu.dtype,
        device=mu.device,
    ).clamp(min=eps, max=1.0 - eps)

    log_term = torch.logaddexp(
        torch.log(u),
        torch.log1p(-u) - 2.0 * kappa,
    )

    safe_kappa = kappa.clamp_min(eps)
    concentrated_w = 1.0 + log_term / safe_kappa

    # For kappa -> 0, w is uniform on [-1, 1].
    uniform_w = 2.0 * u - 1.0

    w = torch.where(
        kappa > 1e-5,
        concentrated_w,
        uniform_w,
    ).clamp(-1.0, 1.0)

    # ---------------------------------------------------------------
    # Step 2: sample an azimuth uniformly.
    # This initially produces samples centered around e3 = [0, 0, 1].
    # ---------------------------------------------------------------
    phi = (
        2.0
        * math.pi
        * torch.rand(
            output_shape,
            dtype=mu.dtype,
            device=mu.device,
        )
    )

    radial = torch.sqrt(
        torch.clamp(1.0 - w.square(), min=0.0)
    )

    local_sample = torch.stack(
        (
            radial * torch.cos(phi),
            radial * torch.sin(phi),
            w,
        ),
        dim=-1,
    )

    # ---------------------------------------------------------------
    # Step 3: rotate the north-pole distribution toward mu.
    #
    # A Householder reflection maps e3 to mu:
    #
    #     Hx = x - 2v(v^T x)/(v^T v),
    #     v = e3 - mu.
    #
    # When mu is already e3, no rotation is necessary.
    # ---------------------------------------------------------------
    north = torch.zeros_like(mu)
    north[..., 2] = 1.0

    v = north - mu
    v_norm_sq = v.square().sum(dim=-1, keepdim=True)

    reflected = local_sample - (
        2.0
        * v
        * (local_sample * v).sum(dim=-1, keepdim=True)
        / v_norm_sq.clamp_min(eps)
    )

    sample = torch.where(
        v_norm_sq > eps,
        reflected,
        local_sample,
    )

    # Remove tiny numerical deviations from the unit sphere.
    sample = sample / torch.linalg.vector_norm(
        sample,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    return sample


def entropy_vmf3(
    kappa: Optional[Tensor] = None,
    *,
    eta: Optional[Tensor] = None,
    small_kappa_threshold: float = 1e-3,
) -> Tensor:
    """
    Compute the differential entropy of a von Mises-Fisher distribution
    on S^2.

    Supply either:

        kappa: concentration, shape batch_shape

    or:

        eta: Pyro-style natural parameter eta = kappa * mu,
             shape batch_shape + (3,)

    The entropy is

        H = log(4*pi*sinh(kappa)/kappa)
            - kappa * (coth(kappa) - 1/kappa).

    Equivalently,

        H = -log C_3(kappa) - kappa * A_3(kappa).

    Returns
    -------
    Tensor
        Entropy with shape batch_shape.

    Notes
    -----
    The entropy depends only on kappa, not on the mean direction mu.
    At kappa = 0, the distribution is uniform on S^2 and

        H = log(4*pi).
    """
    if eta is not None:
        if kappa is not None:
            raise ValueError("Provide either eta or kappa, not both.")

        eta = torch.as_tensor(eta)

        if eta.shape[-1] != 3:
            raise ValueError("eta must have final dimension 3.")

        if not eta.is_floating_point():
            eta = eta.float()

        kappa = torch.linalg.vector_norm(eta, dim=-1)

    elif kappa is not None:
        kappa = torch.as_tensor(kappa)

        if not kappa.is_floating_point():
            kappa = kappa.float()

        # Accept shape (..., 1) as well as (...).
        if kappa.ndim > 0 and kappa.shape[-1] == 1:
            kappa = kappa.squeeze(-1)

    else:
        raise ValueError("Provide either eta or kappa.")

    if torch.any(kappa < 0):
        raise ValueError("kappa must be nonnegative.")

    kappa_squared = kappa.square()

    # Taylor expansion around kappa = 0:
    #
    # H = log(4*pi)
    #     - kappa^2 / 6
    #     + kappa^4 / 60
    #     - kappa^6 / 567
    #     + O(kappa^8)
    small_entropy = (
        math.log(4.0 * math.pi)
        - kappa_squared / 6.0
        + kappa_squared.square() / 60.0
        - kappa_squared.pow(3) / 567.0
    )

    # Clamp only for evaluation of the regular branch.
    safe_kappa = kappa.clamp_min(small_kappa_threshold)

    # Stable expression:
    #
    # log(sinh(kappa))
    # = kappa + log(1 - exp(-2*kappa)) - log(2)
    #
    # -expm1(-2*kappa) computes 1 - exp(-2*kappa) stably.
    log_sinh = (
        safe_kappa
        + torch.log(-torch.expm1(-2.0 * safe_kappa))
        - math.log(2.0)
    )

    log_inverse_normalizer = (
        math.log(4.0 * math.pi)
        + log_sinh
        - torch.log(safe_kappa)
    )

    mean_resultant_length = (
        1.0 / torch.tanh(safe_kappa)
        - 1.0 / safe_kappa
    )

    regular_entropy = (
        log_inverse_normalizer
        - safe_kappa * mean_resultant_length
    )

    return torch.where(
        kappa < small_kappa_threshold,
        small_entropy,
        regular_entropy,
    )
