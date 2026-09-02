# periodicwave/observables.py
import jax
import jax.numpy as jnp
import chex
from periodicwave import constants
from typing import Tuple, Callable
from periodicwave import networks
from periodicwave.utils import utils

def make_local_spin_squared(
    f: networks.FermiNetLike, 
    nspins: Tuple[int, int],
    complex_output: bool = True,
) -> Callable:
    """Creates a function for the local S^2 operator.
    
    Args:
      f: Callable which evaluates the wavefunction as a (phase, log magnitude)
        tuple, i.e. the return value of network_blocks.logdet_matmul.
      nspins: Tuple of (number of alpha electrons, number of beta electrons).
      complex_output: whether f was built with complex_output. This selects how
        the first return value of f is interpreted; see _log_psi below.
      
    Returns:
      Callable which evaluates the local S^2 for a single walker configuration.
    """

    # network_blocks.logdet_matmul returns
    #     phase_out = jnp.angle(result)   if complex   (an ANGLE, in radians)
    #     phase_out = jnp.sign(result)    if real      (+1 / -1)
    # despite the comment there claiming a unit-norm complex number. Forming
    # ratios as (phase' / phase) is therefore wrong in the complex case: it
    # divides two angles. Work in the log domain instead, where both conventions
    # reduce to the same expression.
    def _log_psi(phase, logabs):
        if complex_output:
            return logabs + 1j * phase
        return logabs + 1j * jnp.pi * (phase < 0)

    # <S^2> = Sz(Sz + 1) + N_beta - sum_{i in alpha, j in beta} <P_ij>, where
    # P_ij is the spin transposition of electrons i and j. We evaluate <P_ij>
    # directly as the ratio Psi(..., sigma_j, ..., sigma_i, ...) / Psi(...),
    # i.e. by swapping SPINS at fixed positions.
    #
    # NOTE: this previously swapped entries of data.positions, which is a FLAT
    # array of shape (nelectron * ndim,). Indexing it with an electron index
    # picks a single coordinate, not an electron, so the estimator was swapping
    # e.g. x_0 with y_1 rather than exchanging two electrons. Swapping spins
    # avoids that class of bug entirely and needs no knowledge of ndim.
    #
    # NOTE: alpha/beta membership is read from data.spins per walker rather than
    # assumed to be the first n_up indices, so the estimator stays correct once
    # the sampler applies sector-preserving spin swaps (see mcmc.py). Only the
    # SHAPES (n_up, n_down) are taken from nspins; those are fixed because the
    # spin moves conserve Sz.
    n_up, n_down = nspins

    def _s2_over_f(params: networks.ParamTree, data: networks.FermiNetData) -> jax.Array:
        spins = data.spins

        # Sz and N_beta from the actual configuration.
        Sz = 0.5 * jnp.sum(spins)
        n_beta = jnp.sum(spins < 0)
        constant_term = Sz * (Sz + 1) + n_beta

        # Indices of the up and down electrons of THIS walker. Descending sort
        # of the spins puts the up electrons first; counts are static.
        order = jnp.argsort(-spins)
        alpha_indices = order[:n_up]
        beta_indices = order[n_up:n_up + n_down]

        # All (alpha, beta) pairs.
        A_flat = jnp.repeat(alpha_indices, n_down)
        B_flat = jnp.tile(beta_indices, n_up)

        def swap_single_pair(s, idx_a, idx_b):
            """Exchange the spins of electrons idx_a and idx_b."""
            return s.at[idx_a].set(s[idx_b]).at[idx_b].set(s[idx_a])

        batch_swap_fn = jax.vmap(swap_single_pair, in_axes=(None, 0, 0))

        # Network output for the original configuration.
        orig_phase, orig_logabs = f(
            params, data.positions, spins, data.atoms, data.charges
        )

        # The (N_alpha * N_beta) spin-swapped configurations.
        swapped_spins = batch_swap_fn(spins, A_flat, B_flat)

        # Vectorize the network over the swapped spin configurations.
        batched_f = jax.vmap(f, in_axes=(None, None, 0, None, None))
        swap_phases, swap_logabs = batched_f(
            params, data.positions, swapped_spins, data.atoms, data.charges
        )

        # Wavefunction ratio for each swapped state, in the log domain.
        ratios = jnp.exp(
            _log_psi(swap_phases, swap_logabs) - _log_psi(orig_phase, orig_logabs)
        )

        return constant_term - jnp.sum(ratios)

    return _s2_over_f

@chex.dataclass
class S2Data:
    """Data returned by the S^2 evaluation."""
    s2_mean: jax.Array
    s2_variance: jax.Array
    local_s2: jax.Array

def make_s2_evaluation(local_s2_fn: Callable) -> Callable:
    """Creates a batched function to evaluate the expectation value of S^2.
    
    Args:
      local_s2_fn: Callable that evaluates the local S^2 for a single configuration.
                   
    Returns:
      Callable with signature (params, data) -> S2Data
    """
    
    # 1. Vectorize the local S^2 function over the batch of MCMC walkers
    batch_local_s2 = jax.vmap(
        local_s2_fn,
        in_axes=(
            None, 
            networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
        ),
        out_axes=0
    )

    # 2. Compile the evaluation step
    @jax.jit
    def evaluate_s2(
        params: networks.ParamTree,
        data: networks.FermiNetData,
    ) -> S2Data:
        """Evaluates <S^2> and its variance for a batch of MCMC configurations."""
        
        # Compute local S^2 for every walker
        s2_local = batch_local_s2(params, data)
        
        # Compute mean <S^2> across batch and across parallel devices
        s2_mean = constants.pmean(jnp.mean(s2_local))
        
        # Compute the variance of the estimator
        s2_diff = s2_local - s2_mean
        s2_variance = constants.pmean(jnp.mean(s2_diff * jnp.conjugate(s2_diff)))
        
        return S2Data(
            s2_mean=s2_mean.real, 
            s2_variance=s2_variance.real,
            local_s2=s2_local
        )

    return evaluate_s2
