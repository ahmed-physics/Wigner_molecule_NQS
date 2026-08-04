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
    nspins: Tuple[int, int]
) -> Callable:
    """Creates a function for the local S^2 operator.
    
    Args:
      f: Callable which evaluates the wavefunction as a (sign, log magnitude) tuple.
      nspins: Tuple of (number of alpha electrons, number of beta electrons).
      
    Returns:
      Callable which evaluates the local S^2 for a single walker configuration.
    """
    
    # 1. Setup static constants and indices
    n_up, n_down = nspins
    Sz = (n_up - n_down) / 2.0
    constant_term = Sz * (Sz + 1) + n_down
    
    # Generate static swap indices for JAX compilation
    alpha_indices = jnp.arange(n_up)
    beta_indices = jnp.arange(n_up, n_up + n_down)
    A, B = jnp.meshgrid(alpha_indices, beta_indices, indexing='ij')
    A_flat = A.flatten()
    B_flat = B.flatten()

    # 2. Define the vectorized single-swap function
    def swap_single_pair(pos, idx_a, idx_b):
        """Takes a single position array and swaps the rows at idx_a and idx_b."""
        row_a = pos[idx_a]
        row_b = pos[idx_b]
        return pos.at[idx_a].set(row_b).at[idx_b].set(row_a)

    batch_swap_fn = jax.vmap(swap_single_pair, in_axes=(None, 0, 0))

    # 3. Define the actual local operator evaluation
    def _s2_over_f(params: networks.ParamTree, data: networks.FermiNetData) -> jax.Array:
        # Extract network output for the original configuration
        orig_phase, orig_logabs = f(
            params, data.positions, data.spins, data.atoms, data.charges
        )
        
        # Generate the (N_alpha * N_beta) swapped spatial configurations
        swapped_positions = batch_swap_fn(data.positions, A_flat, B_flat)
        
        # Vectorize the network f over the 0th axis of the swapped_positions array
        batched_f = jax.vmap(f, in_axes=(None, 0, None, None, None))
        
        # Evaluate the network on all swapped configurations simultaneously
        swap_phases, swap_logabs = batched_f(
            params, swapped_positions, data.spins, data.atoms, data.charges
        )
        
        # Compute the wave function ratio for each swapped state
        ratios = (swap_phases / orig_phase) * jnp.exp(swap_logabs - orig_logabs)
        
        # Calculate local S^2
        local_s2 = constant_term - jnp.sum(ratios)
        
        return local_s2

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
