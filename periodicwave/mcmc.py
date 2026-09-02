# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier Massachusetts Institute of Technology, MA, USA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Simplified MCMC move width update routine (called directly in train.py)
# - Added sector-preserving spin-swap Metropolis moves (Sz conserved), with
#   accept/reject decided separately from the coordinate moves.
# - Fixed stale mh_accept call signature in mh_block_update.

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.

Conventions assumed here:
  data.positions : float array, shape [batch, nelec * ndim]  (flattened)
  data.spins     : array,       shape [batch, nelec], sign encodes up/down
                   (any encoding works as long as up > 0 and down < 0)
"""

import chex
from periodicwave import constants
from periodicwave import networks
import jax
from jax import lax
from jax import numpy as jnp

_NEG = -1e30  # stand-in for -inf in categorical logits (avoids inf-inf NaNs)


def mh_accept(x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  spins_new = jnp.where(cond[..., None], spins2, spins1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, spins_new, key, lp_new, num_accepts


def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0

):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks, ndim  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  spins1 = data.spins

  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  spins2 = spins1
  lp_2 = 2.0 * f(
      params, x2, spins2, data.atoms, data.charges
  )  # log prob of proposal
  ratio = lp_2 - lp_1

  x_new, spins_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts)

  new_data = networks.FermiNetData(
      **(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Spins are untouched by this move; use mh_spin_update for spin moves.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  x2 = x1.at[:, ii].add(
      stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
  x2 = jnp.reshape(x2, [batch_size, -1])
  if pad > 0:
    x2 = x2[..., :-pad*ndim]
  # log prob of proposal
  lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
  ratio = lp_2 - lp_1

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  # NOTE: spins are passed through unchanged (previously this call used the
  # old 7-argument mh_accept signature and was broken for blocks > 1).
  x_new, spins_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, data.spins, data.spins, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(
      **(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts


# -----------------------------------------------------------------------------
# Sector-preserving spin updates
# -----------------------------------------------------------------------------
#
# The move exchanges the spin labels of one spin-up and one spin-down electron.
# Total Sz is conserved by construction, so the chain stays in its sector.
#
# Detailed balance: an elementary move picks a pair uniformly among the
# N_up * N_dn opposite-spin pairs. After the swap the same pair is again an
# opposite-spin pair and N_up, N_dn are unchanged, so the reverse proposal has
# identical probability. The proposal kernel Q is therefore symmetric, and so is
# the m-fold composition Q^m and any fixed mixture over m. Acceptance is the
# plain Metropolis ratio |psi'/psi|^2.
#
# Equivalence worth remembering: because psi is antisymmetric in the generalized
# coordinates (r_i, sigma_i), swapping the spins of electrons i and j gives the
# same |psi|^2 as swapping their positions. This is the basis of the sanity
# check in the docstring of make_mcmc_step.


def _sample_updown_pair(key, spins):
  """Per walker, sample one spin-up index and one spin-down index uniformly."""
  up_logits = jnp.where(spins > 0, 0.0, _NEG)
  dn_logits = jnp.where(spins < 0, 0.0, _NEG)
  key_up, key_dn = jax.random.split(key)
  # categorical draws independent Gumbel noise per row, so walkers are
  # independent even though they share a key.
  i_up = jax.random.categorical(key_up, up_logits, axis=-1)
  i_dn = jax.random.categorical(key_dn, dn_logits, axis=-1)
  return i_up, i_dn


def _apply_spin_swap(spins, i_up, i_dn, active):
  """Swap spins[i_up] <-> spins[i_dn] for walkers where active is True."""
  rows = jnp.arange(spins.shape[0])
  s_up = spins[rows, i_up]
  s_dn = spins[rows, i_dn]
  out = spins.at[rows, i_up].set(jnp.where(active, s_dn, s_up))
  out = out.at[rows, i_dn].set(jnp.where(active, s_up, s_dn))
  return out


def _propose_spin_swaps(key, spins, p_swap, max_swaps):
  """Propose m ~ Poisson(p_swap * nelec / 2) opposite-spin swaps per walker.

  Returns:
    (spins2, proposed, key) where proposed[b] is True iff walker b actually
    received at least one swap.
  """
  batch, nelec = spins.shape
  valid = jnp.any(spins > 0, axis=-1) & jnp.any(spins < 0, axis=-1)

  key, subkey = jax.random.split(key)
  lam = p_swap * nelec / 2.0
  m = jax.random.poisson(subkey, lam, shape=(batch,))
  m = jnp.minimum(m, max_swaps)

  def body(k, carry):
    s, key_ = carry
    key_, subkey_ = jax.random.split(key_)
    # Resample indices from the *current* spin configuration so that each
    # elementary step is an independent symmetric move.
    i_up, i_dn = _sample_updown_pair(subkey_, s)
    active = valid & (k < m)
    return _apply_spin_swap(s, i_up, i_dn, active), key_

  spins2, key = lax.fori_loop(0, max_swaps, body, (spins, key))
  proposed = valid & (m > 0)
  return spins2, proposed, key


def mh_spin_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    num_proposals,
    p_swap=0.03,
    max_swaps=4,
):
  """One sector-preserving spin-swap Metropolis step (positions unchanged).

  Args:
    params: Wavefunction parameters.
    f: Callable with LogFermiNetLike signature returning log|psi|.
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: 2*log|psi| evaluated at the current configuration.
    num_accepts: running total of accepted spin moves.
    num_proposals: running total of proposed (non-trivial) spin moves.
    p_swap: sets the mean number of swapped pairs, lambda = p_swap * nelec / 2.
    max_swaps: hard cap on pairs swapped in one proposal (truncates the
      Poisson; keep it a few times larger than lambda).

  Returns:
    (data, key, lp, num_accepts, num_proposals).
  """
  x = data.positions
  spins1 = data.spins

  spins2, proposed, key = _propose_spin_swaps(key, spins1, p_swap, max_swaps)
  lp_2 = 2.0 * f(params, x, spins2, data.atoms, data.charges)

  # Walkers that drew m = 0 have spins2 == spins1 and would otherwise be counted
  # as trivially accepted; force rejection (a no-op) and exclude them from the
  # acceptance statistics.
  ratio = jnp.where(proposed, lp_2 - lp_1, -jnp.inf)

  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  spins_new = jnp.where(cond[..., None], spins2, spins1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  num_proposals += jnp.sum(proposed)

  new_data = networks.FermiNetData(**(dict(data) | {'spins': spins_new}))
  return new_data, key, lp_new, num_accepts, num_proposals


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=3,
                   blocks=1,
                   spin_steps=0,
                   p_swap=0.03,
                   max_swaps=4,
                   return_spin_pmove=False,
                   ):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x, spins, atoms, charges), which
      evaluates the log of the wavefunction (square root of the log probability
      distribution) at x given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of coordinate MCMC moves to attempt in a single call.
    ndim: Dimensionality of the system. Required only for block updates.
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.
    spin_steps: Number of sector-preserving spin-swap proposals per call. 0
      disables spin moves entirely and reproduces the previous behaviour
      bit-for-bit.
    p_swap: mean pairs swapped per proposal is p_swap * nelec / 2.
    max_swaps: cap on pairs swapped within one proposal.
    return_spin_pmove: if True, mcmc_step returns (data, pmove, pmove_spin)
      instead of (data, pmove).

  Returns:
    Callable which performs the set of MCMC steps.

  Sanity checks before trusting this (do these once, offline):
    1. The network must actually consume data.spins. Evaluate
         batch_network(params, x, spins, ...) and
         batch_network(params, x, spins_with_one_pair_swapped, ...)
       and check log|psi| changes. If it does not, spins are only being used to
       set a static up/down block structure and the move is a no-op.
    2. Antisymmetry check: swapping spins of electrons i, j must give the same
       |psi| as swapping their positions r_i, r_j. If this fails the ansatz is
       not antisymmetric in the generalized coordinates and the acceptance ratio
       above is not the right one.
    3. Physics check: converged energies with spin_steps=0 and spin_steps>0 must
       agree within error bars. They sample the same |psi|^2; only the mixing
       differs. A discrepancy means the spin_steps=0 run had an ergodicity bias,
       which is exactly the failure mode this move is meant to remove.
  """
  inner_fun = mh_block_update if blocks > 1 else mh_update
  use_spin = spin_steps > 0 and p_swap > 0.0

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the coordinate move proposal.

    Returns:
      (data, pmove) or (data, pmove, pmove_spin) if return_spin_pmove is set.
    """
    logprob = 2.0 * batch_network(
        params, data.positions, data.spins, data.atoms, data.charges
    )

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          ndim=ndim,
          blocks=blocks,
          i=i
          )

    nsteps = steps * blocks
    new_data, key_out, logprob, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)

    if use_spin:
      def spin_step_fn(i, x):
        del i
        return mh_spin_update(
            params,
            batch_network,
            *x,
            p_swap=p_swap,
            max_swaps=max_swaps,
        )

      new_data, key_out, logprob, spin_accepts, spin_proposals = lax.fori_loop(
          0, spin_steps, spin_step_fn, (new_data, key_out, logprob, 0.0, 0.0)
      )
      # Ratio is formed per device then averaged, matching how pmove is handled.
      pmove_spin = jnp.sum(spin_accepts) / jnp.maximum(
          jnp.sum(spin_proposals), 1.0)
      pmove_spin = constants.pmean(pmove_spin)
    else:
      pmove_spin = jnp.zeros_like(pmove)

    if return_spin_pmove:
      return new_data, pmove, pmove_spin
    return new_data, pmove

  return mcmc_step
