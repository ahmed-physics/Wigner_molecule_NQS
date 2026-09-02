# Copyright 2022 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
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
# - Updated simple-ee Jastrow for Coulomb interaction in two dimensional materials
# - Cusp factors moved into a shared helper (_cusp_factors) so that the values
#   used to evaluate the Jastrow and the values used to initialise alpha cannot
#   drift apart.
# - alpha is initialised as 1/interaction_strength rather than 1, so that the
#   depth of the Jastrow at contact, u(0) = -cusp * alpha, stays O(1) per pair.
#   This is the original alpha = 1 rescaled by 1/interaction_strength, and is
#   identical to it for interaction_strength = 1. Initialising alpha = 1 for a
#   large interaction_strength (e.g. ~33 in moire units) makes the Jastrow
#   exponent and its gradient dominate log|Psi| by several orders of magnitude
#   at the start of training.
# - simple_ee_cusp_fun takes |alpha|, so that a sign change of the (otherwise
#   unconstrained) parameter cannot move the pole of alpha^2/(alpha + r) into
#   the range of sampled electron-electron distances.

""" Multiplicative Jastrow factors. """

import enum
from typing import Any, Callable, Iterable, Mapping, Tuple, Union
import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]

class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  SIMPLE_EE_SHORTRANGE = enum.auto()


def _cusp_factors(
    ndim: int = 3,
    interaction_strength: float = 1.0,
) -> Tuple[float, float]:
  """Returns (cusp_parallel, cusp_anti) enforcing the Coulomb cusp conditions.

  The Jastrow enters the wavefunction as Psi = exp(J) * det, with
  J = sum_{i<j} f(r_ij) and d f / d r |_{r=0} = cusp. Cancelling the divergence
  of the pair Coulomb term U/r against the kinetic energy in d dimensions gives
  cusp = U/(d-1) for antiparallel spins, where the determinant is finite at
  coalescence, and cusp = U/(d+1) for parallel spins, where it vanishes.

  Args:
    ndim: spatial dimension of the system.
    interaction_strength: strength U of the Coulomb interaction, in the same
      units as the Hamiltonian.

  Returns:
    Tuple of the parallel-spin and antiparallel-spin cusp factors.
  """
  if ndim == 3:
    return interaction_strength/4, interaction_strength/2
  elif ndim == 2:
    return interaction_strength/3, interaction_strength
  else:
    raise NotImplementedError("jastrow_ee: Factors to satisfy Coulomb cusp conditions implemented only for ndim = 2 and 3.")


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    spins: jnp.ndarray,
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
    ndim: int = 3,
    interaction_strength: float = 1.0,
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps.

  Pairs are classified as parallel or antiparallel from the actual per-walker
  spins array, sigma_i sigma_j > 0, rather than from the electron index.

  NOTE: this used to take nspins and split r_ee into blocks by index, which
  assumes alpha electrons are always ordered before beta electrons. That
  assumption holds only while the sampler never permutes spins. With the
  sector-preserving spin-swap moves in mcmc.py it is false: the trunk sees the
  permuted spins but an index-based Jastrow does not, so Psi stops being
  antisymmetric under simultaneous exchange of (r_i, sigma_i) and a spin swap no
  longer gives the same |Psi| as the corresponding position swap.

  Args:
    r_ee: electron-electron distances, shape (nelectron, nelectron, 1).
    params: Jastrow parameters, keys 'ee_par' and 'ee_anti'.
    spins: per-electron spins, shape (nelectron,); only the sign is used.
    jastrow_fun: pair function f(r, cusp, alpha).
    ndim: spatial dimension.
    interaction_strength: Coulomb strength U setting the cusp factors.
  """
  cusp_parallel, cusp_anti = _cusp_factors(ndim, interaction_strength)

  n = spins.shape[-1]
  r = jnp.reshape(r_ee, (n, n))

  # The diagonal of r_ee is exactly zero. Shift it off zero so that jastrow_fun
  # cannot evaluate 0/0 should alpha ever reach 0; these entries are masked out
  # of both sums below, so the value and the gradients are unaffected.
  r = r + jnp.eye(n, dtype=r.dtype)

  parallel = (spins[:, None] * spins[None, :]) > 0
  upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

  u_par = jastrow_fun(r, cusp_parallel, params['ee_par'])
  u_anti = jastrow_fun(r, cusp_anti, params['ee_anti'])

  zero = jnp.zeros((), dtype=u_par.dtype)
  jastrow_ee_par = jnp.sum(jnp.where(upper & parallel, u_par, zero))
  jastrow_ee_anti = jnp.sum(jnp.where(upper & ~parallel, u_anti, zero))

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(ndim: int = 3, interaction_strength: float = 1.0) -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""

  cusp_parallel, cusp_anti = _cusp_factors(ndim, interaction_strength)

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    # |alpha| keeps the pole of alpha^2/(alpha + r) at r = -|alpha| < 0, i.e.
    # outside the range of physical distances. The cusp d f / d r |_{r=0} = cusp
    # is unchanged.
    alpha = jnp.abs(alpha)
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    # The depth of the Jastrow at contact is u(0) = -cusp * alpha, and cusp is
    # proportional to interaction_strength, so alpha must scale as
    # 1/interaction_strength to keep the exponent O(1) per pair. This is exactly
    # the original alpha = 1 rescaled by 1/interaction_strength, and recovers it
    # for interaction_strength = 1.
    alpha_0 = 1.0 / interaction_strength
    params = {}
    params['ee_par'] = jnp.full(
        shape=(1,), fill_value=alpha_0,
    )
    params['ee_anti'] = jnp.full(
        shape=(1,), fill_value=alpha_0,
    )
    return params

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      spins: jnp.ndarray,
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    return _jastrow_ee(r_ee, params, spins, jastrow_fun=simple_ee_cusp_fun, 
                       ndim=ndim, interaction_strength=interaction_strength)

  return init, apply

def get_jastrow(jastrow: JastrowType, jastrow_kwargs: dict) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    print("get_jastrow: Using SIMPLE_EE Jastrow with parameters:")
    print(jastrow_kwargs)
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(jastrow_kwargs["ndim"], jastrow_kwargs["interaction_strength"])
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')
  else:
    print("get_jastrow: NOT using Jastrow")

  return jastrow_init, jastrow_apply
