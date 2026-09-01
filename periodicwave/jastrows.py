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
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
    ndim: int = 3,
    interaction_strength: float = 1.0,
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps.

  NOTE: nspins is used to split r_ee into same-spin and opposite-spin blocks by
  electron index, which assumes the standard ordering of alpha electrons before
  beta electrons. The actual (n_up, n_down) must be passed here; passing a
  merged tuple such as (n_total, 0) silently classifies every pair as
  parallel-spin and drops the antiparallel channel entirely.
  """
  cusp_parallel, cusp_anti = _cusp_factors(ndim, interaction_strength)

  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.size > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, cusp_parallel, params['ee_par']) 
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  # NOTE: .size, not .shape[0]: the latter is n_up, which is non-zero even for a
  # fully polarised system with no antiparallel pairs at all.
  if r_ees[0][1].size > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], cusp_anti, params['ee_anti'])) 
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

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
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    return _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun, 
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
