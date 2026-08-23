# Wigner Molecular Crystals: Spin Ground State Search

**Status: 🚧 Work in progress**

This project applies the [PeriodicWave](https://github.com/mg607/PeriodicWave) neural-network variational Monte Carlo (VMC) framework to **Wigner molecular crystals**, with the goal of identifying the **spin ground state** of these systems.

## What this project adds

Building on the PeriodicWave/FermiNet framework, this repository adds:

- **S² operator evaluation** — computes the total-spin expectation value of the trained wavefunction, used to identify the spin quantum number of candidate ground states.
- **Charge and spin density analysis notebook** — a Jupyter notebook (`[notebook filename]`) that loads trained checkpoints and visualizes charge density and spin density across the simulation cell.

## Background

This code builds directly on:

- [PeriodicWave](https://github.com/mg607/PeriodicWave) (Geier & Nazaryan, 2025) — adapts a self-attention neural network wavefunction ansatz (and SlaterNet/Hartree-Fock solver) to two-dimensional periodic solids.
- [FermiNet](https://github.com/deepmind/ferminet) (Pfau, Spencer, de G. Matthews, Foulkes, 2020) — the original neural-network VMC solver for continuous-space fermion systems that PeriodicWave itself extends.

Full citations are in the Credits section below.

## Installation

Installation follows the same steps as upstream PeriodicWave:

1. Clone this repository
2. Create a Python 3.13 environment (conda) and virtualenv
3. From the project directory: `pip install -e .`
4. Optional GPU support: `pip install -U "jax[cuda12]"`

See the [PeriodicWave README](https://github.com/mg607/PeriodicWave) for the full, verified install walkthrough — the underlying package setup is unchanged.

## Usage

- Run a spin ground-state search: `[your command here]`
- Evaluate S²: `[your script/command here]`
- Analyze charge/spin density: open `[notebook filename].ipynb`

## Credits

The underlying VMC method and neural network architecture are described in:

> M. Geier, K. Nazaryan, T. Zaklama, L. Fu, "Self-attention neural network for solving correlated electron problems in solids," *Phys. Rev. B* 112, 045119 (2025).

which itself builds on DeepMind's FermiNet:

> D. Pfau, J. S. Spencer, A. G. de G. Matthews, W. M. C. Foulkes, "Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks," *Phys. Rev. Research* 2, 033429 (2020).

See the [PeriodicWave repository](https://github.com/mg607/PeriodicWave) for the complete set of related citations (PsiFormer, periodic boundary conditions, Forward Laplacian).

## License

Apache-2.0, consistent with upstream PeriodicWave and FermiNet. See [LICENSE](LICENSE).
