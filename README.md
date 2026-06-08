# Particle-level moment data assimilation for Vlasov–Poisson

Reference implementation and reproducibility scripts for the paper

> *Particle-level moment data assimilation for the Vlasov–Poisson system by
> Wasserstein-gradient-flow nudging.*

The code assimilates **hydrodynamic-moment observations** (density `ρ`, bulk velocity
`u`, temperature `T`) directly into a particle-in-cell (PIC) Vlasov–Poisson(–collisions)
simulation by a continuous **nudging** (Azouani–Olson–Titi-type) correction applied to
the particle pusher. Four feedback laws are implemented and compared:

| key   | scheme | description |
|-------|--------|-------------|
| `A`   | Formulation A | velocity-weighted Wasserstein gradient flow of the quadratic moment loss (recommended) |
| `B`   | Formulation B | direction-split, well-posed variant |
| `C`   | Formulation C | information-geometric (KL) feedback with a parameter-free 1:2 velocity-to-temperature rate ratio |
| `aot` | AOT | constant-gain Azouani–Olson–Titi baseline |
| `none`| — | unassimilated reference run |

A small numpy/scipy **1D1V and 2D2V electrostatic-PIC reference backend** is included, so
every result in the paper reproduces with no external dependencies beyond numpy/scipy/
matplotlib. (A WarpX backend stub is present but optional and not needed for the paper.)

## Install

```bash
conda env create -f environment.yml      # creates the `mfda` env
conda activate mfda
pip install -e .
pytest -q                                # unit + smoke tests
```

or, without conda:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

Run the four-formulation driven-recovery comparison (off-resonance BGK, 1D1V):

```bash
python scripts/run_recover_check_ABC.py \
    --config configs/test_moment_obs_ABC_driven_BGK.yaml
```

This runs the truth and each assimilating formulation in lock-step and writes, under
`results/<name>/`, one `.npz` per formulation (truth + assimilated diagnostic logs),
`recover_summary.csv` (late-window errors and improvement ratios), `SUMMARY.md`, and
error/phase-space figures. Smoke mode for a quick check:

```bash
python scripts/run_recover_check_ABC.py \
    --config configs/test_moment_obs_ABC_driven_BGK.yaml \
    --np-override 20000 --n-steps-override 80 --name-suffix _smoke
```

See **[REPRODUCE.md](REPRODUCE.md)** for the exact config + command for every table and
figure in the paper.

## Layout

```
src/mfda/            importable library
  assimilation_moments.py      main 1D1V truth+assim moment-DA loop (run_moments)
  assimilation_moments_2d.py   2D2V version
  backend_reference.py         numpy 1D1V ES-PIC (leapfrog, CIC, FFT Poisson)
  backend_reference_2d.py      2D2V reference backend
  nudging_moments.py           Formulation A/B/C/AOT particle feedback kernels
  collisions.py                BGK / Lenard–Bernstein / Dougherty (local-OU) substeps
  filtering.py                 spectral low-pass mollifier K_h
  observation_moments.py       moment observation operator (full / noisy / sparse)
  diagnostics.py               e_rho, e_u, e_T, e_f, energy, Fourier modes
  poisson.py                   FFT periodic Poisson solver
  config.py / config_2d.py     YAML config loader + validation
scripts/             thin CLI drivers (run_*.py) and plotting (plot_*.py)
configs/             YAML configs (one per experiment; see REPRODUCE.md)
tests/               pytest unit + smoke tests
slurm/               Bridges-2 SLURM batch templates
```

## Conventions

- Units `m_e = q_e = ω_p = 1`; periodic torus with fundamental wavenumber `k`
  (domain length `L = 2π/k`).
- The collision Monte-Carlo substeps conserve momentum/energy only in expectation; at
  finite particle number `N_p` each step incurs an `O(N_p^{-1/2})` stochastic
  conservation error (the same floor that sets the late-time errors).
- The numerical "Lenard–Bernstein" / Dougherty update relaxes toward the **local**
  moments (a conserving, Dougherty-type operator); the fixed-background linear
  Lenard–Bernstein operator is used only as a theoretical surrogate in the paper's
  well-posedness analysis.

## License

MIT (see [LICENSE](LICENSE)).
