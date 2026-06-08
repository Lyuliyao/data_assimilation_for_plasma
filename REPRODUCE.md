# Reproducing the paper

Every table and figure is produced by the scripts below from the YAML configs in
`configs/`. Outputs are written under `results/<name>/` as `.npz` arrays,
`recover_summary.csv` / `rate_summary.csv`, `SUMMARY.md`, and figures. Runs use the
numpy reference backend and need only numpy/scipy/matplotlib.

All runs share: periodic torus with `k = 0.5` (so `L = 4π`), velocity domain `[-8, 8]`,
spectral mollifier `K_h(k) = exp(-(k/k_c)^16)` with `k_c = 0.25 k_max`, regularized
floors `ρ_floor = T_floor = 1e-3`, and **unit gains** for every scheme
(`γ_1 = γ_2 = γ_3 = 1`, `V_* = 1`, `λ = 1`, `μ_ρ = μ_u = μ_T = 1`) unless a flag overrides them.

Typical single 1D run: a few minutes per formulation at `N_p = 5e5`, `n_steps = 1000`
on one core. Heavy / multi-seed sets should be run on a compute node (see `slurm/`),
not a login node. Set `OMP_NUM_THREADS=1` and run one process per core.

## Convergence-rate validation — Table (rates), Fig (e1), Fig (gamma)

```bash
python scripts/run_rate_validation.py --config configs/exp1_homogeneous_rate.yaml
```
Homogeneous Maxwellian, no collisions, no driver; fits the velocity/temperature decay
rates over `t ∈ [0.3, t_floor]`. Formulation C gives the exact 1:2 ratio. The
gain-sweep figure repeats this with `--gamma-override` (or the `exp1_homogeneous_rate_g*`
configs).

## Driven recovery — Table (driven ratios), Fig (driven)

Off-resonance and resonant BGK, four formulations:
```bash
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK_truly_resonant.yaml
```

### Seed spread and absolute-error table (off-resonance, 5 seeds)
```bash
for s in 0 1 2 3 4; do
  python scripts/run_recover_check_ABC.py \
      --config configs/test_moment_obs_ABC_driven_BGK.yaml \
      --seed $s --name-suffix _seed$s
done
python scripts/aggregate_seeds.py            # means ± std (ratios and absolute errors)
```

## Robustness across collision model / collisionality — Table (robustness)

```bash
# collisionality sweep (override ν)
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --nu-override 0.1 --name-suffix _nu0p1
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --nu-override 1.0 --name-suffix _nu1p0
# Lenard–Bernstein / Dougherty (local-OU) and damped-Langmuir
python scripts/run_recover_check_ABC.py --config configs/exp_driven_LB.yaml
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_damped_langmuir.yaml
```

## Imperfect observations and model error — Table (imperfect)

```bash
# observation noise (sets kind=noisy)
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --sigma-u 0.05 --sigma-T 0.05 --name-suffix _noise05
# sparse-in-time observations
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --every-q 50  --name-suffix _q50
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --every-q 200 --name-suffix _q200
# model error: wrong collision frequency / wrong operator in the assimilating run
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --assim-nu 0.25 --name-suffix _assimNu0p25
python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK.yaml --assim-kind lb --name-suffix _assimLB
```

## Density-mismatch and resolution study — Table (density), Table (resolution)

Deliberate wrong initial density (`assim_ic.rho_amp = 0.3`), 5 seeds; independent
higher-resolution truth via `--truth-np-override` (truth and assim then use independent
RNG streams):
```bash
for s in 0 1 2 3 4; do
  # density mismatch, matched counts
  python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK_rhomis03.yaml \
      --truth-np-override 500000  --seed $s --name-suffix _s$s
  # higher-resolution truth (4x), fixed assim count
  python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK_rhomis03.yaml \
      --truth-np-override 2000000 --seed $s --name-suffix _truth2M_s$s
  # assimilating-particle sweep at fixed high-resolution truth
  python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK_rhomis03.yaml \
      --np-override 125000 --truth-np-override 2000000 --seed $s --formulations none,A --name-suffix _truth2M_np125000_s$s
  python scripts/run_recover_check_ABC.py --config configs/test_moment_obs_ABC_driven_BGK_rhomis03.yaml \
      --np-override 250000 --truth-np-override 2000000 --seed $s --formulations none,A --name-suffix _truth2M_np250000_s$s
done
```
A SLURM batch template for this campaign is in `slurm/`.

## 2D2V driven recovery — Table (2D), Fig (2D)

```bash
python scripts/run_recover_check_2D.py --config configs/exp_2d_driven_highNp.yaml
python scripts/aggregate_recover_check_2D.py
```

## Limitation: non-equilibrium (two-stream) truth

```bash
python scripts/run_recover_check_ABC.py --config configs/exp2a_two_stream.yaml
python scripts/run_recover_check_ABC.py --config configs/exp_obstruction_two_stream.yaml
```
A prescribed, held-fixed bimodal truth (an observation-model stress test, not a
self-consistent VP–BGK solution): the moment channels recover `u, T`, but the
non-hydrodynamic shape `e_f` is not reconstructible from moment data.

## Figures

`scripts/plot_*.py` and `scripts/regenerate_recover_figs.py` render the paper figures
from the `.npz` outputs above; shared styling is in `scripts/plot_style.py`.
