# Robust DPS over a climate × demand ensemble — methodology

Status: **DRAFT for review (no code written yet).** This document fixes the design
of the Robust DPS extension before implementation, including the open issues found
during scoping. Decisions already taken are marked ✅; open questions are marked ❓.

---

## 1. Goal

The current DPS optimizes each policy against a **single deterministic future** —
the baseline `run_id = 0` input series (climate, population, agric area, demand all
frozen). Its NSGA-II therefore finds policies that are optimal *for that one future*,
with no protection against climate or demand uncertainty.

**Robust DPS** instead evaluates every candidate policy against an **ensemble of
plausible futures** and optimizes a *robustness statistic* of the objectives across
that ensemble. The result is a Pareto front of policies that perform well *across*
uncertainty, not just on one trajectory.

---

## 2. What already exists (no rebuild needed)

| Piece | File | Status |
|---|---|---|
| Per-scenario loop + objective aggregation | `pipe_simulation_weap.py` (`simulation()` already loops `self.scenarios` and `np.nanmean`s the objectives) | reusable |
| Climate overwrite (precip/temp per subcuenca) | `climate_sampler.py` (`apply_scenario_to_X`) | reusable |
| Demand transforms (pop growth, area mult, demand scaling) | `demand_builder.py` | reusable (see open issue #1) |
| Ensemble parameters | `config_weap.py` (`GCM_LIST`, `POP_GROWTH_RATES`, `AREA_MULTIPLIERS`, `n_climate_scenarios`) | reusable |

The bridge `PipeWEAP` accepts a `scenarios=[...]` list; `main_par_weap.py` just passes
`None`. So the missing pieces are: a **scenario builder**, the **robustness metric**,
and an **entry point** that wires them — plus fixing two data issues (§6).

---

## 3. Decisions taken

✅ **Ensemble = climate × demand (cross product).** Most thorough; robust to both
hydrologic and demand uncertainty. Implies a runtime multiplier = `N_climate × N_demand`
(see §5, §7).

✅ **Robustness metric = mean + λ·std across scenarios** (risk-penalized expectation).
For each objective `Jk` (all stored in NSGA's *minimize* convention), the value the
optimizer sees is

```
Jk_robust = mean_s(Jk_s) + λ · std_s(Jk_s)
```

over scenarios `s`. This rewards low average objective **and** low spread (consistency
across futures). `λ` is a tunable risk-aversion knob (`λ=0` → pure expected value;
larger `λ` → more conservative). Applied uniformly to all 5 minimized objectives, so
for the maximized ones (J1 storage, J3 agri — stored negated) the +λ·std term correctly
penalizes futures where the outcome is *bad*. ❓ **Default `λ` to confirm** (proposed
`λ = 1.0`).

✅ **Keep the running single-scenario production as the deterministic baseline.** It
will finish (~40 h) and serve as the "optimize-for-one-future" reference to compare the
robust fronts against.

✅ **Robust pass uses only 2 of the 4 models** (spend the compute budget on **climate
breadth** — the priority — and a few demand corners, not on model breadth).
✅ **The 2 models are chosen *after* the single-scenario baseline finishes** — pick the
**2 best of the 4** from the baseline fronts (ranked by Pareto quality: hypervolume +
how each does on the priority objectives J1 storage / J2 unmet / J4 cost). Not pre-committed.

✅ **Budget = Preset C** (§7): 3 seeds × 4000 evals × (5 climate × 3 demand = 15 scenarios),
2 models → 6 runs, ~15 h wall-clock on 6 cores.

✅ **Run the robust optimizations in parallel** (one process per physical core) — RAM is
not the constraint (~500 MB/run, 23.8 GB total ⇒ ~30 fit), the **6 physical cores** are.
Pin `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` per process so they don't oversubscribe.

✅ **Zero-touch isolation of the baseline.** The production loop spawns a fresh
`python main_par_weap.py` per seed (~every 2 h), so editing any file on its code path
would corrupt the in-flight baseline. Robust DPS will therefore live in **new files
only** — `scenario_builder.py`, a subclass `RobustPipeWEAP` (overrides `simulation()`
for the mean+λ·std metric), a new entry point `main_robust_weap.py`, and a runner
`run_robust.sh`. `main_par_weap.py` and `pipe_simulation_weap.py` are left untouched.
Additions to `config_weap.py` (if any) are purely additive, env-gated, and never alter
the existing default path.

---

## 4. Ensemble definition

**Priority: climate breadth.** Climate is the dominant uncertainty, so we spend the
ensemble budget mostly on climate realizations and keep demand to a few **corners**
(not the full 2×3 grid).

### 4.1 Demand axis — corners only (deterministic, from `demand_builder`)
Demand stress ∝ population × irrigated area. Instead of the full 6-combo grid, use
**3 corners** spanning the stress range:
- **HIGH** demand = pop **5%** × area **1.00**
- **MID**  demand = pop **2%** × area **1.00**  (or 5% × 0.85)
- **LOW**  demand = pop **2%** × area **0.50**

❓ Confirm corner count (2 = HIGH/LOW, or 3 = HIGH/MID/LOW). Pop rates from
`POP_GROWTH_RATES={2%,5%}`; areas from `AREA_MULTIPLIERS={1.00,0.85,0.50}`.

### 4.2 Climate axis — the breadth dimension (from the zarr's raw precip/temp)
GCM *labels* are not cleanly recoverable from the zarr, but the climate *series* are.
Realizations are selected by **total horizon precipitation** across the 6 subcuencas
(filtering out degenerate P=0 runs — issue #2), picking spread points dry→wet:
- proposed **4–5 climate realizations** (e.g. DRY, DRY-MID, MEDIAN, WET-MID, WET).
- ❓ Confirm count (4 or 5).

### 4.3 Cross product
`N_climate × N_demand_corners`. Examples: **4×2 = 8**, **5×3 = 15**. This is the ensemble
each policy is evaluated on, every NSGA-II evaluation (drives the runtime multiplier).

---

## 5. Scenario builder (planned `scenario_builder.py`)

Each ensemble member is a fully-specified **normalized** input `X` (1872 × n_x) that
the surrogate can roll out. To avoid disturbing the recursive **GW-lag init columns**,
scenarios are built by **copying the normalized template and re-writing only the
changed columns**:

1. Start from the normalized template `X_filtered` (run-0 baseline).
2. **Climate**: for each `Precipitation__Subcuenca_*` / `Temperature__Subcuenca_*`
   column, take the *raw* series from the chosen climate run (zarr `X`), normalize it
   per-column (transform + z-score via the surrogate's scalers), overwrite.
3. **Demand**: from the raw base values, apply pop growth (exponential, annual) and area
   multiplier (constant); scale the `AP_WaterDemand__*` series by pop growth; normalize;
   overwrite. (Population-count columns handled per open issue #1.)
4. Everything else (GW lags, actions placeholder, etc.) stays at the template values; the
   DPS policy still injects the 5 actions during rollout as today.

Output: `list[np.ndarray]` of normalized X + a label per scenario (`climate=DRY,pop=5%,area=0.50`).

Column name lists reused from `climate_sampler.SUBCUENCAS`, `demand_builder.{AREA_COLUMNS,
DEMAND_AP_COLUMNS, POP_COLUMNS}`. Template ↔ zarr feature order verified identical.

---

## 6. Open issues found during scoping (must resolve before coding)

1. **Population columns don't match (`pop: 0/8`).** `demand_builder.POP_COLUMNS` (e.g.
   `"AP_Poblacion__APR_Q01_Dem_JuntaTilama (cap)"`) are **absent** from the template's
   feature names — so `apply_population_growth` is a no-op as written. Need to find the
   real population feature names (or confirm population enters the model *only* through
   the `AP_WaterDemand__*` demand columns, in which case pop growth is applied solely via
   `scale_ap_demand_with_population`, which uses the 3 present `DEMAND_AP_COLUMNS`).
   ❓ Also reconcile: template has **8** `AP_WaterDemand__*` features but
   `DEMAND_AP_COLUMNS` lists only **3** — decide whether all 8 should scale with pop.

2. **Zero-precip climate runs.** The naive "driest run" pick returned `run109` with total
   precip = 0 — a degenerate/empty climate series, not a real dry future. The climate
   selector must **filter to runs with total precip > 0** (and ideally sanity-check
   temp/precip ranges) before ranking dry→wet. (MEDIAN=run136, WET=run449 looked valid.)

3. **`λ` and ensemble counts unset** — see ❓ in §3/§4.

---

## 7. Runtime model (with parallelism)

Each NSGA-II evaluation runs the surrogate **once per scenario**:

```
per_run     = evals × scenarios × t_eval          (t_eval ≈ 0.9 s/scenario, from smoke)
n_runs      = n_models × n_seeds
wall_clock  = ceil(n_runs / P) × per_run           (P = parallel processes ≈ 6 cores)
RAM         = P × ~0.5 GB        (≪ 23.8 GB — never the limit)
```

The PC has **6 physical cores** (5 while the baseline still runs). Independent
`(model,seed)` runs are launched in parallel waves; the ensemble is the cost multiplier.

### Presets (2 models, climate-prioritized, parallel on 6 cores)

| Preset | seeds | evals | climate × demand | scen. | n_runs | per_run | waves | **wall (≈6 cores)** |
|---|---|---|---|---|---|---|---|---|
| **A — fast**     | 2 | 3000 | 4 × 2 | 8  | 4 | ~6.0 h  | 1 | **~6 h** |
| **B — balanced** | 2 | 3000 | 5 × 3 | 15 | 4 | ~11.3 h | 1 | **~11 h** |
| **C — thorough** | 3 | 4000 | 5 × 3 | 15 | 6 | ~15.0 h | 1 | **~15 h** |

(`per_run = evals × scen × 0.9 s`. All presets are 2 models → `n_runs = 2×seeds`, all of
which fit in one parallel wave on 6 cores, so wall ≈ per_run.) RAM at P=6 ≈ 3 GB.

> If run **alongside** the baseline (5 cores free), B still fits in one wave (4 runs ≤ 5);
> C's 6 runs would need 2 waves → ~30 h. Cleanest is to start the robust pass once the
> baseline finishes, or cap the baseline-coexisting robust pass at ≤5 runs.

---

## 8. Proposed implementation plan (after this doc is approved)

1. Resolve open issues §6 (pop column names; valid climate-run filter).
2. `scenario_builder.py` — build + label the climate × demand ensemble (normalized X list).
3. `RobustPipeWEAP(PipeWEAP)` — override `simulation()`: collect per-scenario `all_J`,
   return `mean + λ·std` per objective. (Isolated; base class untouched.)
4. `main_robust_weap.py` — build ensemble, run NSGA-II, save front (+ ensemble metadata).
5. `run_robust.sh` — runner over the chosen variants/seeds; resumable.
6. **Smoke**: 1 variant, full ensemble, ~500 evals → validate correctness + **measure
   per-eval time** → finalize the production budget (§7).
7. Scale to the agreed budget; compare robust fronts vs the single-scenario baseline via
   `make_comparison_plots.py` (extended to overlay baseline vs robust).

---

## 9. Status & remaining items

✅ Budget = **Preset C** (3 seeds × 4k evals × 15 scenarios, 2 models, ~15 h).
✅ Model choice = **2 best of 4, decided after the baseline finishes** (ranked by Pareto
quality on J1 storage / J2 unmet / J4 cost + hypervolume).
✅ Ensemble = climate **5** × demand **3** (HIGH/MID/LOW) = 15 scenarios.

Still to confirm / do (tomorrow):
- ❓ `λ` (risk aversion) — proposed **1.0**.
- ❓ Open issue #1 (population columns) — confirm pop enters via the demand columns only.
- ⏳ Build per §8 (scenario builder, `RobustPipeWEAP`, entry point, runner), smoke to
  measure real `t_eval`, then launch Preset C.

## 10. Execution timeline
1. **Now → ~Jun 12 AM:** single-scenario baseline (4 models × 5 seeds × 8k) runs to
   completion (~40 h from 17:09 Jun 10). Resumable; survives overnight.
2. **After baseline:** run `make_comparison_plots.py` on `runs_weap/prod`, rank the 4
   models, **pick the 2 best**.
3. **Then:** resolve issue #1, build the robust files, smoke-measure `t_eval`, launch
   **Preset C** (~15 h) on the 2 chosen models.
