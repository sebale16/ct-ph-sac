# dm_control ports: the porting contract

This package holds the CT-PH-SAC testbeds ported from the DeepMind Control Suite. The
in-house testbeds (windy pendulum, Acrobot-XK) stay in `environment/ph_sac/tasks.py`
and `common/ph_config.py`. Sixteen tasks are registered, all implemented: the fourteen
contact-free tasks (the stubs described below were replaced by the ports on 2026-11-07,
and the stub procedure is kept for future domains) and, since 2026-11-09, the two swimmer
tasks, the first whose true dynamics need a plant option (velocity-dependent damping).
Since 2026-11-10 no port has a failure set: the cart-pole slider (all six tasks), the
reacher wrist and the point mass's slides keep dm_control's soft joint limits, as the
swimmer's joints do (until then those stops were failure boundaries with payoff 0 and
the boundary lift; every cart-pole, reacher and point-mass run before 2026-11-10 used
that convention). Failure sets, the failure payoff and the boundary lift remain only on
the in-house windy pendulum and Acrobot-XK.

```text
environment/ph_sac/
  registry.py          TestbedSpec, register_testbed, testbed_spec, registered_testbeds
  base.py              MujocoPHEnv (interval physics, context support), validated_state_spec
  tasks.py             in-house testbeds and the name-based dispatchers (via the registry)
  dmc/
    __init__.py        imports every domain module, which registers its testbeds
    _common.py         tolerance, failure/payoff/lift helpers (NO_FAILURE for the ports),
                       limit_attributes (the MJCF solreflimit/solimplimit of a kept limit),
                       compiled-model inertials, todo
    chain.py           planar serial-chain dynamics (torch and numpy)
    limited_chain.py   the chain plus MuJoCo's soft joint-limit forces (the reacher's and
                       the multi-pole cart-poles' oracle, torch and numpy)
    pendulum.py        pendulum_swingup                                    (ported)
                       reward: dm_control's tolerance, row margin 2 (dense since
                       2026-11-09; dm_control's task has margin 0, the indicator)
    acrobot.py         acrobot_swingup, acrobot_swingup_sparse             (ported)
                       rewards: lyapunov (acrobot_swingup, Xin--Kaneda), tolerance
                       (acrobot_swingup_sparse), zhang2023 (no row since 2026-11-08),
                       lyapunov_shoulder (optional variant row, mode lyapunov_shoulder)
    acrobot_lqr.py     upright LQR and region of attraction (S, rho) of the zhang2023 reward
    cartpole.py        cartpole_swingup (ported); cartpole_balance, cartpole_balance_sparse,
                       cartpole_swingup_sparse, cartpole_two_poles, cartpole_three_poles (ported;
                       dm_control's soft rail limit KEPT since 2026-11-10, the limit force
                       in the oracle; the rows set contact_handling exclude since
                       2026-11-11, network.limit_port from 2026-11-10 to 11-11); optional
                       Lyapunov rewards lyapunov_cart, lyapunov_cart_straight on the five
                       later tasks (variant rows of two_poles and three_poles, unused in the
                       final run)
    reacher.py         reacher_easy, reacher_hard                          (ported; use the context;
                       the soft wrist limit KEPT since 2026-11-10, rows set network.limit_port)
    point_mass.py      point_mass_easy                                     (ported; the soft slide
                       limits KEPT since 2026-11-10, row sets contact_handling exclude
                       since 2026-11-11, network.limit_port from 2026-11-10 to 11-11)
    lqr.py             lqr_2_1, lqr_6_2                                    (ported)
    swimmer.py         swimmer_swimmer6, swimmer_swimmer15                 (ported 2026-11-09;
                       the target as context, dm_control's soft joint limits KEPT, the
                       fluid drag and the limit forces in the oracle; rows set
                       network.damping_velocity and network.limit_port; dm_control's
                       own 2 ms Euler physics and the semi_implicit model map since
                       2026-11-10)
    joint_limits.py    MuJoCo's soft joint-limit constraint (impedance, reference
                       acceleration, regularizer, one-sided complementarity) in torch and
                       numpy, and mujoco_limit_rows, the efc_* cross-check (reusable: the
                       swimmer, cart-pole, reacher and point-mass oracles)
common/ph_dmc/
  __init__.py          imports the seven factory modules
  _common.py           fixed_clock, box_rule_controller, dmc_state_spec (limited ranges too),
                       dmc_testbed_config, LIMIT_KEYS, MUJOCO_DEFAULT_LIMIT,
                       require_no_failure_set; since 2026-11-11 the irregular clocks:
                       CLOCK_SETTINGS_KEYS (optional testbed_settings clock_kind,
                       tail_probability, tail_split, beta_shape of the pendulum, acrobot,
                       cart-pole and point-mass testbeds), irregular_clock_bounds (T/5 to
                       3T, CT-SAC's rule), dmc_clock, with_clock_settings (the factories);
                       the table's rows of mode irregular use them
  <domain>.py          one config factory per testbed (all implemented)
tests/test_ph_dmc.py   registry, helpers, chain (against dm_control too), context tests
tests/test_ph_dmc_swimmer.py  the swimmer ports, the plant options they use, and the
                       regression of the existing rows' plant and training
tests/test_ph_dmc_joint_limits.py  the kept soft limits of the cart-poles, reachers and
                       point mass: compiled parameters against dm_control's, MuJoCo's efc_*
                       rows, oracle, RK4 interval map and trajectories through the limits,
                       no failure set, the rows' limit port
```

Each stub module's docstring says what its domain needs: the dm_control facts, the
port's state layout, the reward formula, the failure set, the oracle, and the open
choices the table flags. Each stub factory's docstring lists its table row's settings,
the exact `TestbedConfig` fields the row declares, and a sketch built from the helpers.

## Registry API

```python
from environment.ph_sac.registry import TestbedSpec, register_testbed, testbed_spec, registered_testbeds

@dataclass(frozen=True, kw_only=True)
class TestbedSpec:
    name: str                                   # TestbedConfig["name"] and the table's env_testbed
    env_class: Type[MujocoPHEnv]                # env_class(config, decision_mode=..., seed=...)
    config_factory: Callable[..., TestbedConfig]  # factory(**testbed_settings)
    settings_keys: Tuple[str, ...]              # required testbed_settings keys
    optional_settings_keys: Tuple[str, ...] = ()
    reward_rate_tensor: Callable[[Tensor, Tensor, TestbedConfig], Tensor]   # (states [B,D], torques [B,m], config) -> [B,1]
    failure_mask_tensor: Callable[[Tensor, TestbedConfig], Tensor]          # -> bool [B]
    terminal_payoff_value: Callable[[TestbedConfig], float]                 # F; 0 without a failure set
    terminal_boundary_functions: Callable[[TestbedConfig], Tuple[H, B]]     # V = H + B * network
    oracle_callbacks: Callable[[TestbedConfig], Tuple[drift, diffusion]]    # torch.func-compatible
    implemented: bool = True                    # False: a stub; its row parses but does not build
    upright: Optional[Callable[[TestbedConfig, ndarray, float], bool]] = None  # (config, state, tube)
    reward_rate_bounds: Optional[Callable[[TestbedConfig], Tuple[float, float]]] = None  # (low, high)
    physical_change: Optional[Callable[[plant_parameters, kind, scale], plant_parameters]] = None  # adaptation experiment
    discount_setting: bool = False              # True: takes testbed_settings discount_rate (per second)

register_testbed(spec, *, replace=False) -> spec   # a second registration of a name raises
testbed_spec(name) -> TestbedSpec                  # unknown name: ValueError
registered_testbeds(*, implemented_only=False) -> Tuple[str, ...]   # sorted names
```

`testbed_spec` and `registered_testbeds` import `environment.ph_sac.tasks` and
`environment.ph_sac.dmc` first, so every entry exists however the caller started.
Everything name-based reads the registry: `tasks.make_env`, `reward_rate_tensor`,
`failure_mask_tensor`, `terminal_payoff_tensor`, `terminal_boundary_functions`,
`oracle_callbacks`, `make_oracle_plant`, the evaluation's `is_upright`
(`evaluations/ph_sac/protocol.py`, which counts `upright_seconds`; `upright=None` counts
no state as upright), the implicit agent's reward head
(`ImplicitCTPHSAC._build_reward` reads `reward_rate_bounds`), and
`benchmarks/run_ph_sac.py` (its `_TESTBED_SETTINGS`,
`_OPTIONAL_TESTBED_SETTINGS` and `_TESTBED_FACTORIES` are registry snapshots;
`_build_testbed` reads the spec; `--list` prints `builds` for implemented testbeds and
`parses; not implemented` for stubs). `common.ph_types.TestbedName` is plain `str`.

### What a domain module registers, and how a stub is replaced

Edit only your domain's two modules, `environment/ph_sac/dmc/<domain>.py` and
`common/ph_dmc/<domain>.py`, plus your own test file `tests/test_ph_dmc_<domain>.py`. To
replace a stub:

1. Implement the factory in `common/ph_dmc/<domain>.py`. Keep its signature: the
   keyword arguments are the row's `env_*` keys, the four dicts `failure_limits`,
   `reward_parameters`, `reset_parameters`, `plant_parameters`.
2. Replace the placeholder env class and the five placeholder tensor functions in
   `environment/ph_sac/dmc/<domain>.py`. Delete the `todo` imports you no longer use.
3. In the existing `register_testbed(TestbedSpec(...))` call, set `implemented=True`
   (or drop the argument), point the fields at the real functions, and add `upright`
   where the task has an upright pose. Keep `settings_keys` equal to the row's `env_*`
   keys. Do not register a name twice.
4. Add the validation tests below. `python benchmarks/run_ph_sac.py --list` must then
   print `builds` for your rows. `load_protocol_from_table` raises if the factory's
   `TestbedConfig` disagrees with any `task_*` cell. The registry-driven tests in
   `tests/test_ph_hyperparams.py` pick up the change and need no edit.

Do not edit these shared files: `registry.py`, `base.py`, `tasks.py`, `chain.py`,
`_common.py` (both packages), `benchmarks/hyperparams/ct_ph_sac.csv`, and the existing
tests. If one of them must change (for example a new `plant_parameters` key for the lqr
stiffness seed, which changes the CSV row and its generator
`benchmarks/make_ct_ph_sac_table.py`), report it instead of editing it. The row
comments still say "Not ported"; the generator owns that text.

## Conventions

**State layout.** A state is `(positions, velocities)` as `StateSpec` declares it
(`common/ph_types.py`). Positions are the observed physical coordinates
`q[:physical_dim]`, followed by `context_dim` context coordinates. Velocities are
`qd[:velocity_dim]`. The observation projection `P` (`position_dim x velocity_dim`) is
the identity on the physical block and zero on the context rows. `dmc_state_spec`
builds it. Torques are physical: newtons or N m, with gear 1 in our MJCF. The
controller clips them to `torque_low/high`. dm_control's normalized control in its
rewards is `u = torque / torque_limit` (`_common.torque_limit(config)`).

**Angles.** Hinge angles are MuJoCo's joint angles as dm_control's models define them:
zero means pointing up, positive tilts toward +x (hinges about +y, links along +z), and
the second and later links are measured relative to the previous link. `chain.py` uses
the same convention. For the reacher (hinges about +z, links along +x), take
"up" = +x and "side" = +y with gravity 0.

**Wrapping.** `periodic_position_indices` are wrapped to [-pi, pi) in observations
(`MujocoPHEnv.observe`) and enter the networks as sine and cosine. Travel coordinates,
such as a cart or slide position, stay raw. A coordinate whose failure limit needs the
unwrapped value must be left out of `wrapped_observation_indices`, as Acrobot-XK's
elbow is. Use `setpoint_error: wrapped` for periodic actuated coordinates. The failure
predicate and the boundary factor `1 - (x/x_lim)^2` read the observed value.

**Context** (`StateSpec["context_dim"]`, default 0). An episode-constant quantity that
enters the reward, such as the reacher's target `(x*, y*)`. It occupies the last
`context_dim` positions, so `position_dim = nq_observed + context_dim`. The table's
`task_nq` counts only the physical positions, and `task_context_dim` counts the
context. Because the context rows of `P` are zero, its rate `P v` and its drift are
zero. `MujocoPHEnv` support:

- `observe()` returns `(qpos[:physical] wrapped, self._context, qvel)`.
- `_set_state(state)` writes the physical positions and velocities to MuJoCo and stores
  the context through `_set_context`. That calls the hook `_apply_context()`, a no-op
  that the reacher overrides to move its target geom for rendering.
- `reset()` zeroes the context before `_reset_state`. The task draws the context there
  from `self._reset_rng` and writes it with `_set_state`.
- `mujoco_drift` ignores the context and returns zero for its rate. MuJoCo's dynamics
  must not depend on it.
- `snapshot`/`restore` carry it, and `env.context` returns a copy.

On the model side, `PortHamiltonianPlant` gives the context zero drift, zero diffusion
and zero covariance rows. Its networks still read it as a raw position feature, as do
`StateFeatures` and the critics. `plant_likelihood_loss` and the calibration diagnostics
leave the context rows out. An oracle's callbacks must return zero drift and zero
diffusion rows on the context: `chain.chain_callbacks(chain, context_dim=k)` does this.

**Model integrator.** Every model integration of a plant over a decision interval (the
plant fit's endpoint mean, the explicit rollout target, the rate head's rollouts, the
gate paths, the plant diagnostics) goes through `models/ph_sac/rollouts.model_substep`,
with the top-level config key `model_integrator` (table column `model_integrator`).
`rk4`, the default since 2026-11-09, is MuJoCo's RK4 step: classical RK4 of the drift
over each substep of `model_step`, the controller torque (the environment's PD law and
clip) evaluated at every stage from the stage's state under the controller's
`refresh_mode` `continuous` (below), or computed at the substep's start and held through
the four stages under `held`.
With the oracle drift and `model_step = physics_dt` it reproduces the environment's
interval endpoint to rounding error (about 1e-15) on a port simulated with RK4, as every
dm_control port is (1 ms RK4), so the true drift is the optimum of the plant fit at that
step; at a row's larger `model_step` (below) it does so within 1e-3 relative error, and
the fit's optimum is a drift that absorbs the coarser step's discretization error.
`semi_implicit` is the velocity-first semi-implicit Euler(-Maruyama) step, the only map
before 2026-11-09; it is MuJoCo's Euler step when no joint has damping, so the windy
pendulum rows (Euler physics, damping 0) and, since 2026-11-10, the swimmer rows
(dm_control's 2 ms Euler, no dof damping; **Swimmer physics** below) use it. Both maps add the same noise increment
`K_h Sigma dW`. The choice matters where the closed loop is stiff, `z = h k_d / m_eff`
not small (`m_eff = 1 / (M^-1)_aa` of an actuated coordinate, `k_d` up to the box's
damping maximum): on the reacher's wrist `m_eff` is 4.9e-5 kg m^2 and `z` reaches 2.06
(0.26 at the shoulder), against at most 0.05 on the other dm_control rows (the swimmer:
0.14 on swimmer6 and 0.36 on swimmer15, `m_eff` 6.7e-6 to 9.2e-6 kg m^2 at most damping
9.5e-4 and 2.4e-3 N m s/rad) and 0.12 on
the in-house Acrobot. There the semi-implicit fit preferred a plant with an 11% heavier
wrist to the true one (a 10% drift error); with `rk4` a warm refit came to 0.4%. A port
simulated with another integrator states in its row which `model_integrator` matches it.

**Controller refresh mode** (controller key `refresh_mode`, table column
`task_refresh_mode`, `continuous` in every factory and row since 2026-11-09). Under
`continuous` a port's environment evaluates the feedback law at every RK4 stage of every
1 ms physics step: `MujocoPHEnv` installs MuJoCo's control callback
(`mujoco.set_mjcb_control`) around each `mj_step` of a feedback interval and removes it
after the step, and the process-global callback writes `ctrl` only for the stepping
environment's own `MjData`, so comparisons with dm_control's physics, which step their
own data, never see it. The model's RK4 evaluates the law at its stages the same way, so
with the oracle drift it matches the environment to about 1e-15 at `model_step =
physics_dt`, and a larger `model_step` is a coarser RK4 of the same continuous closed
loop rather than a different controller. `held` is the sample-and-hold of every run
before 2026-11-09 (the torque computed from the state every `refresh_steps` physics steps
and held through each step and its stages); it reproduces those runs bitwise. A port
test that steps `env.step_dt` with feedback decisions against a hand-written MuJoCo loop
must apply the same convention; torque decisions are held in either mode.

**Model step** (`model_step`, the maximum model substep; per task since 2026-11-09).
`common.ph_config.MODEL_STEPS` declares it by testbed name (`default_model_step` falls
back to the physics step for a testbed not listed), `benchmarks/make_ct_ph_sac_table.py`
writes it into the row, and a factory does not declare it. **The model step is coarsened
only on tasks without joint limits.** On a limit-free task each value is the largest step
tried (1, 2, 5, 10, 15, 20, 30 ms, those dividing the decision interval) at which the
oracle plant's model endpoint over one interval stays within 1e-3 relative error
`|x_h - x_env| / |x_env - x_0|` of the environment's (1 ms RK4, the law at every stage) at
the 99th percentile over 2000 rows, with decisions uniform in the box, half of them with
stiffness and damping in the top 10% of their range (measured 2026-11-09 under
`continuous`). A task with joint limits (a dm_control limit kept: `limited_position_ranges`
and `network.limit_port`) integrates at its physics step (since 2026-11-10; 1 ms RK4, the swimmers' 2 ms Euler; limit-free intervals may take a coarse step, **Contact-triggered fine substepping** below): with the
limit in the oracle, an interval that touches a limit meets the criterion only there, the
limit's damping switching on with a jump (impedance 0.9 at zero penetration) that a longer
RK4 step straddles, and such intervals are common (30% of training intervals on the point
mass and 48% on the cart-pole ended pressed on a limit; the swimmers' joints sat on one
45-92% of the warm-up time).

| Rows | Decision interval | `model_step` | Substeps | p99 error at the step / next step tried |
| --- | --- | --- | --- | --- |
| cartpole (all six) | 10 ms | 1 ms (joint limit); coarse 5 ms with contact substepping | 10 (2 when clear of the rail) | at the rail 2e-15 to 6e-15 / 0.35 to 0.65 at 2 ms; without contact 3.2e-4 to 5.0e-4 at 5 ms, 1.1e-3 to 1.7e-3 at 10 ms |
| pendulum-swingup | 20 ms | 10 ms | 2 | 6.4e-4 / 3.1e-3 at 20 ms |
| acrobot-swingup, swingup_sparse | 10 ms | 2 ms | 5 | 2.6e-4 / 1.5e-3 at 5 ms |
| reacher-easy, reacher-hard | 20 ms | 1 ms (joint limit) | 20 | 6e-16 / 0.75 at 2 ms (RK4 unstable above 1.1 ms: the wrist); at the wrist limit 4e-16 / 0.047 |
| point_mass-easy | 20 ms | 1 ms (joint limits); coarse 10 ms with contact substepping | 20 (2 when clear of the slides' limits) | at a slide 6e-16 / 0.30 at 2 ms; whole-episode starts 3.0e-3 at 20 ms, 2.6e-4 at 10 ms |
| lqr-lqr_2_1, lqr_6_2 | 30 ms | 10 ms | 3 | 5.3e-4, 6.7e-4 / 1.2e-3, 2.2e-3 at 15 ms |
| swimmer6, swimmer15 | 30 ms | 2 ms, the physics step (dm_control's Euler, since 2026-11-10), `semi_implicit` | 15 | the simulator's own map: p99 4.2e-14, 8.7e-13 (max 9.1e-14, 2.7e-6 at MuJoCo's solver tolerance 1e-8; 1.4e-12 at 1e-12); before, at 1 ms RK4 physics: 2e-14 at 1 ms, 1.9e-4, 3.3e-4 at 2 ms over all rows, RK4 stable to 9.8, 3.8 ms |

The sparse cart-pole and acrobot rows and reacher-hard share their measured row's
dynamics, box and clock. The windy pendulum and Acrobot-XK rows keep the physics step.
Every value divides its interval, so all substeps equal `model_step`. The contact
measurement (2026-11-10): rows of the state pool with one limited coordinate moved to its
limit, half past it, half inside within an interval's travel and moving toward it
(scratchpad `limits/step/`, whole-episode pools `limits/step_full/`; the free-state
measurement `continuous_ctrl/step/summary_raw.tsv`).
`tests/test_ph_model_step.py` keeps the measured bounds (free-state and limit-contact),
checks every row against them and checks the rule; a new port without joint limits
measures its bound (the oracle endpoint error above) and adds it there and to
`MODEL_STEPS`, or stays at the physics step; a new port with joint limits stays at the
physics step. Under `held` with `refresh_steps` 1 a larger `model_step` changes nothing.
Check (2026-11-10, implicit_learned_admissions, 1500 decisions, reduced replay/batch/
rollouts, 4 threads): the sampled drift error at every 250 decisions agreed to 0.2%
between the row's step and 1 ms (lqr_2_1 at 10 ms: 2.07 against 2.07 at the end;
cartpole-swingup at 5 ms, before its return to 1 ms: 6.95 against 6.95), and the run took
8.4x (lqr_2_1) and 4.4x (cartpole-swingup at 5 ms) less training time.

**Contact-triggered fine substepping** (top-level key and table column
`model_coarse_step`, since 2026-11-10; an empty cell is off). Few intervals touch a limit
under uniform decisions (0.1-0.5% on the cart-poles, 0.3-0.7% on the point mass, 1.3-1.9%
on the reacher), yet the physics step was needed for all of them. With a coarse step
`h_c` (`common.ph_config.MODEL_COARSE_STEPS`: the six cart-poles 5 ms and the point mass
10 ms, their free-state bounds; off on the reachers, whose 1 ms is the wrist's RK4
stability bound under stiff gains, and on the swimmers, whose model step is their native
step) every model integration of a decision interval (plant fit and its propagated
covariance, explicit rollout target, rate-head rollouts, gate paths, path Jacobian norms)
runs every row at `h_c` first and runs again at `model_step` (1 ms) the rows the detector
flags; their results replace the coarse ones (`models/ph_sac/rollouts.ContactSubstepping`).
The detector (`contact_flags`), on every coarse substep of length `h_c` and each limited
coordinate `i` of the state spec's `limited_position_ranges`:

    mu_i = 0.01 (high_i - low_i) + 2 h_c max(|qdot_i(start)|, |qdot_i(end)|)
    flag the row unless low_i + mu_i < q_i < high_i - mu_i at both ends of the substep

so a knot past the range, within the margin of an edge, or non-finite flags the row.
Margins: the cart's rail 36 mm plus 10 ms times the cart speed; the point mass's slides
5.8 mm plus 20 ms times the slide speed. Proof sketch: if within one coarse step the
coordinate's speed stays below twice its larger knot speed plus `m_i / h_c` (assumption
A), the path moves less than `mu_i` from the substep's start knot, which lies more than
`mu_i` inside the range, so it never reaches an edge; MuJoCo's limit acts only past the
range (margin 0), so the coarse integration of an unflagged row is limit-free, and the
RK4 stages, which move less than the path's step, never evaluate the limit force. A
crossing between two knots (a fast approach across the margin, a bounce within one step)
needs a knot within `mu_i` of the edge. The selection is per row (per path in rollouts)
and per decision interval, discrete (no gradient through the mask); gather and scatter
keep row order, so with no row flagged the result is the plain coarse integration's and
with every row flagged the plain fine one's, bitwise. Noise stays on the fine grid; a
coarse substep sums the fine increments it spans. `benchmarks/ph_contact_substep.py
contact <env_id>` measures the scheme's one-interval error against the environment on
whole-episode, limit-enriched and fast-approach pools, with the share of rows flagged and
the touching rows missed (must be zero); `tests/test_ph_contact_substep.py` tests it.
Cluster job 1135351 (2000 rows per pool): no touching row was missed on any pool or task;
the flagged rows were exact (p99 2e-15 to 5e-15, max 1.7e-14); on whole-episode pools
the scheme's p99 was 4.8e-4 (cartpole-swingup), 2.9e-4 (three_poles) and 1.1e-5
(point mass, max 2.5e-4, against 0.39 for the plain 10 ms step) with 0.9%, 0.6% and 13%
of rows flagged (0.2%, 0.2% and 2.1% touching); on limit-enriched pools (96-98% touching)
and fast approaches (66-68% touching, 3-8 m/s on the cart) every row was flagged and the
plain coarse step's p99 was 0.6 to 1.4. Packed timing (40 processes, 2 threads, dense
plant fit; early in training, where almost no plant-fit row was flagged): implicit 127,
123 and 100 ms per decision on cartpole-swingup, three_poles and point_mass against 349,
449 and 698 without the scheme; the cost moves toward the latter with the share of rows
pressed on a limit.

**Contact exclusion** (top-level key `contact_handling`, table column
`model_contact_handling`, `run_ph_sac.py --contact-handling exclude`; since 2026-11-10;
absent or `port`: the limit port). Since 2026-11-11 the six cart-pole rows and
`point_mass-easy` set `exclude` (`make_ct_ph_sac_table.CONTACT_HANDLING`; decided after the
`exclude_thin` experiment: the port's final scores at 3-5x less time per decision); the
reacher and swimmer rows keep the port. `--contact-handling port` on an `exclude` row
turns the limit port back on.
The limit port cannot represent MuJoCo's soft limit, which besides its spring-damper
cancels about 90-95% of every other force on the joint (held-out one-interval errors of
17-34% on intervals touching a limit, under 1% on free ones; a cart-pole plant's drift
error went from 2.4% to 14-20% once the early policy pushed the cart into the rail).
Under `exclude` the learned plant is the free mechanism (no limit port; validation
requires `network.limit_port` false) and is kept away from the limits:

- Rule (`models/ph_sac/rollouts.ContactExclusion`, `start_state_flags`; constants in
  `common.ph_config`: `EXCLUSION_MARGIN_FRACTION` 0.01, `EXCLUSION_VELOCITY_FACTOR` 1.5,
  `EXCLUSION_ACCELERATION_FACTOR` 2, `EXCLUSION_RULE_VERSION` 2). A replay row is flagged
  when at its start state a limited coordinate `i` (`limited_position_ranges`) is within

      mu_i = 0.01 (high_i - low_i) + max(v Delta, v Delta phi1(z) + a_i Delta^2 phi2(z)),
      v = 1.5 |qdot_i|,  z = k_i Delta,
      phi1(z) = (1 - e^-z) / z,  phi2(z) = (z - 1 + e^-z) / z^2  (1 and 1/2 at z = 0)

  of an edge or past it, with `a_i` twice the acceleration the motors alone give the
  coordinate at qpos0 (`MujocoPHEnv.motor_acceleration_bounds`, `sum_j |(M^-1 G)_ij| u_max_j`)
  and `k_i = d_i (M^-1)_ii` its damping rate there (`MujocoPHEnv.velocity_decay_rates`,
  `dof_damping` over the effective inertia), or when the environment recorded that the
  real interval was past a range at a physics knot (`Transition.limit_touched`, exact:
  MuJoCo's limit row exists exactly there). The travel term (`rollouts.reach_travel`) is
  the largest travel over `Delta` of the damped double integrator `qdd = s a_i - k_i qd`,
  `|s| <= 1`, from speed `v`: below the terminal speed `a_i / k_i` the full push toward the
  edge attains the second branch; above it the speed only falls and `v Delta` bounds it.
  Without damping it is `v Delta + a_i Delta^2 / 2`, the rule's first version (the
  experiment of 2026-11-10 afternoon, `exclude_thin`), which ignored the reacher wrist's
  damping (0.01 N m s/rad on an effective inertia of 4.8e-5 kg m^2 at the straight arm:
  `T = m / d` = 4.8 ms, terminal speed `a / k` = 13 rad/s with the factor 2) and flagged
  10% of warm-up rows while 0.7% touched. Margins at rest, version 1 -> 2 (cluster job
  1135791): reacher wrist (20 ms) 0.5966 -> 0.2547 rad of a +-2.79 rad range; cart rail
  (10 ms; `T` = 2055 s) 36.973 mm, unchanged to 2e-9 m; point-mass slides (20 ms; `T` =
  0.3 s) 5.933 -> 5.930 mm. On 20 uniform-decision episodes of reacher-easy the share of
  flagged intervals fell from 20.4% to 10.6% (2.0% touch), on 20 limit-seeking episodes
  from 68.4% to 61.7% (55.1% touch), with no touched interval missed by the start rule
  alone on any of the four tasks and two pools.
  Left over for the forces the model omits: the factor 2 in `a_i` (which also doubles the
  terminal speed; on the reacher the other half covers the shoulder's coupling, its
  damping through `M^-1` and the Coriolis and centripetal terms, up to joint speeds of
  about 8.6 rad/s at the straight arm and 16 rad/s near the wrist limit), the speed factor
  1.5 and the 1% of the range (gravity, the pose dependence of `M`; qpos0 has the largest
  decay rate and terminal speed on these tasks, checked on a pose grid by
  `tests/test_ph_exclusion_reach.py`). The derivation is in `docs/ct_ph_sac_layout.md`.
  The flag depends on the start state and the interval, not on whether this interval
  happened to turn away from the wall; the outcome-based part is a safety net, and
  `benchmarks/ph_contact_substep.py reach ENV_ID OUT` reports the share of touched
  intervals the start rule alone misses on uniform and limit-seeking episodes. The replay
  stores both flags in either mode. The rule's version is part of the resume fingerprint
  of `exclude` runs (`contact_exclusion_setting`), so a run started under version 1
  refuses to resume under version 2; `port` runs are unaffected (their stored
  `near_limit` diagnostic column follows the current rule).
- Excluded: flagged rows from the plant fit (warm fit included; the loss is the mean over
  unflagged rows, a batch of flagged rows only takes no step) and from the plant
  diagnostics; not from the reward-head fit.
- Fallbacks wherever the plant would be read at a flagged state: the implicit value
  residual uses the observed interval (`ph_resolvent.observed_interval_residuals`), the
  implicit rate head's rollout and generator targets become the data target of the
  observed interval (`ph_targets.excluded_rate_targets`), the explicit critic target is
  `r + (1-d) e^(-lambda Delta) Vbar(s')`, a flagged row is not rolled out, and a model path
  that comes within `0.01 (high - low) + 2 h |qdot|` of a limit is truncated at its last
  clean knot (the observed endpoint replaces it in the one-interval targets; the gate
  bootstraps there, and flagged gate rows carry no evidence). The formulas are in
  `docs/ct_ph_sac_layout.md`.
- Model step: every model integration at the row's `model_coarse_step` (cart-poles 5 ms,
  point mass 10 ms) with no fine pass, since the plant has no limit force; the reachers
  keep 1 ms (the wrist's RK4 stability bound). The swimmers stay on the port (40% of their
  intervals touch a limit).

**Swimmer physics** (since 2026-11-10). dm_control's `swimmer.xml` sets
`timestep="0.002"` and no integrator, so dm_control steps the swimmer with MuJoCo's default
Euler at 2 ms (15 steps per 30 ms control step), and no joint sets damping. The ports now
use that physics (`common/ph_dmc/swimmer.py`: `PHYSICS_DT`, `INTEGRATOR`; 1 ms RK4
before). MuJoCo's Euler step is velocity-first semi-implicit,
`qvel += h qacc`, `qpos += h qvel_new`, with `qacc` from the step-start forward pass
(the inertia-box fluid force and the soft joint limits' constraint force evaluated there),
and it treats joint damping implicitly (`M + h diag(dof_damping)`, the `eulerdamp` flag)
only for `dof_damping > 0`: zero on the swimmer, so the model's `semi_implicit` map at
`h = 2 ms` with the oracle drift is that step, to MuJoCo's constraint-solver tolerance.
Under `refresh_mode: continuous` the law is evaluated once per Euler step, at its start:
the environment runs its held path with a refresh at every physics step, which is what the
control callback would do at the step's single forward pass, and the semi-implicit map
reads the torque at the substep's start. The rows therefore set `model_integrator:
semi_implicit` and `model_step` 0.002, one drift evaluation per physics step (15 per
interval, against 120 at 1 ms RK4). The plant-fit covariance's Jacobian of the map,
`I + h K_h Db` with the closed-loop `dg/ds`, reads `plant.drift_jacobian`, which carries
`dD/dv` (`damping_velocity`) and the limit port. Stability at the boxes' maxima
(swimmer6: stiffness 0.0047746 N m/rad, damping 9.5493e-4 N m s/rad; swimmer15: 0.011937,
0.0023873) is measured by `benchmarks/ph_contact_substep.py swimmer <env_id>` (2000
intervals and 20 whole uniform-decision episodes: one-interval growth, non-finite and
exploding paths). Cluster job 1135351: over 2000 intervals from uniform-decision episodes
(joints past their ranges at 46% / 90% of the starts) the one-interval closed-loop
Jacobian's spectral radius is the same at 2 ms Euler and 1 ms RK4 (p99 1.25 and 1.33,
max 1.30 and 1.37, ratio about 1), the velocity growth too, and 20 whole episodes per
testbed stayed finite with rates below 5 rad/s; the 2 ms Euler endpoint differs from the
1 ms RK4 one by 1-2% of the interval's displacement (p50; max 3.8%). One caveat is
MuJoCo's own: Euler treats the quadratic fluid drag explicitly, which is unstable for a
link moving fast enough (drag rate 600 |u| /s on a 0.01 kg link against 2 / h =
1000 /s: |u| above about 1.7 m/s). Uniform decisions never got there (the motor is
+-5e-4 N m), but a start with joints 0.15 rad deep past their ranges and a hard stroke
did (1 of 16 test rows on swimmer15): MuJoCo flagged the acceleration
(`mjWARN_BADQACC`) and reset, and the model's map diverged on the same row.

**Cyclic positions** (`StateSpec["cyclic_position_indices"]`, default empty). Observed
positions, context included, on which no force depends. The learned plant
(`PortHamiltonianPlant`) leaves them out of the features of all its networks, so its
`U`, `M`, `D` and diffusion are exactly independent of them, while the actor, critics,
value, rate and reward networks still read them. The swimmer declares its root position
`x, y` and its target: the reward needs the absolute nose against the absolute target,
but the forces are translation invariant, and a plant fed `x, y` would have to learn that
and would extrapolate as the swimmer swims into unvisited ground. Unobserved positions
(not in `P q`) are cyclic already; this key covers observed ones.

**Velocity-dependent damping** (network key `damping_velocity`, table column
`model_damping_velocity`, default False). The learned damping factor reads the scaled
velocities `v / velocity_feature_scale` besides the position features,
`D(y, v) = L(y, v) L(y, v)^T`. The dissipation `v^T D v` stays nonnegative, and
`drift_jacobian` adds the `dD/dv` terms (`dL_k L^T v + L dL_k^T v` per velocity
direction). MuJoCo's fluid force (`density`, `viscosity` options, the inertia-box model of
`mj_passive`) is `-D_fl(q, v) v` with `D_fl` linear in `|v|`, outside `D(y)`; the swimmer
rows set the key, every other row leaves the cell empty, which keeps the plant they had.
The key goes into `OptionalNetworkFields` (`common/ph_config.py`), because the
`TypedDict`s there count `NotRequired` keys as required.

**Soft joint limits** (`joint_limits.py`, since 2026-11-09). A port keeps a dm_control
joint limit as MuJoCo models it, never as a failure boundary: the swimmer's joints (since
2026-11-09), the cart-pole slider (+-1.8 m, `solreflimit 0.08 1`), the reacher wrist
(+-160 deg) and the point mass's slides (+-0.29 m), the last two with MuJoCo's default
`solref` and `solimp` (since 2026-11-10). Keep the
joint `limited` in the MJCF with dm_control's `range`, `solreflimit` and `solimplimit`
(MuJoCo's defaults `0.02 1` and `0.9 0.95 0.001 0.5 2` apply where the XML omits them;
the ports write all seven values from `plant_parameters`, the keys
`common.ph_dmc._common.LIMIT_KEYS`, through `_common.limit_attributes`, plus the range:
`cart_range`, `wrist_range_degrees` (in degrees, as dm_control's XML), `slide_range`;
the compiled `jnt_range`, `jnt_solref`, `jnt_solimp`, `jnt_margin` and `dof_invweight0`
equal dm_control's bitwise), leave the constraint solver on, and in the oracle compute the
smooth acceleration first,
then call `constrained_accelerations(joint_limits_from_model(model, joint_names),
mass_factor, qacc_smooth, positions, velocities)["qacc"]` (torch; `constrained_accelerations_numpy`
for one state in numpy). It reproduces `mj_forward`'s `qacc` and `qfrc_constraint` with the
default Newton solver to rounding (1e-10 relative in the tests), including several limits
at once and limits that let go (`f >= 0`): per violated side a row with `pos` the
penetration, impedance `d` from `solimp` (with `d0` and `dmax` clipped to [1e-4, 0.9999]
first), `aref = -b vel - k d pos` from `solref` (`b = 2 / (dmax tc)`,
`k = 1 / (dmax^2 tc^2 zeta^2)`, `tc = max(timeconst, 2 timestep)`), `R = (1 - d) / d *
dof_invweight0`, and the complementarity problem `(A + R) f - (aref - J qacc_smooth) >= 0`,
`f >= 0`, solved by a fixed number of principal-pivoting passes (torch.func-safe). Check a
port with `mujoco_limit_rows(model, data, limits)`, which reads MuJoCo's `efc_*` rows of
the same joints (`tests/test_ph_dmc_swimmer.py` shows how; compare forces, not
`efc_state`, which MuJoCo does not update after its last Newton step). Inside `mj_step`
the solver is warm-started and stops at `opt.tolerance` (1e-8, dm_control's default, kept),
so with limits active a 30 ms RK4 interval of the environment and of the oracle differ by
up to about 5e-9 (1e-12 with `opt.tolerance` 1e-12) on the swimmer's coupled rows, and to
rounding (at most 1.8e-15 over 64 intervals through the limits) on the cart-poles, reachers
and point mass, whose single rows the Newton step solves exactly; single `mj_forward` calls
agree to rounding (efc rows, `qfrc_constraint` and `qacc` to 1.7e-14 relative on 300-400
states each, 60% past a range). The task then has no failure
set for that stop, and the row sets the learned plant's limit port. Inside every range no
limit row exists, so the environment and the oracle are bitwise those of the model without
the limit (`tests/test_ph_dmc_joint_limits.py`). MuJoCo floors the time constant at
`2 x timestep` (`refsafe`): at our 1 ms step the compiled `0.02` acts as 0.02 s, but in
dm_control's 20 ms reacher and point-mass models it acts as 0.04 s, so the same compiled
limit is about twice as stiff at our step (the cart-pole's 0.08 s is above the floor at
both steps); this is a consequence of the physics step, not of the model. At dm_control's
own step our models reproduce dm_control's trajectories into the limits (bitwise).
With the limits in the dynamics (re-measured 2026-11-10) the cart-poles' 5 ms and the
reacher's 1 ms still met the 1e-3 criterion on the uniform-box rows, and the point mass's
20 ms met it on starts from 3 s rollouts (0.2% of rows touch a slide, p99 3.5e-5) but not
on starts from whole 20 s episodes (2% touch, p99 3.0e-3 at 20 ms, 2.6e-4 at 10 ms); on
rows that touch a limit only the 1 ms physics step stays within 1e-3 (p99 0.3 to 0.65 at
2 ms): the limit's damping switches on with the row (impedance 0.9 at zero penetration), a
jump that a longer RK4 step straddles (docs/dmc_environments, "Kept joint limits"). Hence
every task with joint limits integrates at the physics step (**Model step** above).

**Limit port start.** The port starts at the compiled limit, not at the plant's generic
(1e-2, 1e-3, 0.1), which at the training learning rate stayed inert (0.02 N/m after 2000
warm-fit steps against the 160 and 750 N/m the limits need). The factory compiles the
task's model, reads its limits (`joint_limits_from_model`) and stores
`StateSpec["limit_port_init"]`, one `(k, c, w)` per entry of `limited_position_ranges`
(`common.ph_dmc._common.with_limit_port_init`, which also checks that each entry's
position is the joint's dof and its range the compiled one), from
`joint_limits.limit_port_init`. Derivation: with one active row and the joint's inverse
inertia `A` at its qpos0 value `dof_invweight0`, `R = (1 - d)/d * dof_invweight0` gives
`A + R = dof_invweight0 / d` and, along the joint, `J qacc = (1 - d) J qacc_smooth +
d aref` with `aref = -b vel + k d r` (`r` the penetration). In force units the limit is a
spring `k d^2 / dof_invweight0` and a damper `b d / dof_invweight0`, gated by the
impedance, plus a term cancelling the fraction `d` of every other force on the joint (the
added inertia, which no potential or damping reproduces). At full impedance `d = dmax`,
with `b = 2 / (dmax tc)` and `k = 1 / (dmax^2 tc^2 zeta^2)`:

| Port parameter | Start | cart-pole slider | reacher wrist | point-mass slides | swimmer joints |
| --- | --- | --- | --- | --- | --- |
| stiffness `k` | `1 / (dof_invweight0 tc^2 zeta^2)` | 160.5 N/m (161.0 two, three poles) | 0.1205 N m/rad | 750 N/m | 2.7e-3 to 3.7e-3 N m/rad |
| damping `c` | `2 / (dof_invweight0 tc)` | 25.7 N s/m | 4.82e-3 N m s/rad | 30 N s/m | 2.7e-4 to 3.7e-4 N m s/rad |
| width `w` | `1e-3 + W (1 - d0 / dmax)` | 1.053e-3 m | 1.053e-3 rad | 1.053e-3 m | 0.1010 rad |

that is the joint's own inertia on a spring-damper of time constant `tc` and damping ratio
`zeta`, which is what solref means, with MuJoCo's floor `tc = max(solref[0], 2 timestep)`
kept: at our 1 ms step the reacher's and point mass's 0.02 s stays 0.02 s (at dm_control's
20 ms step it would act as 0.04 s, a quarter of the stiffness). The impedance rises from
`d0` to `dmax` over solimp's width `W`, while the port's stiffness and damping rise from 0
over its width, which cannot go below the plant's floor 1e-3: the start is the floor plus
the part of `W` over which the impedance still rises (`d0`, `dmax` clipped as MuJoCo
clips them; the swimmer's `d0 = 0` acts as 1e-4). The prediction holds: a high-rate fit
from the generic start came to 164 N/m (cart-pole) and 746 to 764 N/m (point mass); from
the compiled start at the training rate 3e-4 the error past the limits after 800 steps is
2.7e-4 on the point mass (5.9e-4 from the generic start at 5e-2, 5.2e-2 without the port)
and 5.65e-3 on the cart-pole at a tenth of the motor force (5.64e-3, 7.3e-3), where both
reach the error of the added-inertia term (`tests/test_ph_dmc_joint_limits.py`). Inside
the ranges the port is exactly zero whatever its start, so the plant there is bitwise the
same. A state spec without `limit_port_init` (configs and records before 2026-11-10)
starts at the generic constants.

**Limit port** (network key `limit_port`, table column `model_limit_port`, default False;
`StateSpec["limited_position_ranges"]` = `((position index, low, high), ...)`, set by the
factory, with the start `StateSpec["limit_port_init"]` above). The learned plant adds, for each listed coordinate, a potential and a damping
that are exactly zero inside `[low, high]`: with penetration `r` and `x = r / w`,
`V = k w^2 Pi(x)` (stiffness ramping from 0 to `k` over the width `w` by a smoothstep) and
damping `c sigma(x)`, with learned `k, c >= 0` and `w > 1e-3` per coordinate. A soft
limit's force is zero inside the range and turns on within its impedance width beyond it
(0.1 rad on the swimmer, 0.001 with MuJoCo's default `solimp`), a kink at a known place
that the plant's smooth tanh networks of the position features would have to build from a
few states past the range. MuJoCo's constraint also adds the constraint mass `D` to the
joint's inertia while active (it cancels part of every other force on the joint); the
port has no such term, so it approximates the limit, and the fit is the judge.

**Failure set and payoff.** No dm_control port has a failure set (since 2026-11-10; the
cart-pole rail `|x| >= 1.8 m`, the reacher wrist `|theta_w| >= 160 deg` and the point-mass
box `max(|x|, |y|) >= 0.29 m` were failure boundaries with payoff 0 and the lift until
then, and a config with those `failure_limits` is now rejected by
`common.ph_dmc._common.require_no_failure_set`). A joint stop stays MuJoCo's soft limit
(above): `failed` is False, the mask is `limit_failure_mask(states, NO_FAILURE)` (all
False), the payoff `failure_rate_payoff` (0) is never charged, and
`limit_boundary_functions(payoff, NO_FAILURE)` gives `H = payoff`, `B = 1`. The helpers
for `(state index, limit)` pairs (`limit_failure_mask`, `limit_failed`,
`limit_boundary_functions`) remain for the in-house tasks with failure sets (the windy
pendulum, Acrobot-XK).

**Upright.** `upright(config, state, tube)`: every link's absolute angle is within
`tube` of upright. Tasks without an upright pose (reacher, point mass, lqr) leave it as
`None` or give their success predicate, and say which.

**Reward range.** `reward_rate_bounds(config) -> (low, high)` declares the range of the
reward rate. The implicit agent's learned reward head (`models/ph_sac/reward.py`) then
outputs `low + (high - low) * sigmoid(z)`; with `None` it stays linear. A task whose
reward is a product of tolerances lies in [0, 1] and passes
`_common.unit_interval_bounds` (pendulum, cart-pole, reacher, point mass; the pendulum
row's dense margin-2 reward, 0.102 hanging to 1 upright, included). lqr's
`1 - cost` and acrobot_swingup's Lyapunov reward `-log(max(V / V_down, reward_floor))`
(at most `-log(reward_floor)` = 13.8, unbounded below, logarithmically in the energy)
are unbounded, so both declare `None` and keep a linear head; acrobot_swingup_sparse
shares acrobot's registration and declares `None` as well. The five later cart-pole
tasks declare `cartpole_task_reward_rate_bounds`, which returns (0, 1) for their
tolerance rewards and `None` (a linear head) when the reward_parameters select an
optional Lyapunov reward.
Declare a range when you know one. On the reacher a linear head learned a
negative reward inside the wrist limit, so the stop (then a failure boundary with payoff
0, `B = 0`) became the value maximum and the policy drove into it.

## Config factories

They go in `common/ph_dmc/<domain>.py`. `common` must not import `environment`, and
`common.ph_config` imports `common.ph_dmc` (to re-export `cartpole_swingup_config`), so
import `common.ph_config` there only under `TYPE_CHECKING`. The returned
`TestbedConfig` must reproduce the row's `task_*` cells exactly. These are checked by
`benchmarks.ph_hyperparams.declared_testbed_mismatches(entry.task, config) == []` and
at protocol build. The helpers in `common/ph_dmc/_common.py` reproduce every row's
cells: `tests/test_ph_dmc.py::TestConfigHelpers` builds all twelve stub rows' configs
with them.

```python
return dmc_testbed_config(
    name="pendulum_swingup",
    state_spec=dmc_state_spec(physical_dim=1, action_dim=1, periodic=(0,), actuated=(0,)),
    reference_period=0.02, episode_duration=20.0,
    controller=box_rule_controller(offsets=(math.pi,), torque_limit=1.0,
                                   setpoint_mode="absolute", setpoint_error="wrapped"),
    plant_parameters=plant_parameters, failure_limits=failure_limits,
    reward_parameters=reward_parameters, reset_parameters=reset_parameters,
)
```

Check with `python benchmarks/run_ph_sac.py --list`, where your rows should read
`builds`, and with `run_ph_sac.load_protocol_from_table("<env_id>")`.

## Environment class

Subclass `MujocoPHEnv`, following `dmc/cartpole.py::CartpoleSwingupEnv`:

- `__init__(config, *, decision_mode="feedback", seed=None)`: check the keys with
  `_common.require_keys`, build the MJCF string from `plant_parameters`, then call
  `super().__init__(config, model, validated_state_spec(config), ...)`. The MJCF
  disables contact and constraints, sets the timestep and integrator from the config,
  and uses motors with gear 1 and `ctrlrange` equal to the torque limits.
- `_reset_state(options)`: honour `options={"state": s}`, otherwise draw dm_control's
  reset from `self._reset_rng`.
- `reward_rate(state, torque)`, `failed(state)`, `terminal_payoff(state)`.
- `oracle_drift(state, torque)` (numpy, for example `chain.chain_drift_numpy`) and
  `oracle_diffusion(state)` (zeros `(D, 1)`).

## chain.py

`PlanarChain(link_masses, link_com_distances, link_inertias, link_lengths,
joint_damping, input_matrix, gravity=0.0, cart_mass=None, cart_damping=0.0)`. Per link
i: mass, distance from joint i to its centre of mass along the link, inertia about the
centre of mass about the joint axis, length from joint i to joint i+1, and viscous
damping on the relative rate. `input_matrix` is `G` (`nv x m`), from inputs to
generalized forces. `cart_mass=None` is a fixed base. Coordinates are
`q = (x, theta_1, ..., theta_n)` with a cart, else `(theta_1, ..., theta_n)`. The state
is `(q, context, qd)`.

| torch (batched, `torch.func`-safe) | numpy | returns |
| --- | --- | --- |
| `chain_drift(chain, states, torques, *, context_dim=0)` | `chain_drift_numpy` | `(qd, 0, qdd)` `[..., 2nv+context_dim]` |
| `chain_accelerations(chain, positions, velocities, torques)` | `chain_accelerations_numpy` | `qdd [..., nv]` |
| `chain_mass_matrix(chain, positions)` | `chain_mass_matrix_numpy` | `M(q) [..., nv, nv]` |
| `chain_potential_energy(chain, positions)` | `chain_potential_energy_numpy` | `g sum_j h_j cos phi_j [..., 1]` |
| `chain_energy(chain, positions, velocities)` | `chain_energy_numpy` | `T + U [..., 1]` |
| `chain_points(chain, positions)` | `chain_points_numpy` | `(side, up)` of each joint and the tip `[..., n+1, 2]` |
| `chain_callbacks(chain, *, context_dim=0)` | | `(drift, diffusion)` for `OraclePlant` |

The torch solve uses Cholesky. **Do not use `torch.linalg.solve` (or `lu_solve`) in
oracle callbacks.** In torch 2.12 their forward-mode derivative under
`torch.func.vmap` is wrong, and `OraclePlant.drift_jacobian` is
`vmap(jacfwd(drift))`. Use `torch.linalg.cholesky` with `torch.cholesky_solve`, or a
closed form. `_common.planar_chain_from_model(model, link_bodies, *, along, joint_axis,
cart_body=None, last_length=0.0)` builds a `PlanarChain` from a compiled MuJoCo model:
ours, or dm_control's `physics`. Use `along=2, joint_axis=1` for dm_control's vertical
models and `along=0, joint_axis=2` for the reacher. `body_inertials(model, names)` reads
masses, centres of mass and inertias, merging welded (jointless) children such as the
reacher's finger. The chain is tested against the analytic cart-pole and Acrobot-XK
callbacks to 1e-10, against `env.mujoco_drift`, and against dm_control's compiled
pendulum, acrobot, three-pole cart-pole and reacher to 1e-8 (`tests/test_ph_dmc.py`).

## Validation each port must pass

Put the tests in `tests/test_ph_dmc_<domain>.py`. dm_control is importable in the test
venv (`from dm_control import suite`). Load with
`env = suite.load(domain, task, task_kwargs={"random": 0})` and use `env.physics`
(`physics.model.ptr` is the raw `mujoco.MjModel`) and `env.task`. Our torque `tau`
corresponds to dm_control's `ctrl = tau / gear`.

1. **Oracle vs MuJoCo**: as `tests/test_ph_env.py::TestOracleAgainstMujoco.check`. On
   random states (inside the failure limits of an in-house task; past the kept joint
   limits too, with MuJoCo's `efc_*` rows compared, for a port that keeps them),
   `env.oracle_drift`, the tensor `oracle_callbacks` drift and `env.mujoco_drift` agree
   to 1e-6 relative (1e-10 in the dm_control port tests).
2. **Our MuJoCo vs dm_control's forward dynamics**: set `physics.data.qpos`, `qvel`
   and `ctrl` (the context is not in dm_control's qpos), call `physics.forward()`, and
   compare `physics.data.qacc` with `env.mujoco_drift(state, tau)[position_dim:]`, with
   states past the kept joint limits too (`tests/test_ph_dmc_joint_limits.py`).
3. **Reward vs `task.get_reward(physics)`**: on random states and controls, compare
   `env.reward_rate(state, tau)` with dm_control's reward after
   `physics.forward()`. The reacher must also set the target geom's position and
   `geom_size` to the episode's values. For indicator rewards, keep samples away from
   the boundary or count disagreements, which must be zero. A row whose reward departs
   from dm_control's task compares a config with dm_control's reward parameters and
   tests its own against the definition: the pendulum (margin 2 since 2026-11-09)
   compares margin 0 with `get_reward` and margin 2 with `rewards.tolerance`.
4. **Trajectory vs dm_control's simulation**: with the config's `physics_dt` and
   `integrator` overridden to dm_control's for this check (pendulum 0.02 Euler, acrobot
   and cart-pole 0.01 RK4, reacher and point mass 0.02 Euler, lqr 0.03 Euler, swimmer
   0.002 Euler), start
   both from the same state, apply the same control sequence (`decision_mode="torque"`
   and `env.step_dt(tau, physics_dt)` against `physics.data.ctrl[:] = tau / gear;
   physics.step()`), and compare the states over a few hundred steps, through the kept
   joint limits too.
5. **Tensor/numpy agreement**: as `tests/test_ph_env.py::TestTaskTensors.check_agreement`.
   Check `reward_rate_tensor` against `env.reward_rate` (to 1e-10),
   `failure_mask_tensor` against `env.failed` (with some states outside the limits; all
   False on a dm_control port, which has no failure set), and `terminal_payoff_tensor`
   against `env.terminal_payoff`. The boundary factor must vanish on every finite failure
   limit (and is 1 without a failure set).
6. **torch.func**: `make_oracle_plant(env).drift_jacobian` matches central finite
   differences of the drift, as in `tests/test_ph_dmc.py`.
7. **Factory vs table**: `load_protocol_from_table(env_id)` builds, and `--list` reads
   `builds`.
8. **Reset law**: the start-state statistics match dm_control's reset; for example,
   compare means and ranges over a few hundred resets.
9. **Context (reacher)**: the drift is zero on the context, the context stays constant
   over an episode, `options={"state": s}` and `snapshot`/`restore` carry it, and the
   reward reads it.
10. **Model integrator vs environment**: with the oracle plant, the row's
   `model_integrator` at `model_step = physics_dt` (`integrate_endpoint_moments`, and
   `stochastic_rollout` without noise) reproduces `env.step_dt(decision, duration)` to
   1e-10 in the wrapped residual, on random states and decisions including damping near
   the box maximum. `tests/test_ph_model_integrator.py` does this for reacher-easy,
   cartpole-swingup and acrobot-swingup.

Run the whole suite one process at a time:

```bash
python -m unittest tests.test_ph_env tests.test_ph_models tests.test_ph_algorithms \
  tests.test_ph_evaluation tests.test_ph_hyperparams tests.test_ph_baseline \
  tests.test_ph_operators tests.test_ph_replay tests.test_ph_theory tests.test_ph_dmc \
  tests.test_ph_model_integrator tests.test_ph_dmc_swimmer tests.test_ph_dmc_joint_limits \
  tests.test_ph_dmc_<domain>
python benchmarks/run_ph_sac.py --list
```
