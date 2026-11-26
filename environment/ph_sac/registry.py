"""Registry of CT-PH-SAC testbeds: one TestbedSpec per testbed name.

Every name-based lookup goes through this registry: environment construction
(tasks.make_env), the tensor task functions the models read (tasks.reward_rate_tensor,
failure_mask_tensor, terminal_payoff_tensor, terminal_boundary_functions), the oracle
plant (tasks.oracle_callbacks, make_oracle_plant), the evaluation's upright test and the
protocol builder (benchmarks/run_ph_sac.py, which derives its testbed settings and
factories from the registered specs).

Registrations happen at import: environment/ph_sac/tasks.py registers the in-house
testbeds (windy_pendulum, acrobot_xk) and the package environment/ph_sac/dmc registers
the dm_control ports, one module per domain. testbed_spec and registered_testbeds import
both first, so a lookup never depends on what the caller happened to import.

A spec with implemented=False is a stub: its row of the hyperparameter table parses and
its settings keys are known, but its environment, factory and task functions raise
NotImplementedError naming the testbed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Tuple, Type

from .base import MujocoPHEnv

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor

    from common.ph_config import TestbedConfig

# (drift(states [B,D], torques [B,m]) -> [B,D], diffusion(states [B,D]) -> [B,D,K])
OracleCallbacks = Tuple[Callable[["Tensor", "Tensor"], "Tensor"], Callable[["Tensor"], "Tensor"]]
# (H(states) -> [B,1] payoff lift, B(states) -> [B,1] vanishing on the failure boundary)
BoundaryFunctions = Tuple[Callable[["Tensor"], "Tensor"], Callable[["Tensor"], "Tensor"]]


@dataclass(frozen=True, kw_only=True)
class TestbedSpec:
    """Everything the suite needs to know about one testbed, keyed by its name.

    name                        testbed name: TestbedConfig["name"], the table's env_testbed
    env_class                   MujocoPHEnv subclass, called as
                                env_class(config, decision_mode=..., seed=...)
    config_factory              builds the TestbedConfig from the testbed_settings keyword
                                arguments (the table row's env_* columns)
    settings_keys               required testbed_settings keys, in the factory's order
    optional_settings_keys      testbed_settings keys the factory accepts but does not need
    reward_rate_tensor          (states [B,D], torques [B,m], config) -> [B,1]; must equal
                                env.reward_rate row by row
    failure_mask_tensor         (states [B,D], config) -> bool [B]; must equal env.failed
    terminal_payoff_value       (config) -> float, the failure boundary payoff F; must equal
                                env.terminal_payoff (0 for a task without a failure set)
    terminal_boundary_functions (config) -> (H, B) with V = H + B * network: H the payoff,
                                B > 0 inside the failure limits and 0 on them
    oracle_callbacks            (config) -> (drift, diffusion) torch callbacks in the state
                                layout, compatible with torch.func (see plant.OraclePlant)
    implemented                 False for a stub: its row parses but does not build
    upright                     optional (config, state ndarray, tube) -> bool, whether every
                                link is within tube radians of upright, for the evaluation's
                                upright_seconds; None counts no state as upright
    reward_rate_bounds          optional (config) -> (low, high), the known range of the
                                reward rate (dm_control's rewards give (0, 1)); a learned
                                reward head then maps a sigmoid onto this range; None
                                (unknown or unbounded range) keeps a linear output, and so
                                does a callable that returns None for a config (a task
                                whose reward_parameters can select an unbounded reward:
                                the cart-pole tasks' optional Lyapunov rewards)
    physical_change             optional (plant_parameters, kind, scale) -> the changed
                                plant_parameters of the adaptation experiment's robot
                                (common.ph_dmc._common.with_physical_change, kinds
                                "length" and "damping"); a testbed with one accepts the
                                testbed_settings key physical_change (benchmarks/
                                run_ph_sac.py _build_testbed), which is not a table column
                                and not among optional_settings_keys. None: no change
    discount_setting            True: the testbed accepts the testbed_settings key
                                discount_rate, a run's discount rate per second in place of
                                the factory's (common.ph_dmc._common.with_discount_rate;
                                benchmarks/run_ph_sac.py _build_testbed), which is not a
                                table column and not among optional_settings_keys
    """

    name: str
    env_class: Type[MujocoPHEnv]
    config_factory: Callable[..., "TestbedConfig"]
    settings_keys: Tuple[str, ...]
    optional_settings_keys: Tuple[str, ...] = ()
    reward_rate_tensor: Callable[["Tensor", "Tensor", "TestbedConfig"], "Tensor"]
    failure_mask_tensor: Callable[["Tensor", "TestbedConfig"], "Tensor"]
    terminal_payoff_value: Callable[["TestbedConfig"], float]
    terminal_boundary_functions: Callable[["TestbedConfig"], BoundaryFunctions]
    oracle_callbacks: Callable[["TestbedConfig"], OracleCallbacks]
    implemented: bool = True
    upright: Optional[Callable[["TestbedConfig", "ndarray", float], bool]] = None
    reward_rate_bounds: Optional[
        Callable[["TestbedConfig"], Optional[Tuple[float, float]]]
    ] = None
    physical_change: Optional[
        Callable[[Mapping[str, float], str, float], Mapping[str, float]]
    ] = None
    discount_setting: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A testbed needs a nonempty string name.")
        if not (isinstance(self.env_class, type) and issubclass(self.env_class, MujocoPHEnv)):
            raise TypeError(f"{self.name}: env_class must subclass MujocoPHEnv.")
        for label in ("settings_keys", "optional_settings_keys"):
            keys = getattr(self, label)
            if not isinstance(keys, tuple) or not all(isinstance(k, str) for k in keys):
                raise TypeError(f"{self.name}: {label} must be a tuple of strings.")
        overlap = set(self.settings_keys) & set(self.optional_settings_keys)
        if overlap:
            raise ValueError(f"{self.name}: keys both required and optional: {sorted(overlap)}.")
        for label in (
            "config_factory",
            "reward_rate_tensor",
            "failure_mask_tensor",
            "terminal_payoff_value",
            "terminal_boundary_functions",
            "oracle_callbacks",
        ):
            if not callable(getattr(self, label)):
                raise TypeError(f"{self.name}: {label} must be callable.")
        if self.reward_rate_bounds is not None and not callable(self.reward_rate_bounds):
            raise TypeError(f"{self.name}: reward_rate_bounds must be callable or None.")
        if self.upright is not None and not callable(self.upright):
            raise TypeError(f"{self.name}: upright must be callable or None.")
        if self.physical_change is not None and not callable(self.physical_change):
            raise TypeError(f"{self.name}: physical_change must be callable or None.")
        if not isinstance(self.discount_setting, bool):
            raise TypeError(f"{self.name}: discount_setting must be True or False.")


_REGISTRY: Dict[str, TestbedSpec] = {}
_BUILTINS_LOADED = False


def register_testbed(spec: TestbedSpec, *, replace: bool = False) -> TestbedSpec:
    """Add spec under spec.name and return it; a second registration of a name raises.

    replace=True overwrites an existing entry (for tests that register a toy testbed).
    A domain module replaces its stub by editing its own register_testbed call, not by
    registering the name twice.
    """
    if not isinstance(spec, TestbedSpec):
        raise TypeError("register_testbed needs a TestbedSpec.")
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"Testbed {spec.name!r} is already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def _load_builtin_testbeds() -> None:
    """Import the modules that register the in-house testbeds and the dm_control ports."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    importlib.import_module("environment.ph_sac.tasks")
    importlib.import_module("environment.ph_sac.dmc")
    _BUILTINS_LOADED = True


def testbed_spec(name: str) -> TestbedSpec:
    """The registered spec of a testbed; an unknown name raises ValueError."""
    _load_builtin_testbeds()
    try:
        return _REGISTRY[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown testbed {name!r}; registered testbeds: {sorted(_REGISTRY)}."
        ) from None


def registered_testbeds(*, implemented_only: bool = False) -> Tuple[str, ...]:
    """Sorted names of every registered testbed, or only of the implemented ones."""
    _load_builtin_testbeds()
    return tuple(
        sorted(
            name
            for name, spec in _REGISTRY.items()
            if spec.implemented or not implemented_only
        )
    )
