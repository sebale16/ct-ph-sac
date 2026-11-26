"""Port-Hamiltonian and oracle plant models in observation coordinates.

States are s=(y,v) with y=Pq; forces depend on q only through y, so the derivatives
in q are P^T times those in y and unobserved positions are cyclic. Second derivatives
are exact tangents pushed through the perceptrons as ordinary tensor operations, so
the drift and its Jacobian work under torch.no_grad(), compose with torch.func, and
backpropagate to the parameters through every model step.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import func, nn

from common.ph_types import context_position_indices

if TYPE_CHECKING:
    from torch import Tensor

    from common.ph_config import NetworkConfig
    from common.ph_types import StateSpec

# Values of the network config key "diffusion": a learned noise factor, or none at all.
DIFFUSION_MODES = ("learned", "none")
# The limit port's initial stiffness k, damping c and ramp width w when the state spec
# gives no limit_port_init (every config and record before 2026-11-10), and the width's
# floor. k and c are exp of their parameters, so an optimizer step changes them by a
# factor; still, at the training learning rate (3e-4) these constants stay decades from a
# real limit (about 160 N/m on the cart-pole's rail, 750 N/m on the point mass's slides)
# and leave the port inert, so the dm_control factories give limit_port_init, the values
# of the compiled limit (environment/ph_sac/dmc/joint_limits.limit_port_init).
LIMIT_STIFFNESS_INIT = 1e-2
LIMIT_DAMPING_INIT = 1e-3
LIMIT_WIDTH_INIT = 0.1
LIMIT_WIDTH_FLOOR = 1e-3


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _limit_port_start(
    initial: Sequence[Sequence[float]], count: int
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Check StateSpec limit_port_init, one (k, c, w) per limited range with k > 0, c > 0,
    w > LIMIT_WIDTH_FLOOR, all finite; return its stiffnesses, dampings and widths."""
    rows = [tuple(float(value) for value in entry) for entry in initial]
    if len(rows) != count or any(len(row) != 3 for row in rows):
        raise ValueError(
            "limit_port_init needs one (stiffness, damping, width) per limited_position_ranges entry."
        )
    for stiffness, damping, width in rows:
        finite = all(math.isfinite(value) for value in (stiffness, damping, width))
        if not (finite and stiffness > 0.0 and damping > 0.0 and width > LIMIT_WIDTH_FLOOR):
            raise ValueError(
                "limit_port_init needs finite stiffness > 0, damping > 0 and width > "
                f"LIMIT_WIDTH_FLOOR ({LIMIT_WIDTH_FLOOR}), not {(stiffness, damping, width)}."
            )
    stiffness, damping, width = zip(*rows)
    return stiffness, damping, width


def diffusion_is_learned(config: NetworkConfig) -> bool:
    """Read network["diffusion"]: "learned" (the default) or "none" for a drift-only plant."""
    mode = config.get("diffusion", "learned")
    if mode not in DIFFUSION_MODES:
        raise ValueError(
            f"network diffusion must be one of {DIFFUSION_MODES}, not {mode!r}."
        )
    return mode == "learned"


class Plant(nn.Module):
    """Shared differentiable model interface used by both target plants and oracle arms."""

    state_spec: StateSpec
    observation_projection: Tensor

    def __init__(self, state_spec: StateSpec) -> None:
        super().__init__()
        self.state_spec = state_spec
        self.position_dim = int(state_spec["position_dim"])
        self.velocity_dim = int(state_spec["velocity_dim"])
        self.action_dim = int(state_spec["action_dim"])
        self.state_dim = self.position_dim + self.velocity_dim
        projection = np.asarray(state_spec["observation_projection"], dtype=np.float64)
        self.register_buffer(
            "observation_projection",
            torch.as_tensor(projection, dtype=torch.get_default_dtype()),
        )

    @property
    def noise_dim(self) -> int:
        """Column count K of the diffusion factor."""
        raise NotImplementedError

    def drift(self, states: Tensor, actions: Tensor) -> Tensor:
        raise NotImplementedError

    def diffusion(self, states: Tensor) -> Tensor:
        """Return rectangular physical Sigma, allowing velocity-only or zero-rank oracle noise."""
        raise NotImplementedError

    def covariance(self, states: Tensor) -> Tensor:
        """Return physical A=Sigma Sigma^T; do not assume the true A is invertible."""
        factor = self.diffusion(states)
        return factor @ factor.mT

    def input_matrix(self) -> Tensor:
        raise NotImplementedError

    def drift_jacobian(
        self, states: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return b [B,D], db/ds [B,D,D] and db/da [B,D,m] at the same point."""

        def row(state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
            value = self.drift(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
            return value, value

        jacobians, values = func.vmap(
            func.jacfwd(row, argnums=(0, 1), has_aux=True)
        )(states, actions)
        return values, jacobians[0], jacobians[1]

    def frozen_copy(self) -> Plant:
        """Detach a copy for a fixed operator or target; never step its optimizer."""
        frozen = copy.deepcopy(self)
        frozen.requires_grad_(False)
        frozen.eval()
        return frozen


class _TangentMLP(nn.Module):
    """Tanh perceptron that also maps first and second input tangents, to second order.

    With k input directions, dx has shape [B,k,in] and ddx has shape [B,k,k,in].
    """

    def __init__(self, inputs: int, hidden: Sequence[int], outputs: int) -> None:
        super().__init__()
        sizes = (inputs, *hidden, outputs)
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def initialize_output(self, bias: Tensor) -> None:
        """Shrink the final weights so the untrained output sits at the given bias."""
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.01)
            self.layers[-1].bias.copy_(bias)

    def forward(
        self, x: Tensor, dx: Optional[Tensor] = None, ddx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if dx is not None:
                dx = dx @ layer.weight.T
            if ddx is not None:
                ddx = ddx @ layer.weight.T
            if i < last:
                x = torch.tanh(x)
                slope = 1.0 - x.square()
                if ddx is not None:
                    outer = dx.unsqueeze(2) * dx.unsqueeze(1)
                    ddx = (
                        slope[:, None, None] * ddx
                        - 2.0 * (x * slope)[:, None, None] * outer
                    )
                if dx is not None:
                    dx = slope[:, None] * dx
        return x, dx, ddx


class PortHamiltonianPlant(Plant):
    """Learned U, M, D, free G, and diffusion with smooth position features.

    With learn_diffusion=False the plant is drift-only: it builds no noise network,
    its diffusion is a zero column [...,D,1] (noise_dim 1, the shape of a noiseless
    oracle's) and its covariance is zero. Its generator then has no second-order term,
    rate weights no noise term and rollouts no kicks, and its endpoint moments carry
    drift_only=True, so plant_likelihood_loss fits it by plant_drift_loss.

    Episode-constant context coordinates (StateSpec context_dim) have zero rows in P, so
    their drift is zero, and zero rows in the learned diffusion and covariance, so model
    paths never move them; the networks still read them as raw position features.

    Positions listed in StateSpec cyclic_position_indices (no force depends on them: the
    swimmer's root x, y and target) are left out of every network's features, so U, M, D
    and the diffusion are exactly independent of them. With network damping_velocity the
    damping factor reads the velocities too (divided by velocity_feature_scale):
    D(y, v) = L(y, v) L(y, v)^T, still positive semidefinite, so v^T D v >= 0, and
    drift_jacobian adds the dD/dv terms.

    With network limit_port, each position the StateSpec lists in limited_position_ranges
    as (index, low, high) gets a one-sided port that is exactly zero inside [low, high]:
    with the penetration r = max(0, y - high) + max(0, low - y), x = r / w and the
    smoothstep sigma(x) = x^2 (3 - 2x) (1 for x >= 1), a potential
    V = k w^2 Pi(x) (Pi(x) = x^4/4 - x^5/10, then 0.15 + x^2/2 - x/2 beyond x = 1), whose
    stiffness d2V/dy2 = k sigma(x) ramps from 0 to k over the width w, and a damping
    c sigma(x) on the coordinate's velocity, with learned k = exp(.) > 0,
    c = exp(.) > 0 and w = LIMIT_WIDTH_FLOOR + softplus(.). V joins U (potential_energy,
    hamiltonian), the damping joins D, so the port is passive; drift_jacobian includes
    both. It mimics a soft joint limit (MuJoCo's, environment/ph_sac/dmc/joint_limits.py),
    whose force is zero inside the range and ramps up over the impedance width beyond it:
    a kink at a known place that the smooth position networks would have to learn from
    data. The port starts at the StateSpec's limit_port_init, (k, c, w) per range, when
    given (the dm_control factories derive it from the compiled limit:
    joint_limits.limit_port_init), else at LIMIT_STIFFNESS_INIT, LIMIT_DAMPING_INIT and
    LIMIT_WIDTH_INIT; either way inside the ranges the plant is bitwise the same. Without
    these keys the plant is the one of every row before 2026-11-09.
    """

    def __init__(
        self,
        state_spec: StateSpec,
        config: NetworkConfig,
        *,
        learn_diffusion: bool = True,
    ) -> None:
        super().__init__(state_spec)
        n_o, n, m, d = (
            self.position_dim,
            self.velocity_dim,
            self.action_dim,
            self.state_dim,
        )
        self.learns_diffusion = bool(learn_diffusion)
        self.damping_rank = int(config["damping_rank"])
        self.damping_velocity = bool(config.get("damping_velocity", False))
        self.velocity_feature_scale = float(config.get("velocity_feature_scale", 1.0))
        self.mass_floor = float(config["mass_floor"])
        self.diffusion_floor = float(config["diffusion_floor"])

        # Feature slots: raw positions, or sine then cosine of the periodic ones; cyclic
        # positions (no force depends on them) get none.
        periodic = set(state_spec["periodic_position_indices"])
        cyclic = {int(i) for i in state_spec.get("cyclic_position_indices", ())}
        if any(not 0 <= i < n_o for i in cyclic) or len(cyclic) >= n_o:
            raise ValueError("cyclic_position_indices must name some, not all, positions.")
        source, kind = [], []
        for i in range(n_o):
            if i in cyclic:
                continue
            if i in periodic:
                source += [i, i]
                kind += [1, 2]
            else:
                source.append(i)
                kind.append(0)
        features = len(source)
        slots = torch.as_tensor(source, dtype=torch.long)
        kinds = torch.as_tensor(kind, dtype=torch.long)
        self.register_buffer("_slot_source", slots)
        self.register_buffer("_slot_raw", kinds == 0)
        self.register_buffer("_slot_sine", kinds == 1)
        self.register_buffer(
            "_slot_directions", nn.functional.one_hot(slots, n_o).T.bool()
        )

        factors = (("_mass", n), ("_noise", d)) if self.learns_diffusion else (("_mass", n),)
        for name, size in factors:
            rows, cols = torch.tril_indices(size, size)
            self.register_buffer(
                f"{name}_scatter",
                nn.functional.one_hot(rows * size + cols, size * size).to(
                    torch.get_default_dtype()
                ),
            )
            self.register_buffer(f"{name}_diagonal", rows == cols)

        # Untrained model: M=I, U~0, D=0.01I (rank permitting), small position noise.
        hidden = tuple(config["plant_hidden_sizes"])
        self.potential_net = _TangentMLP(features, hidden, 1)
        self.mass_net = _TangentMLP(features, hidden, len(self._mass_diagonal))
        damping_inputs = features + (n if self.damping_velocity else 0)
        self.damping_net = _TangentMLP(damping_inputs, hidden, n * self.damping_rank)
        self.noise_net: Optional[_TangentMLP] = None
        if self.learns_diffusion:
            self.noise_net = _TangentMLP(features, hidden, len(self._noise_diagonal))

        self.potential_net.initialize_output(torch.zeros(1))
        unit_mass = math.log(math.expm1(1.0 - self.mass_floor))
        self.mass_net.initialize_output(
            torch.where(self._mass_diagonal, unit_mass, 0.0).to(
                torch.get_default_dtype()
            )
        )
        damping = torch.zeros(n, self.damping_rank)
        damping.diagonal().fill_(0.1)
        self.damping_net.initialize_output(damping.flatten())
        if self.noise_net is not None:
            rows = torch.tril_indices(d, d)[0]
            self.noise_net.initialize_output(
                torch.where(
                    self._noise_diagonal,
                    torch.where(rows < n_o, math.log(0.01), math.log(1.0)),
                    0.0,
                ).to(torch.get_default_dtype())
            )

        # Rows of the diffusion that may carry noise: every state row but the context's.
        context = context_position_indices(state_spec)
        self._has_context = bool(context)
        noise_rows = torch.ones(d, dtype=torch.get_default_dtype())
        noise_rows[list(context)] = 0.0
        # Not persistent: checkpoints keep the state_dict they had before contexts.
        self.register_buffer("_noise_rows", noise_rows, persistent=False)

        force = torch.zeros(n, m)
        projection = np.asarray(state_spec["observation_projection"])
        for actuator, position in enumerate(state_spec["actuated_position_indices"]):
            force[int(np.argmax(np.abs(projection[position]))), actuator] = 1.0
        self.force_matrix = nn.Parameter(force)

        # The limit port: deterministic initial values, created last (no draw of the RNG).
        self.limit_port = bool(config.get("limit_port", False))
        if self.limit_port:
            ranges = tuple(state_spec.get("limited_position_ranges", ()))
            if not ranges:
                raise ValueError("network limit_port needs StateSpec limited_position_ranges.")
            positions = [int(entry[0]) for entry in ranges]
            physical = n_o - len(context)
            if any(not 0 <= i < physical for i in positions) or len(set(positions)) != len(positions):
                raise ValueError("limited_position_ranges must name distinct physical positions.")
            lows = torch.tensor([float(entry[1]) for entry in ranges])
            highs = torch.tensor([float(entry[2]) for entry in ranges])
            if bool((lows >= highs).any()):
                raise ValueError("limited_position_ranges need low < high.")
            dofs = [int(np.argmax(np.abs(projection[i]))) for i in positions]
            count = len(positions)
            self.register_buffer("_limit_positions", torch.as_tensor(positions, dtype=torch.long))
            self.register_buffer("_limit_velocities", torch.as_tensor(dofs, dtype=torch.long))
            self.register_buffer("_limit_low", lows.to(torch.get_default_dtype()))
            self.register_buffer("_limit_high", highs.to(torch.get_default_dtype()))
            self.register_buffer(
                "_limit_position_map",
                nn.functional.one_hot(self._limit_positions, n_o).to(torch.get_default_dtype()),
            )
            self.register_buffer(
                "_limit_velocity_map",
                nn.functional.one_hot(self._limit_velocities, n).to(torch.get_default_dtype()),
            )
            initial = state_spec.get("limit_port_init")
            if initial is None:
                self.limit_stiffness = nn.Parameter(torch.full((count,), math.log(LIMIT_STIFFNESS_INIT)))
                self.limit_damping = nn.Parameter(torch.full((count,), math.log(LIMIT_DAMPING_INIT)))
                self.limit_width = nn.Parameter(
                    torch.full((count,), _inverse_softplus(LIMIT_WIDTH_INIT - LIMIT_WIDTH_FLOOR))
                )
            else:
                stiffness, damping, width = _limit_port_start(initial, count)
                self.limit_stiffness = nn.Parameter(torch.tensor([math.log(k) for k in stiffness]))
                self.limit_damping = nn.Parameter(torch.tensor([math.log(c) for c in damping]))
                self.limit_width = nn.Parameter(
                    torch.tensor([_inverse_softplus(w - LIMIT_WIDTH_FLOOR) for w in width])
                )

    @property
    def noise_dim(self) -> int:
        return 2 * self.state_dim if self.learns_diffusion else 1

    # --- parts -------------------------------------------------------------------------

    def part_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """This plant's parameters by part (common.ph_config.PLANT_PARTS): "potential"
        (U's network), "mass" (the network of M's Cholesky factor), "damping" (the network
        of D's factor), "input" (G, force_matrix), "diffusion" (the noise network, absent
        from a drift-only plant) and "limit_port" (its stiffness, damping and width, with
        network limit_port). Every parameter belongs to exactly one part."""
        parts: Dict[str, List[nn.Parameter]] = {
            "potential": list(self.potential_net.parameters()),
            "mass": list(self.mass_net.parameters()),
            "damping": list(self.damping_net.parameters()),
            "input": [self.force_matrix],
        }
        if self.noise_net is not None:
            parts["diffusion"] = list(self.noise_net.parameters())
        if self.limit_port:
            parts["limit_port"] = [self.limit_stiffness, self.limit_damping, self.limit_width]
        return parts

    def trainable_parameters(self, parts: Optional[Sequence[str]] = None) -> List[nn.Parameter]:
        """The parameters an online fit moves, in parameters() order.

        parts None: every parameter, nothing changed (every run without the implicit key
        trainable_plant_parts). Otherwise the parameters of the named parts
        (part_parameters), and every other parameter is frozen (requires_grad False), so
        the plant's loss reaches only the named parts and an optimizer built on the
        returned list never moves the rest. A part this plant does not have raises.
        """
        if parts is None:
            return list(self.parameters())
        available = self.part_parameters()
        missing = [part for part in parts if part not in available]
        if missing or not parts:
            raise ValueError(
                f"trainable_plant_parts {list(parts)!r}: this plant's parts are "
                f"{sorted(available)}."
            )
        everything = [parameter for group in available.values() for parameter in group]
        if sorted(map(id, everything)) != sorted(map(id, self.parameters())):
            raise RuntimeError("Every plant parameter must belong to exactly one part.")
        chosen = {id(parameter) for part in parts for parameter in available[part]}
        selected = []
        for parameter in self.parameters():
            trainable = id(parameter) in chosen
            parameter.requires_grad_(trainable)
            if trainable:
                selected.append(parameter)
        return selected

    def initialize_from_physics(
        self, input_matrix: Tensor, velocity_diffusion_scale: Tensor
    ) -> None:
        """Choose the true-G initial gauge representative; free G retains the scale symmetry.

        A drift-only plant has no noise to start, so it ignores the diffusion scale.
        """
        with torch.no_grad():
            self.force_matrix.copy_(torch.as_tensor(input_matrix).to(self.force_matrix))
            if self.noise_net is None:
                return
            bias = self.noise_net.layers[-1].bias
            scale = torch.as_tensor(velocity_diffusion_scale).to(bias).reshape(-1)
            scale = scale.expand(self.velocity_dim)
            rows, cols = torch.tril_indices(self.state_dim, self.state_dim)
            for i in range(self.velocity_dim):
                row = self.position_dim + i
                entry = (rows == row) & (cols == row)
                bias[entry] = scale[i].clamp_min(self.diffusion_floor).log()

    # --- position features with exact tangents ---------------------------------------

    def position_features(self, observed_positions: Tensor) -> Tensor:
        """Replace periodic positions by sine and cosine features."""
        return self._features(observed_positions, order=0)[0]

    def _features(
        self, observed_positions: Tensor, *, order: int
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Return features [B,F] with d/dy [B,n_o,F] and d2/dy2 [B,n_o,n_o,F]."""
        y = observed_positions[:, self._slot_source]
        sine, cosine = y.sin(), y.cos()
        raw, is_sine = self._slot_raw, self._slot_sine
        features = torch.where(raw, y, torch.where(is_sine, sine, cosine))
        if order == 0:
            return features, None, None
        slope = torch.where(raw, torch.ones_like(y), torch.where(is_sine, cosine, -sine))
        directions = self._slot_directions.to(y.dtype)
        first = directions * slope.unsqueeze(1)
        if order == 1:
            return features, first, None
        curvature = torch.where(raw, torch.zeros_like(y), -features)
        second = (
            directions.unsqueeze(1) * directions.unsqueeze(0) * curvature[:, None, None]
        )
        return features, first, second

    @staticmethod
    def _lower(entries: Tensor, scatter: Tensor, size: int) -> Tensor:
        """Place tril entries [...,t] into a lower-triangular matrix [...,size,size]."""
        return (entries @ scatter.to(entries.dtype)).unflatten(-1, (size, size))

    def _mass_jets(
        self, features: Tensor, first: Optional[Tensor], second: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Return L_M with d/dy and d2/dy2, through the softplus diagonal."""
        raw, draw, ddraw = self.mass_net(features, first, second)
        diagonal = self._mass_diagonal
        n = self.velocity_dim
        entries = torch.where(
            diagonal, nn.functional.softplus(raw) + self.mass_floor, raw
        )
        factor = self._lower(entries, self._mass_scatter, n)
        if draw is None:
            return factor, None, None
        gain = torch.sigmoid(raw)
        tangent = torch.where(diagonal, gain.unsqueeze(1) * draw, draw)
        dfactor = self._lower(tangent, self._mass_scatter, n)
        if ddraw is None:
            return factor, dfactor, None
        outer = draw.unsqueeze(2) * draw.unsqueeze(1)
        curved = (
            gain[:, None, None] * ddraw + (gain * (1.0 - gain))[:, None, None] * outer
        )
        ddfactor = self._lower(
            torch.where(diagonal, curved, ddraw), self._mass_scatter, n
        )
        return factor, dfactor, ddfactor

    # --- energy, inertia and damping --------------------------------------------------

    def potential_energy(self, observed_positions: Tensor) -> Tensor:
        """U(y), plus the limit port's V(y) with network limit_port."""
        potential = self.potential_net(self.position_features(observed_positions))[0]
        if self.limit_port:
            potential = potential + self.limit_terms(observed_positions)["potential"]
        return potential

    def limit_terms(self, observed_positions: Tensor) -> dict:
        """The limit port at positions [B,n_o]: "potential" V [B,1], "gradient" dV/dy_i,
        "curvature" d2V/dy_i2, "damping" c sigma and "damping_slope" its d/dy_i, each
        [B,L] per limited coordinate; all exactly zero inside the ranges."""
        y = observed_positions[:, self._limit_positions]
        above = y - self._limit_high.to(y.dtype)
        below = self._limit_low.to(y.dtype) - y
        depth = torch.clamp(above, min=0.0) + torch.clamp(below, min=0.0)
        side = torch.where(above > 0.0, torch.ones_like(y), torch.where(below > 0.0, -torch.ones_like(y), torch.zeros_like(y)))
        stiffness = self.limit_stiffness.exp().to(y.dtype)
        damping = self.limit_damping.exp().to(y.dtype)
        width = LIMIT_WIDTH_FLOOR + nn.functional.softplus(self.limit_width).to(y.dtype)
        x = depth / width
        inner = torch.clamp(x, max=1.0)
        ramp = x < 1.0
        sigma = inner.square() * (3.0 - 2.0 * inner)
        slope = torch.where(ramp, 6.0 * inner * (1.0 - inner), torch.zeros_like(x))
        integral = torch.where(ramp, x.pow(3) - 0.5 * x.pow(4), x - 0.5)
        double = torch.where(ramp, 0.25 * x.pow(4) - 0.1 * x.pow(5), 0.15 + 0.5 * x.square() - 0.5 * x)
        return {
            "potential": (stiffness * width.square() * double).sum(-1, keepdim=True),
            "gradient": stiffness * width * integral * side,
            "curvature": stiffness * sigma,
            "damping": damping * sigma,
            "damping_slope": damping * slope * side / width,
        }

    def _limit_forces(self, positions: Tensor, velocities: Tensor) -> Tuple[dict, Tensor, Tensor]:
        """The port's terms, dV/dy scattered to [B,n_o], and its damping force [B,n]."""
        port = self.limit_terms(positions)
        gradient = port["gradient"] @ self._limit_position_map.to(positions.dtype)
        rates = velocities[:, self._limit_velocities]
        dissipation = (port["damping"] * rates) @ self._limit_velocity_map.to(positions.dtype)
        return port, gradient, dissipation

    def mass_factor(self, observed_positions: Tensor) -> Tensor:
        """Lower-triangular factor with softplus diagonal plus mass floor."""
        return self._mass_jets(self.position_features(observed_positions), None, None)[0]

    def mass_matrix(self, observed_positions: Tensor) -> Tensor:
        factor = self.mass_factor(observed_positions)
        return factor @ factor.mT

    def mass_derivatives(self, observed_positions: Tensor) -> Tensor:
        """Exact forward-mode position tangents dM/dy_j with shape [B,n_o,n,n]."""
        features, first, _ = self._features(observed_positions, order=1)
        factor, dfactor, _ = self._mass_jets(features, first, None)
        half = dfactor @ factor.unsqueeze(1).mT
        return half + half.mT

    def _damping_inputs(self, features: Tensor, velocities: Optional[Tensor]) -> Tensor:
        """The damping network's input: the position features, then the scaled velocities
        when network damping_velocity is set."""
        if not self.damping_velocity:
            return features
        if velocities is None:
            raise ValueError("A velocity-dependent damping needs the velocities.")
        return torch.cat((features, velocities / self.velocity_feature_scale), dim=-1)

    def damping_factor(
        self, observed_positions: Tensor, velocities: Optional[Tensor] = None
    ) -> Tensor:
        """L [B,n,rank]; velocities are required with network damping_velocity."""
        features = self._damping_inputs(self.position_features(observed_positions), velocities)
        entries = self.damping_net(features)[0]
        return entries.unflatten(-1, (self.velocity_dim, self.damping_rank))

    def damping_matrix(
        self, observed_positions: Tensor, velocities: Optional[Tensor] = None
    ) -> Tensor:
        factor = self.damping_factor(observed_positions, velocities)
        return factor @ factor.mT

    def input_matrix(self) -> Tensor:
        return self.force_matrix

    def _noise_factor(self, states: Tensor) -> Tensor:
        raw = self.noise_net(self.position_features(states[:, : self.position_dim]))[0]
        entries = torch.where(self._noise_diagonal, raw.exp(), raw)
        return self._lower(entries, self._noise_scatter, self.state_dim)

    def diffusion(self, states: Tensor) -> Tensor:
        """Return physical [L_Sigma, sigma_min I]; its model floor need not be true noise.

        A drift-only plant returns a zero column [...,D,1].
        """
        if not self.learns_diffusion:
            return states.new_zeros(states.shape[:-1] + (self.state_dim, 1))
        factor = self._noise_factor(states)
        floor = self.diffusion_floor * torch.eye(
            self.state_dim, dtype=states.dtype, device=states.device
        )
        diffusion = torch.cat((factor, floor.expand_as(factor)), dim=-1)
        if self._has_context:
            diffusion = diffusion * self._noise_rows.to(states.dtype).unsqueeze(-1)
        return diffusion

    def covariance(self, states: Tensor) -> Tensor:
        """Return L_Sigma L_Sigma^T + sigma_min^2 I without forming the rectangular factor.

        A drift-only plant returns zeros [...,D,D].
        """
        if not self.learns_diffusion:
            return states.new_zeros(states.shape[:-1] + (self.state_dim, self.state_dim))
        factor = self._noise_factor(states)
        eye = torch.eye(self.state_dim, dtype=states.dtype, device=states.device)
        covariance = factor @ factor.mT + self.diffusion_floor**2 * eye
        if self._has_context:
            rows = self._noise_rows.to(states.dtype)
            covariance = covariance * rows.unsqueeze(-1) * rows
        return covariance

    def hamiltonian(self, observed_positions: Tensor, momenta: Tensor) -> Tensor:
        potential = self.potential_energy(observed_positions)
        solved = torch.cholesky_solve(
            momenta.unsqueeze(-1), self.mass_factor(observed_positions)
        )
        kinetic = 0.5 * (momenta.unsqueeze(-2) @ solved).squeeze(-1)
        return potential + kinetic

    # --- dynamics ---------------------------------------------------------------------

    def drift(self, states: Tensor, actions: Tensor) -> Tensor:
        """Transform Hamilton's equations to (P q, q_dot) observation coordinates."""
        positions = states[:, : self.position_dim]
        velocities = states[:, self.position_dim :]
        features, first, _ = self._features(positions, order=1)
        _, dpotential, _ = self.potential_net(features, first)
        factor, dfactor, _ = self._mass_jets(features, first, None)
        damping = self.damping_net(self._damping_inputs(features, velocities))[0].unflatten(
            -1, (self.velocity_dim, self.damping_rank)
        )
        half = dfactor @ factor.unsqueeze(1).mT
        dmass = half + half.mT
        if not self.limit_port:
            terms = self._momentum_terms(
                velocities, dpotential.squeeze(-1), dmass, damping, actions
            )
            acceleration = torch.cholesky_solve(terms[0].unsqueeze(-1), factor).squeeze(-1)
            return torch.cat((terms[1], acceleration), dim=-1)
        _, gradient, dissipation = self._limit_forces(positions, velocities)
        terms = self._momentum_terms(
            velocities, dpotential.squeeze(-1) + gradient, dmass, damping, actions
        )
        momentum = terms[0] - dissipation
        acceleration = torch.cholesky_solve(momentum.unsqueeze(-1), factor).squeeze(-1)
        return torch.cat((terms[1], acceleration), dim=-1)

    def _momentum_terms(
        self,
        velocities: Tensor,
        dpotential: Tensor,
        dmass: Tensor,
        damping: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return p_dot - M_dot v, the observed velocity P v, and dM_j v [B,n_o,n]."""
        projection = self.observation_projection.to(velocities.dtype)
        observed = velocities @ projection.T
        directional = (dmass @ velocities[:, None, :, None]).squeeze(-1)
        coriolis = (directional * velocities.unsqueeze(1)).sum(-1)
        grad_q_h = (dpotential - 0.5 * coriolis) @ projection
        dissipation = (
            damping @ (damping.mT @ velocities.unsqueeze(-1))
        ).squeeze(-1)
        applied = actions @ self.force_matrix.T
        mass_rate = (observed.unsqueeze(1) @ directional).squeeze(1)
        return -grad_q_h - dissipation + applied - mass_rate, observed, directional

    def drift_jacobian(
        self, states: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Analytic b, db/ds and db/da from second-order tangents through the perceptrons."""
        n_o, n = self.position_dim, self.velocity_dim
        positions = states[:, :n_o]
        velocities = states[:, n_o:]
        projection = self.observation_projection.to(states.dtype)

        features, first, second = self._features(positions, order=2)
        _, dpotential, ddpotential = self.potential_net(features, first, second)
        dpotential = dpotential.squeeze(-1)
        ddpotential = ddpotential.squeeze(-1)
        factor, dfactor, ddfactor = self._mass_jets(features, first, second)
        if self.damping_velocity:
            # Tangents of the damping input in the n_o positions, then the n velocities.
            width = features.shape[-1]
            scaled = torch.eye(n, dtype=states.dtype, device=states.device)
            scaled = (scaled / self.velocity_feature_scale).expand(len(states), n, n)
            tangents = torch.cat(
                (
                    torch.cat((first, first.new_zeros(len(states), n_o, n)), dim=-1),
                    torch.cat((scaled.new_zeros(len(states), n, width), scaled), dim=-1),
                ),
                dim=1,
            )
            raw_damping, draw_damping, _ = self.damping_net(
                self._damping_inputs(features, velocities), tangents
            )
            vdamping = draw_damping[:, n_o:].unflatten(-1, (n, self.damping_rank))
            draw_damping = draw_damping[:, :n_o]
        else:
            raw_damping, draw_damping, _ = self.damping_net(features, first)
        damping = raw_damping.unflatten(-1, (n, self.damping_rank))
        ddamping = draw_damping.unflatten(-1, (n, self.damping_rank))

        half = dfactor @ factor.unsqueeze(1).mT
        dmass = half + half.mT
        cross = dfactor.unsqueeze(2) @ dfactor.unsqueeze(1).mT
        curved = ddfactor @ factor[:, None, None].mT
        ddmass = curved + curved.mT + cross + cross.mT

        if self.limit_port:
            port, gradient, dissipation = self._limit_forces(positions, velocities)
            position_map = self._limit_position_map.to(states.dtype)
            dpotential = dpotential + gradient
            ddpotential = ddpotential + torch.diag_embed(port["curvature"] @ position_map)
        momentum, observed, directional = self._momentum_terms(
            velocities, dpotential, dmass, damping, actions
        )
        if self.limit_port:
            momentum = momentum - dissipation
        acceleration = torch.cholesky_solve(momentum.unsqueeze(-1), factor).squeeze(-1)

        # d/dy_l: -P^T d2U + P^T (v^T d2M v)/2 - dD_l v - sum_k (Pv)_k d2M_kl v - dM_l a.
        curved_velocity = (ddmass @ velocities[:, None, None, :, None]).squeeze(-1)
        coriolis = (curved_velocity * velocities[:, None, None]).sum(-1)
        ddamping_half = ddamping @ (damping.unsqueeze(1).mT @ velocities[:, None, :, None])
        ddamping_half = ddamping_half.squeeze(-1)
        damping_half = (
            damping.unsqueeze(1) @ (ddamping.mT @ velocities[:, None, :, None])
        ).squeeze(-1)
        mass_curve = (observed[:, None, None] @ curved_velocity.transpose(1, 2)).squeeze(2)
        mass_force = (dmass @ acceleration[:, None, :, None]).squeeze(-1)
        by_position = (
            (-ddpotential + 0.5 * coriolis.transpose(1, 2)) @ projection
            - ddamping_half
            - damping_half
            - mass_curve
            - mass_force
        )

        # d/dv: P^T A - D - M_dot - A^T P, with A_ki = (dM_k v)_i.
        mass_rate = (observed[:, :, None, None] * dmass).sum(1)
        by_velocity = (
            projection.T @ directional
            - damping @ damping.mT
            - mass_rate
            - directional.mT @ projection
        )
        if self.damping_velocity:
            # d/dv_k of L L^T v beyond L L^T: dL_k L^T v + L dL_k^T v, [B,k,n] -> [B,n,k].
            rows = velocities[:, None, :, None]
            through = (vdamping @ (damping.unsqueeze(1).mT @ rows)).squeeze(-1)
            back = (damping.unsqueeze(1) @ (vdamping.mT @ rows)).squeeze(-1)
            by_velocity = by_velocity - (through + back).mT
        if self.limit_port:
            # The port's damping c sigma(y) on its coordinate's rate: d/dy and d/dv.
            velocity_map = self._limit_velocity_map.to(states.dtype)
            rates = velocities[:, self._limit_velocities]
            by_position = by_position - (
                position_map.T.unsqueeze(0) * (port["damping_slope"] * rates).unsqueeze(1)
            ) @ velocity_map
            by_velocity = by_velocity - torch.diag_embed(port["damping"] @ velocity_map)

        accel_position = torch.cholesky_solve(by_position.mT, factor)
        accel_velocity = torch.cholesky_solve(by_velocity, factor)
        accel_action = torch.cholesky_solve(
            self.force_matrix.expand(len(states), -1, -1), factor
        )

        top = torch.cat(
            (
                states.new_zeros(len(states), n_o, n_o),
                projection.expand(len(states), -1, -1),
            ),
            dim=-1,
        )
        ds = torch.cat(
            (top, torch.cat((accel_position, accel_velocity), dim=-1)), dim=1
        )
        da = torch.cat(
            (states.new_zeros(len(states), n_o, self.action_dim), accel_action), dim=1
        )
        return torch.cat((observed, acceleration), dim=-1), ds, da


class OraclePlant(Plant):
    """Exact drift and diffusion supplied by the MuJoCo testbed specification."""

    def __init__(
        self,
        drift_function: Callable[[Tensor, Tensor], Tensor],
        diffusion_function: Callable[[Tensor], Tensor],
        input_matrix: Tensor,
        *,
        state_spec: StateSpec,
    ) -> None:
        super().__init__(state_spec)
        self._drift_function = drift_function
        self._diffusion_function = diffusion_function
        self.register_buffer(
            "_input_matrix", torch.as_tensor(input_matrix).to(torch.get_default_dtype())
        )

    def __deepcopy__(self, memo: dict) -> OraclePlant:
        # The callbacks may be bound to a simulator; an oracle has no state to copy.
        clone = OraclePlant(
            self._drift_function,
            self._diffusion_function,
            self._input_matrix.clone(),
            state_spec=self.state_spec,
        )
        clone.observation_projection = self.observation_projection.clone()
        return clone

    @property
    def noise_dim(self) -> int:
        probe = self.observation_projection.new_zeros(1, self.state_dim)
        return int(self.diffusion(probe).shape[-1])

    def drift(self, states: Tensor, actions: Tensor) -> Tensor:
        return self._drift_function(states, actions)

    def diffusion(self, states: Tensor) -> Tensor:
        return self._diffusion_function(states)

    def input_matrix(self) -> Tensor:
        return self._input_matrix
