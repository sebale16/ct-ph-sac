"""Config factories of the dm_control ports, one module per domain.

The environments, tensor task functions and registry entries live in the parallel
package environment/ph_sac/dmc/ (common must not import environment). _common holds the
shared clock, decision-box rule, state spec and TestbedConfig builders.
"""

from . import acrobot, cartpole, lqr, pendulum, point_mass, reacher, swimmer  # noqa: F401
