"""dm_control ports as CT-PH-SAC testbeds, one module per domain.

Importing the package registers every port in environment/ph_sac/registry.py, all
implemented: pendulum_swingup, acrobot_swingup, acrobot_swingup_sparse, the six cart-pole
tasks, reacher_easy, reacher_hard, point_mass_easy, lqr_2_1, lqr_6_2, and (since
2026-11-09) swimmer_swimmer6 and swimmer_swimmer15. chain.py holds the planar-chain
dynamics, _common.py the shared helpers; README.md is the porting contract. The config
factories are in common/ph_dmc/.
"""

from . import acrobot, cartpole, lqr, pendulum, point_mass, reacher, swimmer  # noqa: F401
