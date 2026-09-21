from gymnasium.envs.registration import register

from .landing_env import RocketLandingEnv

register(id="RocketLanding-v0", entry_point="rocket_sim.envs.landing_env:RocketLandingEnv")

__all__ = ["RocketLandingEnv"]
