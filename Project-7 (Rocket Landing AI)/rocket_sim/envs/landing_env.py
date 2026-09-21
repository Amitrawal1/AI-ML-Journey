"""Gymnasium environment: land the booster on the pad.

The agent controls the same physics as tools/fly.py. Each episode starts with
the rocket falling toward the pad; `difficulty` (0..1) sets how high, fast and
far off-target it starts, so training can begin easy and ramp up (curriculum).

Action  (Box, 3 values in [-1, 1])
    0  throttle   -1..1 -> 0..100 %  (below 20 % the engine is off; it cannot
                  run between 0 and 35 %, the physics clamps that)
    1  steer      +1 rotates the nose left: gimbal + cold-gas thrusters + grid fins
                  with autopilot=True it is instead the tilt to hold, -1..1 ->
                  -max_tilt..+max_tilt, and a PD attitude controller does the steering
    2  legs       > 0 deploy, <= 0 stow

Observation  (Box, 12 values, roughly in [-3, 3])
    x/500, altitude/1000, vx/50, vy/100, angle/0.3, omega/0.3, fuel fraction,
    throttle, gimbal/max, leg deploy, braking capability, stop-distance ratio
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..physics import Controls, PhysicsParams, RocketPhysics, RocketState
from ..render.geometry import RocketGeometry


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, difficulty: float = 1.0,
                 pad_width: float = 300.0, control_hz: float = 20.0, max_time: float = 60.0,
                 autopilot: bool = False, max_tilt_deg: float = 12.0):
        self.render_mode = render_mode
        self.autopilot = autopilot
        self.max_tilt = math.radians(max_tilt_deg)
        self.difficulty = float(difficulty)
        self.geom = RocketGeometry()
        self.params = PhysicsParams(pad_radius=pad_width / 2)
        self.phys = RocketPhysics(self.geom, self.params)
        self.dt = 1.0 / control_hz
        self.max_steps = int(max_time * control_hz)
        self.gimbal_lim = math.radians(self.geom.max_gimbal_deg)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(12,), dtype=np.float32)

        self.controls = Controls(engines=1, fins_deployed=True)
        self._view = None
        self._screen = None
        self._clock = None
        self._hud = None

    # ------------------------------------------------------------ curriculum
    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def set_pad_width(self, pad_width: float):
        """Shrink the pad during training (a second curriculum, for precision)."""
        self.params.pad_radius = float(pad_width) / 2

    def _sample_start(self) -> RocketState:
        d, rng = self.difficulty, self.np_random
        alt = rng.uniform(100.0 + 400.0 * d, 100.0 + 2500.0 * d)
        # never start faster than the engine can stop from this height (with margin)
        v_cap = math.sqrt(2 * 8.0 * alt)
        vy = -rng.uniform(0.3, 0.9) * min(v_cap, 20.0 + 160.0 * d)
        return RocketState(
            # like a real booster after its boostback burn: the lower it starts, the closer
            # it already is to the pad (a 450 m miss from 100 m up cannot be fixed in time)
            x=rng.uniform(-1, 1) * min(30.0 + 420.0 * d, 30.0 + 0.3 * alt),
            y=alt + self.params.pad_height + self.geom.com_y,
            vx=rng.uniform(-1, 1) * (3.0 + 17.0 * d),
            vy=vy,
            angle=rng.uniform(-1, 1) * (0.02 + 0.05 * d),
            omega=rng.uniform(-1, 1) * 0.02,
            fuel=rng.uniform(5500.0, 7000.0),
            engines=1,
            fin_deploy=1.0,
        )

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "difficulty" in options:
            self.set_difficulty(options["difficulty"])
        state = self._sample_start()
        self.phys.reset(state, fuel_capacity=state.fuel)
        self.controls = Controls(engines=1, fins_deployed=True)
        self.steps = 0
        self.prev_shaping = self._shaping()
        if self._view is not None:
            self._view.reset()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thr = (a[0] + 1.0) / 2.0
        c = self.controls
        c.throttle = 0.0 if thr < 0.2 else float(thr)
        if self.autopilot:
            steer = self._hold_tilt(float(a[1]) * self.max_tilt)
        else:
            steer = float(a[1])
        c.gimbal = -steer * self.gimbal_lim
        c.rcs = steer
        c.fins = -steer
        c.legs = bool(a[2] > 0.0)

        s = self.phys.state
        v_before = (s.vx, s.vy)
        was_done = self.phys.status.done
        status = self.phys.step(c, self.dt)
        self.steps += 1

        # a crash zeroes the velocity; score it with the speed it hit the ground at
        shaping = self._shaping(v_before if status.crashed else None)
        reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= 0.05 * self.phys.state.throttle          # fuel use
        reward -= 0.05                                      # time cost: hovering is not free

        terminated, truncated = False, False
        info = {}
        if status.done and not was_done:
            terminated = True
            if status.landed:
                x, tilt = self.phys.state.x, abs(self.phys.state.angle)
                on_pad = status.reason == "on the pad"          # every foot on the pad
                # a soft landing beside the pad still beats crashing, but only just:
                # the big payout is for hitting the pad, centred and standing straight
                if on_pad:
                    centred = max(0.0, 1.0 - abs(x) / self.params.pad_radius)
                    upright = max(0.0, 1.0 - tilt / self.params.max_touchdown_angle)
                    reward += 120.0 + 40.0 * centred + 40.0 * upright
                else:
                    reward += 20.0
                info["outcome"] = "landed" if on_pad else "landed_off_pad"
            else:
                impact = math.hypot(*v_before)
                reward -= 100.0 + min(100.0, impact)
                info["outcome"] = "crashed"
            info["reason"] = status.reason
        elif abs(s.x) > 3000.0 or self.phys.altitude() > 4000.0:
            terminated = True
            # must cost more than any crash or timeout, or flying away becomes the escape hatch
            reward -= 400.0
            info["outcome"] = "out_of_bounds"
        elif self.steps >= self.max_steps:
            truncated = True
            # worse than a gentle crash, so trying to touch down always beats hovering
            reward -= 150.0
            info["outcome"] = "timeout"

        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------- internals
    KP, KD = 15.0, 8.0                    # tuned by step tests: ~1 deg overshoot, engine on or off

    def _hold_tilt(self, target: float) -> float:
        """PD attitude hold, like a real booster's inner control loop: guidance
        (the agent) picks the lean, this turns it into a steering command."""
        s = self.phys.state
        return float(np.clip(self.KP * (target - s.angle) - self.KD * s.omega, -1.0, 1.0))

    def _shaping(self, velocity=None) -> float:
        """Potential function: closer, slower, straighter and legs-ready is better."""
        s = self.phys.state
        vx, vy = velocity if velocity is not None else (s.vx, s.vy)
        alt = max(0.0, self.phys.altitude())
        dist = math.hypot(s.x / 250.0, alt / 1000.0)            # sideways error weighs double
        # only sinking faster than a stoppable profile (half the available braking) is penalised,
        # so falling fast from high up is free and the brake comes late, like a real booster
        v_safe = math.sqrt(2.0 * 0.5 * max(self.phys.max_decel(), 0.5) * alt) + 2.0
        sink_excess = max(0.0, -vy - v_safe)
        climb = max(0.0, vy)                               # going up never helps a landing
        speed = math.hypot(vx / 50.0, sink_excess / 20.0, climb / 20.0)
        legs = s.leg_deploy if alt < 300.0 else 0.0
        return -100.0 * dist - 100.0 * speed - 100.0 * abs(s.angle) - 20.0 * abs(s.omega) + 20.0 * legs

    def _obs(self) -> np.ndarray:
        s, ph = self.phys.state, self.phys
        alt = max(0.0, ph.altitude())
        decel = ph.max_decel()
        stop = (s.vy * s.vy / (2 * max(decel, 0.5))) if s.vy < 0 else 0.0
        obs = np.array([
            s.x / 500.0,
            alt / 1000.0,
            s.vx / 50.0,
            s.vy / 100.0,
            s.angle / 0.3,
            s.omega / 0.3,
            s.fuel / max(ph.fuel_capacity, 1.0),
            s.throttle,
            s.gimbal / self.gimbal_lim,
            s.leg_deploy,
            decel / 15.0,
            min(3.0, stop / max(alt, 1.0)),
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        from ..render.flight_view import FlightView, Hud

        size = (1280, 800)
        if self._view is None:
            if self.render_mode == "human":
                pygame.init()
                self._screen = pygame.display.set_mode(size)
                pygame.display.set_caption("Reusable booster - AI pilot")
                self._clock = pygame.time.Clock()
            else:
                pygame.init()
                self._screen = pygame.Surface(size)
            self._view = FlightView(self.phys, size, self.params.pad_radius)
            self._hud = Hud()
            self._last_done = False
        done = self.phys.status.done
        self._view.update(self.dt, self.dt, done and not self._last_done)
        self._last_done = done
        self._view.draw_world(self._screen)
        self._hud.draw(self._screen, self._view, self.controls, "AI pilot", "trained agent  -  ESC to quit")
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self._screen), (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            self._screen = None
            self._view = None
