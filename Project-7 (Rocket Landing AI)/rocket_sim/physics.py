"""2D rigid-body flight dynamics for the booster.

Pure numpy/math, no pygame: the manual flight tool uses it now and the
Gymnasium environment will use it later. Units are SI; world frame has +y up
and the ground at y = 0. Angles are counter-clockwise positive, 0 = upright.
"""

import math
from dataclasses import dataclass

from .render.geometry import RocketGeometry

G0 = 9.80665


@dataclass
class PhysicsParams:
    dry_mass: float = 25_000.0            # kg
    engine_thrust_sl: float = 700_000.0   # N per engine at sea level
    engine_thrust_vac: float = 780_000.0  # N per engine in vacuum
    isp_sl: float = 282.0                 # s
    isp_vac: float = 311.0                # s
    min_throttle: float = 0.35            # engines cannot run below this
    throttle_rate: float = 1.5            # 1/s, how fast the throttle follows the command
    gimbal_rate: float = math.radians(20) # rad/s
    rcs_torque: float = 250_000.0         # N*m from cold-gas thrusters at the top
    fin_force_coeff: float = 2.2          # lateral force per (dynamic pressure * rad)
    fin_area: float = 4 * 1.6 * 1.3       # m^2, all four fins
    cd_axial: float = 0.85
    cd_lateral: float = 1.2
    leg_deploy_time: float = 2.5          # s
    fin_deploy_time: float = 1.0          # s
    rho0: float = 1.225                   # kg/m^3 at sea level
    scale_height: float = 8500.0          # m

    # touchdown limits for a successful landing
    max_touchdown_vy: float = 4.0         # m/s
    max_touchdown_vx: float = 2.5         # m/s
    max_touchdown_angle: float = math.radians(8)
    max_touchdown_omega: float = 0.25     # rad/s
    pad_radius: float = 150.0           # 300 m wide pad (forgiving, for manual flying)
    pad_height: float = 1.2


@dataclass
class Controls:
    """Commands. The actual actuators follow these with realistic rate limits."""
    throttle: float = 0.0     # 0 = engines off, otherwise clamped to [min_throttle, 1]
    gimbal: float = 0.0       # rad, CCW positive, clamped to the geometry's max gimbal
    rcs: float = 0.0          # -1..1, +1 = torque that rotates the rocket CCW
    fins: float = 0.0         # -1..1 grid-fin deflection command
    engines: int = 1          # 1, 3 or 9
    legs: bool = False
    fins_deployed: bool = False


@dataclass
class RocketState:
    x: float = 0.0
    y: float = 0.0            # centre of mass height
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0
    omega: float = 0.0
    fuel: float = 0.0         # kg
    throttle: float = 0.0     # actual
    gimbal: float = 0.0       # actual
    engines: int = 1
    leg_deploy: float = 0.0
    fin_deploy: float = 0.0
    fin_twist: float = 0.0
    time: float = 0.0
    held: bool = False        # on the launch clamps: released once thrust exceeds weight


@dataclass
class FlightStatus:
    landed: bool = False
    crashed: bool = False
    reason: str = ""

    @property
    def done(self) -> bool:
        return self.landed or self.crashed


class RocketPhysics:
    def __init__(self, geometry: RocketGeometry | None = None, params: PhysicsParams | None = None):
        self.g = geometry or RocketGeometry()
        self.p = params or PhysicsParams()
        self.state = RocketState()
        self.status = FlightStatus()
        self.fuel_capacity = 1.0

    # ------------------------------------------------------------ set-up
    def reset(self, state: RocketState, fuel_capacity: float | None = None):
        self.state = state
        self.status = FlightStatus()
        self.fuel_capacity = fuel_capacity or max(state.fuel, 1.0)

    # ---------------------------------------------------------- helpers
    @property
    def mass(self) -> float:
        return self.p.dry_mass + self.state.fuel

    @property
    def inertia(self) -> float:
        L = self.g.total_length
        return self.mass * L * L / 12.0

    def altitude(self) -> float:
        """Height of the lowest point of the vehicle above the local surface."""
        return min(y - self.surface_height(x) for x, y in self.contact_points())

    def surface_height(self, x: float) -> float:
        return self.p.pad_height if abs(x) <= self.p.pad_radius else 0.0

    def air_density(self, y: float) -> float:
        return self.p.rho0 * math.exp(-max(0.0, y) / self.p.scale_height)

    def pressure_ratio(self) -> float:
        return math.exp(-max(0.0, self.state.y) / self.p.scale_height)

    def body_to_world(self, bx: float, by: float):
        s = self.state
        dy = by - self.g.com_y
        c, sn = math.cos(s.angle), math.sin(s.angle)
        return s.x + bx * c - dy * sn, s.y + bx * sn + dy * c

    def contact_points(self):
        """World positions of the parts that can touch the ground first."""
        g = self.g
        pts = [self.body_to_world(sx * (g.engine_ring_radius + g.nozzle_exit_radius), 0.0) for sx in (-1, 1)]
        if self.state.leg_deploy > 0.02:
            psi = math.radians(g.leg_deployed_deg) * _ease(self.state.leg_deploy)
            hx = g.body_radius + 0.22
            for side in (-1, 1):
                fx = side * (hx + math.sin(psi) * g.leg_length)
                fy = g.leg_hinge_y + math.cos(psi) * g.leg_length - 0.28
                pts.append(self.body_to_world(fx, fy))
        return pts

    def engine_thrust(self) -> float:
        s, p = self.state, self.p
        if s.throttle <= 0 or s.fuel <= 0:
            return 0.0
        pr = self.pressure_ratio()
        per_engine = p.engine_thrust_sl * pr + p.engine_thrust_vac * (1 - pr)
        return s.engines * per_engine * s.throttle

    def mass_flow(self) -> float:
        p = self.p
        pr = self.pressure_ratio()
        isp = p.isp_sl * pr + p.isp_vac * (1 - pr)
        return self.engine_thrust() / (isp * G0)

    def twr(self) -> float:
        return self.engine_thrust() / (self.mass * G0)

    def max_decel(self, engines: int | None = None) -> float:
        """Net upward deceleration available at full throttle (m/s^2)."""
        p, s = self.p, self.state
        pr = self.pressure_ratio()
        n = engines or s.engines
        thrust = n * (p.engine_thrust_sl * pr + p.engine_thrust_vac * (1 - pr))
        return thrust / self.mass - G0

    # ------------------------------------------------------------- step
    def step(self, controls: Controls, dt: float, substeps: int = 4) -> FlightStatus:
        if self.status.done:
            return self.status
        h = dt / substeps
        for _ in range(substeps):
            self._substep(controls, h)
            if self.status.done:
                break
        return self.status

    def _substep(self, c: Controls, dt: float):
        s, p, g = self.state, self.p, self.g
        s.time += dt

        # --- actuators
        s.engines = c.engines if c.engines in (1, 3, 9) else 1
        if c.throttle <= 0 or s.fuel <= 0:
            target = 0.0
        else:
            target = min(1.0, max(p.min_throttle, c.throttle))
        if target == 0.0:
            s.throttle = max(0.0, s.throttle - p.throttle_rate * 3 * dt)   # shutdown is fast
            if s.throttle < p.min_throttle * 0.5:
                s.throttle = 0.0
        else:
            if s.throttle == 0.0:
                s.throttle = p.min_throttle * 0.5                        # ignition
            s.throttle += max(-p.throttle_rate * dt, min(p.throttle_rate * dt, target - s.throttle))
        lim = math.radians(g.max_gimbal_deg)
        gcmd = max(-lim, min(lim, c.gimbal))
        s.gimbal += max(-p.gimbal_rate * dt, min(p.gimbal_rate * dt, gcmd - s.gimbal))
        s.leg_deploy = min(1.0, s.leg_deploy + dt / p.leg_deploy_time) if c.legs else \
            max(0.0, s.leg_deploy - dt / p.leg_deploy_time)
        s.fin_deploy = min(1.0, s.fin_deploy + dt / p.fin_deploy_time) if c.fins_deployed else \
            max(0.0, s.fin_deploy - dt / p.fin_deploy_time)
        twist_target = math.radians(g.fin_max_twist_deg) * max(-1.0, min(1.0, c.fins)) * s.fin_deploy
        s.fin_twist += max(-1.5 * dt, min(1.5 * dt, twist_target - s.fin_twist))

        m = self.mass
        fx, fy = 0.0, -m * G0
        torque = 0.0
        ca, sa = math.cos(s.angle), math.sin(s.angle)

        # --- thrust through the gimballed engines
        thrust = self.engine_thrust()
        if thrust > 0:
            d = s.angle + s.gimbal
            tx, ty = -math.sin(d) * thrust, math.cos(d) * thrust
            fx += tx
            fy += ty
            rx, ry = self._lever(g.nozzle_throat_y)
            torque += rx * ty - ry * tx
            s.fuel = max(0.0, s.fuel - self.mass_flow() * dt)

        # --- aerodynamics: split airflow into body axial / lateral components
        rho = self.air_density(s.y)
        v_ax = s.vx * -sa + s.vy * ca          # along the nose direction
        v_lat = s.vx * ca + s.vy * sa          # toward the body's right side
        R = g.body_radius
        area_ax = math.pi * R * R * (1 + 0.6 * s.leg_deploy)
        area_lat = g.total_length * 2 * R
        f_ax = -0.5 * rho * p.cd_axial * area_ax * v_ax * abs(v_ax)
        f_lat = -0.5 * rho * p.cd_lateral * area_lat * v_lat * abs(v_lat)
        # the grid fins shift the centre of pressure toward the top, which makes an
        # engines-first descent weathervane-stable
        y_cp = 17.0 + 8.0 * s.fin_deploy
        fxw, fyw = f_ax * -sa + f_lat * ca, f_ax * ca + f_lat * sa
        fx += fxw
        fy += fyw
        rx, ry = self._lever(y_cp)
        torque += rx * fyw - ry * fxw
        # fin control force (only works with airflow)
        q = 0.5 * rho * (s.vx * s.vx + s.vy * s.vy)
        if s.fin_deploy > 0:
            f_fin = q * p.fin_area * p.fin_force_coeff * s.fin_twist * s.fin_deploy
            # lateral in body frame, applied at the fins
            ffx, ffy = f_fin * ca, f_fin * sa
            fx += ffx
            fy += ffy
            rx, ry = self._lever(g.fin_hinge_y)
            torque += rx * ffy - ry * ffx
        # aerodynamic rotational damping
        torque += -0.5 * rho * 0.6 * area_lat * g.total_length ** 2 / 8 * s.omega * abs(s.omega) * 2
        # cold-gas thrusters
        torque += p.rcs_torque * max(-1.0, min(1.0, c.rcs))

        if s.held:
            if thrust > 1.15 * m * G0:
                s.held = False
            else:
                return
        # --- integrate (semi-implicit Euler)
        s.vx += fx / m * dt
        s.vy += fy / m * dt
        s.omega += torque / self.inertia * dt
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.angle += s.omega * dt
        s.angle = (s.angle + math.pi) % (2 * math.pi) - math.pi

        self._check_ground()

    def _lever(self, body_y: float):
        """World-frame vector from the centre of mass to a point on the axis."""
        d = body_y - self.g.com_y
        return -math.sin(self.state.angle) * d, math.cos(self.state.angle) * d

    # ----------------------------------------------------------- ground
    def _check_ground(self):
        s, p = self.state, self.p
        pts = self.contact_points()
        pen = [(self.surface_height(x) - y, x) for x, y in pts]
        worst = max(pen, key=lambda t: t[0])
        if worst[0] < 0:
            return
        s.y += worst[0]                           # resolve penetration
        legs_ok = s.leg_deploy > 0.99
        feet = pts[2:] if legs_ok else []
        on_pad = legs_ok and all(abs(x) <= p.pad_radius for x, _ in feet)
        problems = []
        if not legs_ok:
            problems.append("legs not deployed")
        if abs(s.vy) > p.max_touchdown_vy:
            problems.append(f"vertical speed {abs(s.vy):.1f} m/s")
        if abs(s.vx) > p.max_touchdown_vx:
            problems.append(f"horizontal speed {abs(s.vx):.1f} m/s")
        if abs(s.angle) > p.max_touchdown_angle:
            problems.append(f"tilt {math.degrees(abs(s.angle)):.0f} deg")
        if abs(s.omega) > p.max_touchdown_omega:
            problems.append("rotating too fast")
        if problems:
            self.status = FlightStatus(crashed=True, reason=", ".join(problems))
        else:
            self.status = FlightStatus(landed=True, reason="on the pad" if on_pad else "off the pad")
        s.vx = s.vy = s.omega = 0.0
        s.throttle = 0.0


def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)
