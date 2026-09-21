"""Physical dimensions of the booster, shared by rendering and (later) physics.

Body frame convention (metres):
    x  -> to the right when the rocket is upright
    y  -> up along the rocket's long axis
    origin at the centre of the engine nozzle exit plane
The centre of mass sits on the axis at (0, com_y); it is the rotation pivot.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class RocketGeometry:
    # Main structure
    body_radius: float = 1.85          # 3.7 m diameter
    skirt_bottom_y: float = 2.0        # heat shield plate / bottom of engine section
    skirt_top_y: float = 3.4           # engine section -> propellant tanks
    tank_top_y: float = 33.0           # tanks -> carbon-composite interstage
    interstage_top_y: float = 38.5     # interstage -> nose cone
    nose_length: float = 5.5
    com_y: float = 14.0                # centre of mass (pivot) above nozzle exit

    # Engines: one centre engine + a ring of eight
    engine_count: int = 9
    engine_ring_radius: float = 1.25
    nozzle_length: float = 2.2
    nozzle_exit_radius: float = 0.46
    nozzle_throat_radius: float = 0.21
    max_gimbal_deg: float = 8.0

    # Landing legs (4, spaced 90 deg; two in profile, one facing the viewer, one behind)
    leg_hinge_y: float = 3.0
    leg_length: float = 9.5
    leg_deployed_deg: float = 118.0    # angle of the leg from "straight up"
    strut_body_y: float = 6.0          # where the telescoping pusher meets the body
    strut_leg_frac: float = 0.70       # where it meets the leg (fraction from hinge)
    strut_sleeve_len: float = 4.0
    strut_rod_len: float = 4.8

    # Grid fins (4, spaced 90 deg)
    fin_hinge_y: float = 37.2
    fin_span: float = 1.6              # radial length when deployed
    fin_chord: float = 1.3             # tangential width
    fin_thickness: float = 0.40        # depth along the airflow
    fin_max_twist_deg: float = 25.0

    engine_ring_azimuths_deg: tuple = field(
        default=(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
    )

    @property
    def total_length(self) -> float:
        return self.interstage_top_y + self.nose_length

    @property
    def nozzle_throat_y(self) -> float:
        return self.nozzle_length

    @property
    def foot_drop(self) -> float:
        """How far the deployed foot pads sit below the nozzle exit plane."""
        return -(self.leg_hinge_y + self.leg_length * math.cos(math.radians(self.leg_deployed_deg)))

    def engine_positions(self):
        """(x_projected, depth, index) for each engine, as seen from the side.

        depth > 0 means the engine is closer to the viewer.
        """
        out = [(0.0, 0.0, 0)]
        for i, az in enumerate(self.engine_ring_azimuths_deg, start=1):
            a = math.radians(az)
            out.append((self.engine_ring_radius * math.cos(a), self.engine_ring_radius * math.sin(a), i))
        return out

    def active_engine_set(self, count: int):
        """Which engine indices fire for a given engine count (1, 3 or 9)."""
        if count <= 0:
            return set()
        if count == 1:
            return {0}
        if count <= 3:
            return {0, 1, 5}          # centre + the two profile ring engines
        return set(range(self.engine_count))

    def summary(self) -> dict:
        return {
            "units": "metres",
            "frame": "origin at nozzle exit centre, +y along the rocket axis toward the nose",
            "total_length": self.total_length,
            "body_diameter": 2 * self.body_radius,
            "center_of_mass": [0.0, self.com_y],
            "engine_count": self.engine_count,
            "nozzle_throat_y": self.nozzle_throat_y,
            "leg_hinge_y": self.leg_hinge_y,
            "leg_length": self.leg_length,
            "leg_deployed_angle_deg": self.leg_deployed_deg,
            "foot_pad_y_deployed": -self.foot_drop,
            "grid_fin_hinge_y": self.fin_hinge_y,
            "grid_fin_span": self.fin_span,
            "max_gimbal_deg": self.max_gimbal_deg,
        }
