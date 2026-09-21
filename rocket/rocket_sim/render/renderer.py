"""Composites the rocket parts for a given visual state and draws it in world space."""

import math
from dataclasses import dataclass

import numpy as np
import pygame

from .exhaust import PlumeFactory, SmokeSystem, heat_shimmer, radial_glow
from .geometry import RocketGeometry
from .raster import blit_rotated, rgba_surface
from .rocket_parts import RocketPartFactory, Sprite


@dataclass
class RocketVisualState:
    """Everything the renderer needs. All of it will come from the physics later.

    x, y         : centre-of-mass position in world metres (y up, ground at ground_y)
    angle        : body tilt in radians, counter-clockwise positive (0 = upright)
    throttle     : 0..1
    gimbal       : engine gimbal in radians, CCW positive relative to the body
    engines      : number of lit engines (0, 1, 3 or 9)
    leg_deploy   : 0 stowed .. 1 fully deployed
    fin_deploy   : 0 stowed .. 1 fully deployed
    fin_twist    : grid-fin control deflection in radians
    """
    x: float = 0.0
    y: float = 14.0
    angle: float = 0.0
    throttle: float = 0.0
    gimbal: float = 0.0
    engines: int = 9
    leg_deploy: float = 0.0
    fin_deploy: float = 0.0
    fin_twist: float = 0.0


class Camera:
    def __init__(self, width: int, height: int, ppm: float, center=(0.0, 0.0)):
        self.width, self.height = width, height
        self.ppm = ppm
        self.center = np.array(center, dtype=float)

    def world_to_screen(self, p):
        return (self.width / 2 + (p[0] - self.center[0]) * self.ppm,
                self.height / 2 - (p[1] - self.center[1]) * self.ppm)


def air_pressure(altitude_m: float) -> float:
    """Normalised ambient pressure, 1 at sea level (exponential atmosphere)."""
    return math.exp(-max(0.0, altitude_m) / 8500.0)


class _ArtSet:
    """All static sprites for one pixels-per-metre level."""

    def __init__(self, geometry: RocketGeometry, ppm: float):
        self.ppm = ppm
        self.parts = RocketPartFactory(geometry, ppm)
        self.body = self.parts.build_body()
        self.leg = self.parts.build_leg()
        self.leg_l = self.leg.mirrored()
        self.sleeve = self.parts.build_strut_sleeve()
        self.rod = self.parts.build_strut_rod()
        self.pad = self.parts.build_foot_pad()


class RocketRenderer:
    CANVAS_HALF_W = 12.5      # metres either side of the axis (covers deployed legs)
    CANVAS_BOTTOM = -3.0

    def __init__(self, geometry: RocketGeometry | None = None, seed: int = 0):
        self.g = geometry or RocketGeometry()
        self._arts = {}
        self._plumes = {}
        self.smoke = SmokeSystem(seed)
        self._glows = {}
        self.time = 0.0
        self._rng = np.random.default_rng(seed + 1)
        self._variant = 0
        self._variant_timer = 0.0

    # ------------------------------------------------------------ caching
    @staticmethod
    def _quantize_ppm(ppm: float) -> float:
        k = round(math.log(max(ppm, 1.0)) / math.log(1.15))
        return float(min(48.0, max(2.0, 1.15 ** k)))

    def art(self, ppm: float) -> _ArtSet:
        q = self._quantize_ppm(ppm)
        if q not in self._arts:
            self._arts[q] = _ArtSet(self.g, q)
        return self._arts[q]

    def plume_factory(self, ppm: float) -> PlumeFactory:
        q = self._quantize_ppm(ppm)
        if q not in self._plumes:
            self._plumes[q] = PlumeFactory(self.g, q)
        return self._plumes[q]

    def _glow(self, radius_px: int, color, falloff=2.2):
        key = (radius_px, color, falloff)
        if key not in self._glows:
            if len(self._glows) > 64:
                self._glows.clear()
            self._glows[key] = radial_glow(radius_px, color, falloff)
        return self._glows[key]

    # ------------------------------------------------------- body frame
    def body_to_world(self, state: RocketVisualState, p):
        a = state.angle
        dx, dy = p[0], p[1] - self.g.com_y
        return (state.x + dx * math.cos(a) - dy * math.sin(a),
                state.y + dx * math.sin(a) + dy * math.cos(a))

    def nozzle_exit_body(self, state: RocketVisualState):
        L = self.g.nozzle_throat_y
        return (L * math.sin(state.gimbal), L - L * math.cos(state.gimbal))

    def compose(self, state: RocketVisualState, ppm: float):
        """Draw the upright rocket (body frame) onto a transparent canvas.

        Returns (canvas, pivot_px, art_ppm); pivot is the centre of mass.
        """
        art = self.art(ppm)
        g, q = self.g, art.ppm
        W = 2 * self.CANVAS_HALF_W * q
        top = g.total_length + 1.0
        H = (top - self.CANVAS_BOTTOM) * q
        canvas = rgba_surface(W, H)

        def P(x, y):
            return (self.CANVAS_HALF_W * q + x * q, (top - y) * q)

        R = g.body_radius
        psi = math.radians(g.leg_deployed_deg) * _ease(state.leg_deploy)
        fin_a = 90.0 * _ease(state.fin_deploy)
        twist = math.degrees(state.fin_twist) * _ease(state.fin_deploy)

        # --- profile legs + telescoping struts (behind the body)
        hinge_x = R + 0.22
        deploy_frac = psi / math.radians(g.leg_deployed_deg)
        for side in (1, -1):
            H0 = (side * hinge_x, g.leg_hinge_y)
            d = (side * math.sin(psi), math.cos(psi))
            Pl = (H0[0] + d[0] * g.leg_length * g.strut_leg_frac, H0[1] + d[1] * g.leg_length * g.strut_leg_frac)
            A = (side * (R + 0.05), g.strut_body_y)
            self._link(canvas, art.rod, P(*Pl), P(*A))
            self._link(canvas, art.sleeve, P(*A), P(*Pl))
            leg = art.leg if side > 0 else art.leg_l
            blit_rotated(canvas, leg.surface, leg.pivot, P(*H0), -side * math.degrees(psi))
            tip = (H0[0] + d[0] * g.leg_length, H0[1] + d[1] * g.leg_length)
            # the pad swivels from flush-with-the-leg to level with the ground
            blit_rotated(canvas, art.pad.surface, art.pad.pivot, P(*tip), side * 90 * (1 - deploy_frac))

        # --- profile grid fins
        for side in (1, -1):
            fin = art.parts.profile_fin(twist)
            if side < 0:
                fin = fin.mirrored()
            blit_rotated(canvas, fin.surface, fin.pivot, P(side * (R - 0.05), g.fin_hinge_y), -side * fin_a)

        # --- engines, then body over their tops
        cluster = art.parts.nozzle_cluster(math.degrees(state.gimbal))
        canvas.blit(cluster.surface, (P(0, 0)[0] - cluster.pivot[0], P(0, 0)[1] - cluster.pivot[1]))
        canvas.blit(art.body.surface, (P(0, 0)[0] - art.body.pivot[0], P(0, 0)[1] - art.body.pivot[1]))

        # --- leg facing the viewer: swings toward the camera, so it foreshortens
        c = math.cos(psi)
        depth = math.sin(psi)
        Pf_y = g.leg_hinge_y + c * g.leg_length * g.strut_leg_frac
        strut_dy = Pf_y - g.strut_body_y
        strut_len = math.hypot(strut_dy, depth * g.leg_length * g.strut_leg_frac)
        k = abs(strut_dy) / max(strut_len, 1e-6)
        self._link(canvas, art.rod, P(0, Pf_y), P(0, g.strut_body_y), k)
        self._link(canvas, art.sleeve, P(0, g.strut_body_y), P(0, Pf_y), k)
        fl = art.leg
        h = max(1, int(fl.surface.get_height() * abs(c)))
        img = pygame.transform.smoothscale(fl.surface, (fl.surface.get_width(), h))
        piv_y = fl.pivot[1] * abs(c)
        if c < 0:
            img = pygame.transform.flip(img, False, True)
            piv_y = h - piv_y
        hx, hy = P(0, g.leg_hinge_y)
        canvas.blit(img, (hx - fl.pivot[0], hy - piv_y))
        if c < 0.35:
            fy = g.leg_hinge_y + c * g.leg_length
            px, py = P(0, fy - 0.1)
            art.pad.surface.set_alpha(int(255 * min(1, (0.35 - c) / 0.3)))
            canvas.blit(art.pad.surface, (px - art.pad.pivot[0], py - art.pad.pivot[1]))

        # --- grid fin facing the viewer
        ff = art.parts.front_fin(_ease(state.fin_deploy))
        blit_rotated(canvas, ff.surface, ff.pivot, P(0, g.fin_hinge_y - 0.05), twist)

        return canvas, P(0, g.com_y), q

    @staticmethod
    def _link(canvas, sprite: Sprite, from_px, to_px, length_scale: float = 1.0):
        """Place a beam sprite (drawn pointing up from its pivot) so it points from
        from_px toward to_px (pixel coords)."""
        dx, dy = to_px[0] - from_px[0], -(to_px[1] - from_px[1])
        ang = math.degrees(math.atan2(-dx, dy))
        img, piv = sprite.surface, sprite.pivot
        if abs(length_scale - 1) > 0.01:
            h = max(1, int(img.get_height() * length_scale))
            img = pygame.transform.smoothscale(img, (img.get_width(), h))
            piv = (piv[0], piv[1] * length_scale)
        blit_rotated(canvas, img, piv, from_px, ang)

    # ------------------------------------------------------------ frame
    def update(self, state: RocketVisualState, dt: float, ground_y: float = 0.0):
        """Advance time-dependent effects (flicker, smoke)."""
        self.time += dt
        self._variant_timer -= dt
        if self._variant_timer <= 0:
            self._variant = int(self._rng.integers(PlumeFactory.VARIANTS))
            self._variant_timer = 0.045
        firing = state.throttle > 0.01 and state.engines > 0
        p = air_pressure(self.body_to_world(state, (0, 0))[1] - ground_y)
        if firing:
            exit_w = self.body_to_world(state, self.nozzle_exit_body(state))
            ang = state.angle + state.gimbal
            direction = (math.sin(ang), -math.cos(ang))
            L = PlumeFactory(self.g, 1).plume_length(p) * (0.35 + 0.65 * state.throttle)
            rate = 28 * state.throttle * math.sqrt(state.engines)
            self.smoke.emit(exit_w, direction, rate, dt, p, ground_y, L)
        self.smoke.update(dt, ground_y)

    def draw(self, surface: pygame.Surface, camera: Camera, state: RocketVisualState,
             ground_y: float = 0.0, effects: bool = True):
        ppm = camera.ppm
        altitude = self.body_to_world(state, (0, 0))[1] - ground_y
        p = air_pressure(altitude)
        firing = state.throttle > 0.01 and state.engines > 0

        # ground contact shadow
        if altitude < 60:
            gx, gy = camera.world_to_screen((state.x, ground_y))
            w = (10 + altitude * 0.4) * ppm
            k = max(0.0, 1 - altitude / 60)
            sh = rgba_surface(w, w * 0.12)
            pygame.draw.ellipse(sh, (0, 0, 0, int(110 * k)), sh.get_rect())
            surface.blit(sh, (gx - w / 2, gy - w * 0.06))

        if effects:
            self.smoke.draw(surface, camera)

        exit_w = self.body_to_world(state, self.nozzle_exit_body(state))
        exit_s = camera.world_to_screen(exit_w)
        if firing:
            thr = state.throttle
            # bloom around the engine section
            r = int((4 + 2.2 * math.sqrt(state.engines)) * ppm * (0.6 + 0.4 * thr))
            if r > 2:
                glow = self._glow(min(r, 600), (int(120 * thr), int(62 * thr), int(22 * thr)), 1.8)
                surface.blit(glow, glow.get_rect(center=exit_s), special_flags=pygame.BLEND_RGB_ADD)
            pf = self.plume_factory(ppm)
            tex, piv, tex_ppm = pf.texture(state.engines, p, self._variant)
            length = (0.35 + 0.65 * thr) * (0.97 + 0.06 * self._rng.random())
            width = 0.8 + 0.2 * thr
            tw, th = tex.get_size()
            img = pygame.transform.smoothscale(tex, (max(1, int(tw * width)), max(1, int(th * length))))
            b = int(255 * (0.55 + 0.45 * thr))
            img.fill((b, b, b), special_flags=pygame.BLEND_RGB_MULT)
            rect = blit_rotated(surface, img, (piv[0] * width, piv[1] * length), exit_s,
                                math.degrees(state.angle + state.gimbal), zoom=ppm / tex_ppm,
                                special_flags=pygame.BLEND_RGB_ADD)
            if effects:
                heat_shimmer(surface, rect, 2.2 * p * thr * min(1.0, ppm / 6), self.time)

        canvas, pivot, art_ppm = self.compose(state, ppm)
        blit_rotated(surface, canvas, pivot, camera.world_to_screen((state.x, state.y)),
                     math.degrees(state.angle), zoom=ppm / art_ppm)

        if firing:
            # hot nozzle exits
            active = self.g.active_engine_set(state.engines)
            L = self.g.nozzle_throat_y
            for x, depth, i in self.g.engine_positions():
                if i not in active or depth < -0.1:
                    continue
                bx = x + L * math.sin(state.gimbal)
                s = camera.world_to_screen(self.body_to_world(state, (bx, 0.1)))
                r = max(2, int(self.g.nozzle_exit_radius * 1.6 * ppm))
                t = state.throttle
                glow = self._glow(r, (int(255 * t), int(170 * t), int(90 * t)), 2.5)
                surface.blit(glow, glow.get_rect(center=s), special_flags=pygame.BLEND_RGB_ADD)


def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)
