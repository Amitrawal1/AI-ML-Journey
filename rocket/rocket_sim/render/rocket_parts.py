"""Procedural sprite builders for each rocket component.

Every builder draws at SUPERSAMPLE x resolution and smooth-scales down, which
gives clean anti-aliased edges. Each sprite comes with the pixel position of its
pivot so the compositor can place and rotate it in the body frame.
"""

import math

import numpy as np
import pygame

from .geometry import RocketGeometry
from .raster import (SUPERSAMPLE, array_to_surface, blit_rotated, cylinder_lighting,
                     downsample, fbm, lit_color, rgba_surface, shaded_beam, value_noise)

# Material palette (albedo, before lighting)
STEEL = np.array([128.0, 132.0, 137.0])
SKIRT = np.array([52.0, 53.0, 56.0])
CARBON = np.array([30.0, 31.0, 34.0])
NOSE = np.array([150.0, 152.0, 156.0])
SOOT = np.array([34.0, 30.0, 27.0])
LEG_CARBON = (40, 41, 45)
FITTING = (104, 106, 110)
CHROME = (188, 192, 197)
TITANIUM = (112, 110, 114)


class Sprite:
    """A rendered component and the pixel position of its pivot."""

    def __init__(self, surface: pygame.Surface, pivot):
        self.surface = surface
        self.pivot = (float(pivot[0]), float(pivot[1]))

    def mirrored(self) -> "Sprite":
        w = self.surface.get_width()
        return Sprite(pygame.transform.flip(self.surface, True, False), (w - self.pivot[0], self.pivot[1]))


class RocketPartFactory:
    def __init__(self, geometry: RocketGeometry, ppm: float):
        self.g = geometry
        self.ppm = ppm
        self.s = ppm * SUPERSAMPLE      # supersampled pixels per metre
        self._nozzle_cache = {}
        self._fin_cache = {}

    # ------------------------------------------------------------------ body
    def _half_width(self, y: np.ndarray) -> np.ndarray:
        g = self.g
        R = g.body_radius
        hw = np.zeros_like(y)
        skirt = (y >= g.skirt_bottom_y) & (y < g.skirt_top_y)
        chamfer = np.clip((y - g.skirt_bottom_y) / 0.18, 0, 1)
        hw[skirt] = (R - 0.12 + 0.19 * chamfer[skirt])
        tank = (y >= g.skirt_top_y) & (y < g.interstage_top_y)
        hw[tank] = R
        nose = (y >= g.interstage_top_y) & (y <= g.total_length)
        L = g.nose_length
        xt = np.clip((g.total_length - y[nose]) / L, 0, 1)   # 0 at tip .. 1 at base
        hw[nose] = R * (0.06 + 0.94 * xt ** 0.58)            # power-law nose, small blunt tip
        hw[nose] = np.minimum(hw[nose], R * np.sqrt(np.clip(xt * L / 0.12, 0, 1)))
        return np.maximum(hw, 0)

    def build_body(self) -> Sprite:
        g, s = self.g, self.s
        R = g.body_radius
        x_min, x_max = -R - 0.3, R + 0.3
        y_min, y_max = g.skirt_bottom_y - 0.05, g.total_length + 0.05
        W, H = int((x_max - x_min) * s), int((y_max - y_min) * s)
        xs = x_min + (np.arange(W) + 0.5) / s
        ys = y_max - (np.arange(H) + 0.5) / s
        X, Y = np.meshgrid(xs, ys, indexing="ij")      # (W, H)

        hw_row = self._half_width(ys)
        hw = np.broadcast_to(hw_row, (W, H))
        alpha = np.clip((hw - np.abs(X)) * s + 0.5, 0, 1)
        n = np.where(hw > 1e-4, X / np.maximum(hw, 1e-4), 0)

        dhw = np.gradient(hw_row, ys)                    # slope for the nose normal
        ny = np.broadcast_to(-dhw, (W, H)) * np.sqrt(np.clip(1 - n * n, 0, 1))

        # ---- materials by station
        albedo = np.zeros((W, H, 3))
        spec = np.zeros((W, H))
        shin = np.full((W, H), 24.0)
        y_row = ys[None, :]
        regions = [
            ((y_row < g.skirt_top_y), SKIRT, 0.18, 12.0),
            ((y_row >= g.skirt_top_y) & (y_row < g.tank_top_y), STEEL, 0.30, 42.0),
            ((y_row >= g.tank_top_y) & (y_row < g.interstage_top_y), CARBON, 0.22, 16.0),
            ((y_row >= g.interstage_top_y), NOSE, 0.24, 30.0),
        ]
        for mask, col, sp, sh in regions:
            m = np.broadcast_to(mask, (W, H))
            albedo[m] = col
            spec[m] = sp
            shin[m] = sh

        # dark ablative tip on the nose
        tip = np.broadcast_to(y_row > g.total_length - 0.7, (W, H))
        albedo[tip] = CARBON * 1.2

        # ---- surface texture
        wear = fbm(W, H, 6, 50, seed=3)
        brushed = value_noise(W, H, 90, 4, seed=11)
        weave = value_noise(W, H, 140, 900, seed=19)
        steel_m = np.broadcast_to((y_row >= g.skirt_top_y) & (y_row < g.tank_top_y), (W, H))
        carbon_m = np.broadcast_to((y_row >= g.tank_top_y) & (y_row < g.interstage_top_y), (W, H))
        albedo *= (0.92 + 0.16 * wear)[..., None]
        albedo[steel_m] *= (0.95 + 0.08 * brushed[steel_m])[..., None]
        albedo[carbon_m] *= (0.9 + 0.2 * weave[carbon_m])[..., None]

        # ---- soot from engine plume recirculation and re-entry
        streak = value_noise(W, H, 34, 2.5, seed=5)
        patch = fbm(W, H, 5, 14, seed=8)
        base_soot = np.clip(1 - (Y - g.skirt_top_y) / 19.0, 0, 1) ** 1.3
        fin_soot = np.clip(1 - (g.fin_hinge_y - 0.6 - Y) / 10.0, 0, 1) * (Y < g.fin_hinge_y - 0.6) * 0.45
        soot = np.clip((base_soot + fin_soot) * (0.25 + 0.9 * streak ** 1.5) * (0.7 + 0.7 * patch), 0, 0.95)
        soot[np.broadcast_to(y_row < g.skirt_top_y, (W, H))] = 0.55 + 0.3 * patch[np.broadcast_to(y_row < g.skirt_top_y, (W, H))]
        soot[~steel_m & ~np.broadcast_to(y_row < g.skirt_top_y, (W, H))] *= 0.25
        albedo = albedo * (1 - soot[..., None]) + SOOT * soot[..., None]
        spec = spec * (1 - 0.75 * soot)

        # ---- panel seams (weld rings on the tanks, joints between sections)
        seam_ys = list(np.arange(g.skirt_top_y + 3.3, g.tank_top_y - 1.0, 3.3))
        seam_ys += [g.skirt_top_y, g.tank_top_y, g.interstage_top_y, g.tank_top_y + 2.6]
        shade = np.ones((W, H))
        for sy in seam_ys:
            shade[:, np.abs(ys - sy) < 0.035] *= 0.5
            shade[:, (ys < sy - 0.035) & (ys > sy - 0.09)] *= 1.12
        # vertical panel lines on skirt and interstage (projected azimuths)
        for az in range(-75, 90, 30):
            px = R * math.sin(math.radians(az))
            col = np.abs(xs - px) < 0.022
            m = col[:, None] & ((y_row < g.skirt_top_y) | ((y_row > g.tank_top_y) & (y_row < g.interstage_top_y)))
            shade[m] *= 0.62
        # nose cone seam lines
        for az in (-50, 0, 50):
            px_row = hw_row * math.sin(math.radians(az))
            m = (np.abs(X - px_row[None, :]) < 0.02) & (Y > g.interstage_top_y) & (Y < g.total_length - 0.7)
            shade[m] *= 0.7
        albedo *= shade[..., None]

        # ---- lighting
        diff, sp_unit = cylinder_lighting(n, ny, spec=1.0, shininess=24.0)
        # per-material shininess: re-power the unit specular term
        sp = spec * np.power(np.maximum(sp_unit, 1e-6), shin / 24.0)
        rgb = albedo * diff[..., None] + 255.0 * sp[..., None]

        surf = array_to_surface(rgb, alpha)
        self._body_details(surf, x_min, y_max)
        sprite = downsample(surf)
        return Sprite(sprite, (-x_min * self.ppm, y_max * self.ppm))

    def _body_details(self, surf, x_min, y_max):
        """Rivets, raceway, access panels - drawn with vector primitives at SS."""
        g, s = self.g, self.s
        R = g.body_radius

        def P(x, y):
            return ((x - x_min) * s, (y_max - y) * s)

        # external cable raceway running up the tanks
        rx = 0.62 * R
        top, bot = P(rx, g.tank_top_y - 0.2), P(rx, g.skirt_top_y + 0.15)
        shadow = pygame.Rect(0, 0, 0.09 * s, bot[1] - top[1])
        shadow.topleft = (top[0] + 0.12 * s, top[1])
        pygame.draw.rect(surf, (40, 40, 42, 90), shadow)
        shaded_beam(surf, bot, top, 0.22 * s, 0.22 * s, (122, 125, 129), strips=7, spec=0.35)
        y = g.skirt_top_y + 0.9
        while y < g.tank_top_y - 0.5:
            a, b = P(rx - 0.16, y), P(rx + 0.16, y + 0.12)
            pygame.draw.rect(surf, (82, 84, 88), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]))
            y += 1.6

        # rivet rows at structural joints
        for ry in (g.skirt_top_y - 0.1, g.skirt_top_y + 0.1, g.tank_top_y + 0.12,
                   g.interstage_top_y - 0.12, g.skirt_bottom_y + 0.3):
            for az in np.arange(-84, 85, 6):
                x = R * math.sin(math.radians(az))
                cx, cy = P(x, ry)
                n = x / R
                col = lit_color((150, 150, 154), n, spec=0.6, shininess=30)
                pygame.draw.circle(surf, (20, 20, 22), (cx + 1, cy + 1), max(1.5, 0.028 * s))
                pygame.draw.circle(surf, col, (cx, cy), max(1.5, 0.026 * s))

        # access panels / service doors
        panels = [(-1.05, 2.35, 0.7, 0.75), (0.35, 2.4, 0.55, 0.6),
                  (-0.9, 34.2, 0.8, 1.1), (0.55, 35.0, 0.5, 0.6), (-0.3, 5.0, 0.45, 0.35)]
        for x, y, w, h in panels:
            a, b = P(x, y + h), P(x + w, y)
            r = pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1])
            pygame.draw.rect(surf, (18, 18, 20, 200), r, width=max(2, int(0.03 * s)), border_radius=int(0.05 * s))
            hi = r.move(-max(1, int(0.015 * s)), -max(1, int(0.015 * s)))
            pygame.draw.rect(surf, (255, 255, 255, 28), hi, width=max(1, int(0.012 * s)), border_radius=int(0.05 * s))

        # grid-fin actuator housing for the fin facing the viewer
        a, b = P(-0.45, g.fin_hinge_y + 0.35), P(0.45, g.fin_hinge_y - 0.3)
        pygame.draw.rect(surf, (58, 59, 63), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]), border_radius=int(0.06 * s))

        # engine-section heat shield lip
        a, b = P(-R + 0.1, g.skirt_bottom_y + 0.12), P(R - 0.1, g.skirt_bottom_y)
        pygame.draw.rect(surf, (26, 26, 28), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]))

    # --------------------------------------------------------------- nozzles
    def _nozzle_profile(self, t: np.ndarray) -> np.ndarray:
        g = self.g
        k = np.clip(t, 0, 1)
        return g.nozzle_throat_radius + (g.nozzle_exit_radius - g.nozzle_throat_radius) * (1 - (1 - k) ** 2.3)

    def build_nozzle(self) -> Sprite:
        """A single engine bell, pivot at the gimbal point (throat)."""
        g, s = self.g, self.s
        top_block = 0.45                        # turbopump / gimbal block above the throat
        L = g.nozzle_length
        re = g.nozzle_exit_radius
        W, H = int((2 * re + 0.12) * s), int((L + top_block + 0.02) * s)
        xs = -(re + 0.06) + (np.arange(W) + 0.5) / s
        ds = -top_block + (np.arange(H) + 0.5) / s   # distance below throat
        t = ds / L
        hw_row = np.where(ds < 0, 0.30 - 0.06 * (ds < -0.3), self._nozzle_profile(t))
        hw_row = np.where(ds > L, 0, hw_row)
        X = xs[:, None]
        hw = np.broadcast_to(hw_row, (W, H))
        alpha = np.clip((hw - np.abs(X)) * s + 0.5, 0, 1)
        n = X / np.maximum(hw, 1e-4)
        slope = np.gradient(hw_row, ds)
        ny = np.broadcast_to(-slope, (W, H)) * 0.8

        tt = np.broadcast_to(np.clip(t, 0, 1), (W, H))
        heat = np.array([128.0, 102.0, 86.0])
        tint = np.array([84.0, 74.0, 104.0])
        graphite = np.array([58.0, 58.0, 62.0])
        a1 = np.clip(tt / 0.3, 0, 1)[..., None]
        a2 = np.clip((tt - 0.3) / 0.15, 0, 1)[..., None]
        albedo = heat * (1 - a1) + tint * a1
        albedo = albedo * (1 - a2) + graphite * a2
        # regenerative cooling tubes on the upper bell
        tubes = 1 - 0.14 * (0.5 + 0.5 * np.cos(n * math.pi * 22))
        albedo *= np.where(tt < 0.5, tubes, 1.0)[..., None]
        block = np.broadcast_to(ds < 0, (W, H))
        albedo[block] = (70, 70, 74)
        # exit lip
        lip = np.broadcast_to((ds > L - 0.06) & (ds <= L), (W, H))
        albedo[lip] = (150, 148, 150)
        band = np.broadcast_to((ds > L - 0.16) & (ds <= L - 0.06), (W, H))
        albedo[band] *= 0.7
        albedo *= (0.92 + 0.14 * fbm(W, H, 4, 10, seed=41))[..., None]

        diff, sp = cylinder_lighting(n, ny, spec=0.45, shininess=26)
        rgb = albedo * diff[..., None] + 255 * sp[..., None]
        surf = array_to_surface(rgb, alpha)
        return Sprite(downsample(surf), ((re + 0.06) * self.ppm, top_block * self.ppm))

    def nozzle_cluster(self, gimbal_deg: float) -> Sprite:
        """All engines, each gimballed about its own throat. Pivot = body-frame origin."""
        key = round(gimbal_deg * 2) / 2
        if key in self._nozzle_cache:
            return self._nozzle_cache[key]
        if not hasattr(self, "_nozzle"):
            self._nozzle = self.build_nozzle()
        g, ppm = self.g, self.ppm
        x_min, x_max, y_min, y_max = -2.6, 2.6, -0.7, g.nozzle_length + 0.6
        canvas = rgba_surface((x_max - x_min) * ppm, (y_max - y_min) * ppm)
        for x, depth, _ in sorted(g.engine_positions(), key=lambda e: e[1]):
            img = self._nozzle.surface
            k = 0.58 + 0.42 * (depth + g.engine_ring_radius) / (2 * g.engine_ring_radius)
            if k < 0.999:
                img = img.copy()
                img.fill((int(255 * k),) * 3 + (255,), special_flags=pygame.BLEND_RGBA_MULT)
            dst = ((x - x_min) * ppm, (y_max - g.nozzle_throat_y) * ppm)
            blit_rotated(canvas, img, self._nozzle.pivot, dst, key)
        sprite = Sprite(canvas, (-x_min * ppm, y_max * ppm))
        self._nozzle_cache[key] = sprite
        return sprite

    # ------------------------------------------------------------------ legs
    def build_leg(self) -> Sprite:
        """Landing leg drawn pointing up from its hinge (stowed pose). The foot
        pad is a separate sprite so it can swivel level as the leg deploys."""
        g, s = self.g, self.s
        L = g.leg_length
        x_min, x_max, y_min, y_max = -1.3, 1.3, -0.7, L + 1.1
        surf = rgba_surface((x_max - x_min) * s, (y_max - y_min) * s)

        def P(x, y):
            return ((x - x_min) * s, (y_max - y) * s)

        # main carbon beam with a lighter leading edge
        shaded_beam(surf, P(0, 0.1), P(0, L - 0.8), 0.78 * s, 0.42 * s, LEG_CARBON, strips=11, spec=0.28)
        pygame.draw.line(surf, (74, 76, 82), P(-0.28, 0.6), P(-0.15, L - 1.0), max(1, int(0.03 * s)))
        # structural bands
        for f in (0.25, 0.5, 0.75):
            w = 0.78 + (0.42 - 0.78) * f
            a, b = P(-w / 2, L * f + 0.06), P(w / 2, L * f - 0.06)
            pygame.draw.rect(surf, (62, 63, 68), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]))
        # crush-core shock absorber near the foot
        shaded_beam(surf, P(0, L - 1.0), P(0, L), 0.26 * s, 0.24 * s, CHROME, strips=7, spec=0.6, shininess=40)
        # strut attach lug
        lx, ly = P(0, L * g.strut_leg_frac)
        pygame.draw.circle(surf, FITTING, (lx, ly), 0.17 * s)
        pygame.draw.circle(surf, (40, 40, 44), (lx, ly), 0.07 * s)
        # hinge fitting
        a, b = P(-0.42, 0.45), P(0.42, -0.35)
        pygame.draw.rect(surf, (78, 80, 84), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]), border_radius=int(0.08 * s))
        hx, hy = P(0, 0)
        pygame.draw.circle(surf, (130, 132, 136), (hx, hy), 0.18 * s)
        pygame.draw.circle(surf, (35, 35, 38), (hx, hy), 0.08 * s)

        return Sprite(downsample(surf), (-x_min * self.ppm, y_max * self.ppm))

    def build_foot_pad(self) -> Sprite:
        """Level foot pad seen edge-on; pivot at the top centre (leg attachment)."""
        s = self.s
        w, h = 1.5, 0.28
        surf = rgba_surface((w + 0.1) * s, (h + 0.1) * s)
        r = pygame.Rect(0.05 * s, 0.05 * s, w * s, h * s)
        pygame.draw.rect(surf, (66, 67, 71), r, border_radius=int(0.08 * s))
        pygame.draw.line(surf, (150, 152, 156), (r.left + 0.08 * s, r.top + 1), (r.right - 0.08 * s, r.top + 1), max(1, int(0.05 * s)))
        pygame.draw.rect(surf, (40, 40, 44), r.inflate(-0.5 * s, -0.12 * s).move(0, 0.03 * s), border_radius=int(0.04 * s))
        return Sprite(downsample(surf), ((w + 0.1) / 2 * self.ppm, 0.05 * self.ppm))

    def build_strut_sleeve(self) -> Sprite:
        g, s = self.g, self.s
        L = g.strut_sleeve_len
        x_min, y_max = -0.3, L + 0.2
        surf = rgba_surface(0.6 * s, (L + 0.4) * s)

        def P(x, y):
            return ((x - x_min) * s, (y_max - y) * s)

        shaded_beam(surf, P(0, 0), P(0, L), 0.30 * s, 0.28 * s, FITTING, strips=7, spec=0.4)
        for f in (0.02, 0.97):
            a, b = P(-0.17, L * f + 0.1), P(0.17, L * f - 0.05)
            pygame.draw.rect(surf, (70, 72, 76), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]))
        return Sprite(downsample(surf), (-x_min * self.ppm, y_max * self.ppm))

    def build_strut_rod(self) -> Sprite:
        g, s = self.g, self.s
        L = g.strut_rod_len
        x_min, y_max = -0.2, L + 0.1
        surf = rgba_surface(0.4 * s, (L + 0.2) * s)

        def P(x, y):
            return ((x - x_min) * s, (y_max - y) * s)

        shaded_beam(surf, P(0, 0), P(0, L), 0.16 * s, 0.16 * s, CHROME, strips=7, spec=0.7, shininess=50)
        return Sprite(downsample(surf), (-x_min * self.ppm, y_max * self.ppm))

    # ------------------------------------------------------------- grid fins
    def grid_fin_panel(self, face_w: float, face_h: float, band: float, band_side: str,
                       cells=(7, 8)) -> Sprite:
        """Projected grid fin: a see-through lattice (face_w x face_h, already
        foreshortened) plus a solid side-plate band, with a root arm below.
        Pivot = root hinge (bottom centre)."""
        key = (round(face_w, 2), round(face_h, 2), round(band, 2), band_side)
        if key in self._fin_cache:
            return self._fin_cache[key]
        s = self.s
        root = 0.35
        tw = face_w + (band if band_side in ("left", "right") else 0)
        th = face_h + (band if band_side == "top" else 0)
        pad = 0.1
        W, H = (tw + 2 * pad) * s, (th + root + 2 * pad) * s
        surf = rgba_surface(W, H)
        ox = pad + (band if band_side == "left" else 0)   # face left edge (m)
        oy_top = pad + (band if band_side == "top" else 0)  # face top edge (m)
        face = pygame.Rect(ox * s, oy_top * s, face_w * s, face_h * s)
        frame_w = max(2, int(0.07 * s))
        line_w = max(2, int(0.035 * s))
        lattice = (96, 94, 99)
        if face_w * s > 3 and face_h * s > 3:
            # diamond lattice in unprojected fin coordinates (0..1 x 0..1)
            nx, ny_ = cells
            slope = ny_ / nx          # u travelled per unit v, for 45-degree diamonds
            for i in range(-ny_ - 1, nx + ny_ + 2):
                for sign in (1, -1):
                    c = i / nx
                    (u0, v0), (u1, v1) = (c, 0.0), (c + sign * slope, 1.0)
                    # clip param t in [0,1] so that u stays in [0,1]
                    t0, t1 = 0.0, 1.0
                    du = u1 - u0
                    if abs(du) > 1e-9:
                        ta, tb = (0 - u0) / du, (1 - u0) / du
                        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                    elif not (0 <= u0 <= 1):
                        continue
                    if t1 <= t0:
                        continue
                    a = (u0 + du * t0, v0 + (v1 - v0) * t0)
                    b = (u0 + du * t1, v0 + (v1 - v0) * t1)
                    pa = (face.left + a[0] * face.width, face.top + a[1] * face.height)
                    pb = (face.left + b[0] * face.width, face.top + b[1] * face.height)
                    pygame.draw.line(surf, lattice, pa, pb, line_w)
            pygame.draw.rect(surf, TITANIUM, face, width=frame_w)
            hi = face.inflate(-frame_w, -frame_w)
            pygame.draw.rect(surf, (150, 148, 152), hi, width=max(1, frame_w // 3))
        # solid side plate seen edge-on
        if band > 0.005:
            if band_side == "right":
                r = pygame.Rect(face.right, face.top, band * s, face.height)
            elif band_side == "left":
                r = pygame.Rect(face.left - band * s, face.top, band * s, face.height)
            else:
                r = pygame.Rect(face.left, face.top - band * s, face.width, band * s)
            pygame.draw.rect(surf, (128, 126, 131), r)
            pygame.draw.rect(surf, (80, 78, 84), r, width=max(1, frame_w // 2))
        # root arm down to the hinge
        cx = (pad + tw / 2) * s
        arm = pygame.Rect(0, 0, 0.26 * s, (root + 0.05) * s)
        arm.midtop = (cx, (oy_top + face_h) * s - 2)
        pygame.draw.rect(surf, (66, 66, 70), arm)
        pivot_ss = (cx, (oy_top + face_h + root) * s)
        sprite = Sprite(downsample(surf), (pivot_ss[0] / SUPERSAMPLE, pivot_ss[1] / SUPERSAMPLE))
        self._fin_cache[key] = sprite
        return sprite

    def profile_fin(self, twist_deg: float) -> Sprite:
        """Fin on the right edge, drawn stowed (span pointing up). Twist about the
        span axis reveals the lattice face."""
        g = self.g
        a = math.radians(round(twist_deg))
        face_w = g.fin_chord * abs(math.sin(a))
        band = g.fin_thickness * abs(math.cos(a))
        return self.grid_fin_panel(face_w, g.fin_span, band, "right" if twist_deg >= 0 else "left")

    def front_fin(self, deploy: float) -> Sprite:
        """Fin facing the viewer: face-on when stowed, collapses to its tip plate
        as it swings out toward the camera."""
        g = self.g
        d = round(deploy * 40) / 40 * math.pi / 2
        return self.grid_fin_panel(g.fin_chord, g.fin_span * math.cos(d), g.fin_thickness * math.sin(d), "top")
