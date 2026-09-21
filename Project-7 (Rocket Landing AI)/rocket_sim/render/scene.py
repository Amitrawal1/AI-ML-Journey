"""World backdrop: altitude-dependent sky, stars, clouds, terrain and the landing pad."""

import math

import numpy as np
import pygame

from .exhaust import SmokeSystem
from .raster import rgba_surface

# (altitude m, zenith colour, horizon colour)
_SKY = [
    (0, (62, 104, 156), (196, 184, 168)),
    (6000, (40, 78, 138), (130, 160, 196)),
    (18000, (14, 30, 72), (58, 92, 150)),
    (40000, (4, 7, 20), (18, 32, 70)),
    (90000, (1, 2, 6), (6, 10, 26)),
]


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))


def sky_colors(alt: float):
    for (a0, z0, h0), (a1, z1, h1) in zip(_SKY, _SKY[1:]):
        if alt <= a1:
            t = max(0.0, (alt - a0) / (a1 - a0))
            return _lerp(z0, z1, t), _lerp(h0, h1, t)
    return _SKY[-1][1], _SKY[-1][2]


class SceneRenderer:
    PAD_RADIUS = 150.0
    PAD_HEIGHT = 1.2

    def __init__(self, width: int, height: int, seed: int = 7, pad_radius: float | None = None):
        self.w, self.h = width, height
        self.pad_radius = pad_radius or self.PAD_RADIUS
        rng = np.random.default_rng(seed)
        self.stars = [(rng.random() * width, rng.random() * height, rng.random() ** 3) for _ in range(260)]
        self.clouds = [(rng.uniform(-40000, 40000), rng.uniform(1400, 5200), rng.uniform(250, 900),
                        int(rng.integers(0, 4)), rng.uniform(0.25, 0.6)) for _ in range(120)]
        self._puffs = [SmokeSystem._make_puff(256, seed + k) for k in range(4)]
        self._puff_cache = {}
        # distant skyline / hills, as (x_world, height) samples for a parallax layer
        xs = np.arange(-60000, 60001, 400.0)
        self.hills = (xs, 60 + 50 * np.sin(xs / 3100.0) + 35 * np.sin(xs / 1170.0 + 1.3) + 20 * np.sin(xs / 530.0))
        self.structures = [(x, h, w) for x, h, w in [(-420, 55, 14), (-380, 22, 40), (-610, 90, 6),
                                                       (260, 30, 60), (330, 18, 30), (780, 70, 8)]]

    # ------------------------------------------------------------------ sky
    def draw_sky(self, surface: pygame.Surface, camera, cam_alt: float):
        zen, hor = sky_colors(cam_alt)
        grad = pygame.Surface((1, 64))
        for i in range(64):
            t = (i / 63) ** 1.6
            grad.set_at((0, i), [int(c) for c in _lerp(zen, hor, t)])
        surface.blit(pygame.transform.smoothscale(grad, (self.w, self.h)), (0, 0))
        star_k = min(1.0, max(0.0, (cam_alt - 15000) / 25000))
        if star_k > 0:
            for x, y, b in self.stars:
                v = int(255 * star_k * (0.25 + 0.75 * b))
                surface.set_at((int(x), int(y)), (v, v, min(255, v + 10)))

    def draw_clouds(self, surface: pygame.Surface, camera, layer_front: bool):
        for i, (x, y, size, kind, a) in enumerate(self.clouds):
            if (i % 3 == 0) != layer_front:
                continue
            d = int(size * camera.ppm)
            if d < 4:
                continue
            sx, sy = camera.world_to_screen((x, y))
            if sx < -d or sx > self.w + d or sy < -d or sy > self.h + d:
                continue
            dq = min(4000, max(4, (d // 8) * 8))
            key = (kind, dq)
            img = self._puff_cache.get(key)
            if img is None:
                if len(self._puff_cache) > 60:
                    self._puff_cache.clear()
                img = pygame.transform.smoothscale(self._puffs[kind], (dq, int(dq * 0.45)))
                self._puff_cache[key] = img
            img.set_alpha(int(255 * a))
            surface.blit(img, (sx - dq / 2, sy - dq * 0.225))

    # --------------------------------------------------------------- ground
    def draw_ground(self, surface: pygame.Surface, camera, ground_y: float = 0.0):
        gx, gy = camera.world_to_screen((0, ground_y))
        if gy > self.h + 400 * camera.ppm:
            return
        # distant hills (parallax) sit on the horizon
        par = 0.25
        cx = camera.center[0] * par
        xs, hs = self.hills
        pts = []
        for x, hgt in zip(xs, hs):
            sx = self.w / 2 + (x - cx) * camera.ppm * par
            if -50 < sx < self.w + 50:
                pts.append((sx, gy - hgt * camera.ppm * par))
        if len(pts) > 2:
            pygame.draw.polygon(surface, (78, 88, 96), [(pts[0][0], gy + 2)] + pts + [(pts[-1][0], gy + 2)])
        # structures near the pad
        for x, hgt, w in self.structures:
            a = camera.world_to_screen((x - w / 2, ground_y + hgt))
            b = camera.world_to_screen((x + w / 2, ground_y))
            pygame.draw.rect(surface, (52, 58, 64), pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]))
        # ground plane
        if gy < self.h:
            top = max(0, int(gy))
            band = pygame.Surface((1, 32))
            for i in range(32):
                t = i / 31
                band.set_at((0, i), [int(c) for c in _lerp((84, 86, 78), (40, 42, 38), t)])
            surface.blit(pygame.transform.smoothscale(band, (self.w, self.h - top + 1)), (0, top))
        self._draw_pad(surface, camera, ground_y)

    def _draw_pad(self, surface, camera, ground_y):
        R, H = self.pad_radius, self.PAD_HEIGHT
        a = camera.world_to_screen((-R, ground_y + H))
        b = camera.world_to_screen((R, ground_y))
        rect = pygame.Rect(a[0], a[1], b[0] - a[0], max(2, b[1] - a[1]))
        pygame.draw.rect(surface, (150, 150, 146), rect)
        pygame.draw.line(surface, (196, 196, 190), rect.topleft, rect.topright, max(1, int(0.25 * camera.ppm)))
        # the pad surface seen at a grazing angle: a thin lit ellipse with target ring
        # (polygons instead of pygame ellipses, which step badly at extreme aspect ratios)
        ell_h = max(3, 6 * camera.ppm)
        cx, cy = rect.centerx, rect.top + 1 - ell_h / 2

        def ellipse_pts(rx, ry, n=72):
            return [(cx + rx * math.cos(t), cy + ry * math.sin(t))
                    for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]

        pygame.draw.polygon(surface, (170, 170, 165), ellipse_pts(rect.width / 2, ell_h / 2))
        pygame.draw.lines(surface, (228, 214, 170), True, ellipse_pts(rect.width * 0.275, ell_h * 0.275),
                          max(1, int(0.4 * camera.ppm)))
        # perimeter lights
        n = max(9, int(R / 12))
        for i in range(n):
            x = -R + (2 * R) * i / (n - 1)
            sx, sy = camera.world_to_screen((x, ground_y + H + 0.3))
            pygame.draw.circle(surface, (255, 190, 120), (sx, sy), max(1, int(0.35 * camera.ppm)))

    def rest_height(self, com_y: float, foot_drop: float) -> float:
        """CoM height when standing on the pad with legs deployed."""
        return self.PAD_HEIGHT + foot_drop + 0.28 + com_y
