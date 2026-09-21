"""Engine exhaust: plume textures, nozzle glow, smoke particles, heat shimmer.

Plume textures are RGB-on-black and are meant to be blitted with
BLEND_RGB_ADD, so they read as emitted light over whatever is behind them.
Their shape depends on ambient pressure: a tight jet with shock diamonds at
sea level, a wide faint bloom near vacuum.
"""

import math

import numpy as np
import pygame

from .geometry import RocketGeometry
from .raster import fbm, rgba_surface, value_noise

PRESSURE_LEVELS = (1.0, 0.6, 0.3, 0.1, 0.02)

_STOPS = np.array([0.0, 0.10, 0.28, 0.48, 0.68, 0.86, 1.0])
_COLORS = np.array([
    (0, 0, 0), (60, 14, 3), (190, 62, 10), (255, 132, 34),
    (255, 200, 105), (255, 238, 205), (255, 253, 245)], dtype=float)


def _colormap(T: np.ndarray) -> np.ndarray:
    T = np.clip(T, 0, 1)
    out = np.empty(T.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(T, _STOPS, _COLORS[:, c])
    return out


def nearest_pressure_level(p: float) -> float:
    return min(PRESSURE_LEVELS, key=lambda lv: abs(math.log(max(lv, 1e-3)) - math.log(max(p, 1e-3))))


class PlumeFactory:
    """Builds and caches plume textures. Pivot = nozzle exit centre (top middle)."""

    VARIANTS = 6

    def __init__(self, geometry: RocketGeometry, ppm: float):
        self.g = geometry
        self.ppm = ppm * 0.6          # plume is soft; render it at reduced resolution
        self._cache = {}

    def plume_length(self, pressure: float) -> float:
        return 24.0 + 22.0 * (1 - pressure)

    def texture(self, engines: int, pressure: float, variant: int):
        p = nearest_pressure_level(pressure)
        key = (engines, p, variant % self.VARIANTS)
        if key not in self._cache:
            self._cache[key] = self._build(engines, p, variant % self.VARIANTS)
        return self._cache[key]

    def _jet_xs(self, engines: int):
        g = self.g
        active = g.active_engine_set(engines)
        xs = [round(x, 3) for x, _, i in g.engine_positions() if i in active]
        # engines hidden behind each other in projection add brightness, not width
        uniq = {}
        for x in xs:
            uniq[x] = uniq.get(x, 0) + 1
        return list(uniq.items())

    def _build(self, engines: int, p: float, variant: int):
        g, ppm = self.g, self.ppm
        re = g.nozzle_exit_radius
        Lp = self.plume_length(p)
        spread = 0.075 + 0.5 * (1 - p) ** 1.5          # jet half-angle growth
        jets = self._jet_xs(engines)
        x_ext = max(abs(x) for x, _ in jets) + (re + Lp * spread) * 1.6 + 1.0
        W, H = int(2 * x_ext * ppm), int((Lp + 1.0) * ppm)
        u = -x_ext + (np.arange(W) + 0.5) / ppm
        v = -0.4 + (np.arange(H) + 0.5) / ppm
        U, V = np.meshgrid(u, v, indexing="ij")
        Vc = np.maximum(V, 0)

        r = re + Vc * spread
        sigma = r * 0.62
        turb = fbm(W, H, 5, Lp / 2.2, seed=100 + variant, octaves=3)
        flick = value_noise(W, H, 2, Lp / 5, seed=200 + variant)
        axial = np.exp(-Vc / (Lp * 0.55)) * np.clip(1 - (Vc / Lp) ** 3, 0, 1)
        dilution = re / r                                 # energy spreads as it expands
        start = np.clip((V + 0.4) / 0.5, 0, 1)            # soft start just inside the bell

        jet_sum = np.zeros((W, H))
        core_sum = np.zeros((W, H))
        halo_sum = np.zeros((W, H))
        diamond_sum = np.zeros((W, H))
        lam = 2.4 * re * (1.2 + 0.8 * (1 - p))            # shock-cell spacing
        n_idx = np.round(Vc / lam)
        cell = np.exp(-((Vc - n_idx * lam) ** 2) / (2 * (0.16 * lam) ** 2)) * np.exp(-n_idx / 2.5) * (n_idx >= 1)
        for x, weight in jets:
            w = 1.0 + 0.25 * (weight - 1)
            d2 = (U - x) ** 2
            jet_sum += w * np.exp(-d2 / (2 * sigma ** 2))
            halo_sum += w * np.exp(-d2 / (2 * (2.6 * sigma + 0.6) ** 2))
            core_r = re * (0.55 - 0.25 * np.clip(Vc / 2.0, 0, 1))
            core_sum += w * np.exp(-d2 / (2 * core_r ** 2)) * np.exp(-Vc / (1.6 + 1.5 * (1 - p)))
            diamond_sum += w * np.exp(-d2 / (2 * (0.33 * r) ** 2)) * cell

        jet_sum = np.minimum(jet_sum, 1.8)
        # the gas column necks down and breaks up toward the tail
        taper = np.clip(1 - (Vc / Lp) ** 1.4, 0, 1)
        heat = (1.05 * core_sum
                + 0.75 * jet_sum * axial * dilution ** 0.6 * (0.45 + 0.95 * turb) * taper
                + 0.45 * p ** 2 * diamond_sum
                + 0.12 * halo_sum * axial * (0.7 + 0.6 * flick))
        heat *= start
        T = 1 - np.exp(-1.15 * heat)                       # tone-map
        # ragged, turbulent tail
        tail = np.clip((Vc / Lp - 0.7) / 0.3, 0, 1)
        T *= 1 - tail * (0.6 + 0.4 * (turb > 0.5))
        rgb = _colormap(T) * np.clip(T * 2.2, 0, 1)[..., None]

        surf = pygame.Surface((W, H))
        pygame.surfarray.blit_array(surf, np.clip(rgb, 0, 255).astype(np.uint8))
        return surf, (x_ext * ppm, 0.4 * ppm), ppm


def radial_glow(radius_px: int, color, falloff: float = 2.2) -> pygame.Surface:
    """RGB-on-black soft glow for additive blending."""
    size = max(2, int(radius_px * 2))
    c = (np.arange(size) + 0.5 - size / 2) / (size / 2)
    d = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2)
    k = np.clip(1 - d, 0, 1) ** falloff
    rgb = np.array(color, dtype=float)[None, None, :] * k[..., None]
    surf = pygame.Surface((size, size))
    pygame.surfarray.blit_array(surf, rgb.astype(np.uint8))
    return surf


def heat_shimmer(surface: pygame.Surface, rect: pygame.Rect, strength: float, t: float):
    """Refraction-like wobble: shift pixel rows sideways inside rect."""
    rect = rect.clip(surface.get_rect())
    if strength <= 0.05 or rect.width < 4 or rect.height < 4:
        return
    sub = surface.subsurface(rect)
    arr = pygame.surfarray.pixels3d(sub)
    h = arr.shape[1]
    rows = np.arange(h)
    amp = strength * (0.4 + 0.6 * rows / h)
    shift = np.round(amp * np.sin(rows * 0.21 + t * 17.0) + 0.6 * amp * np.sin(rows * 0.047 - t * 9.0)).astype(int)
    idx = (np.arange(arr.shape[0])[:, None] - shift[None, :]) % arr.shape[0]
    arr[...] = arr[idx, rows[None, :]]
    del arr


class SmokeSystem:
    """World-space exhaust smoke / steam. Thins out with altitude, spreads along the ground."""

    MAX = 260

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self.age = np.zeros(0)
        self.life = np.zeros(0)
        self.size = np.zeros(0)
        self.grow = np.zeros(0)
        self.shade = np.zeros(0)
        self.kind = np.zeros(0, dtype=int)
        self._carry = 0.0
        self._puffs = [self._make_puff(128, seed + k) for k in range(4)]
        self._scaled = {}

    @staticmethod
    def _make_puff(size: int, seed: int) -> pygame.Surface:
        c = (np.arange(size) + 0.5 - size / 2) / (size / 2)
        X, Y = np.meshgrid(c, c, indexing="ij")
        d = np.sqrt(X ** 2 + Y ** 2)
        n = fbm(size, size, 4, 4, seed=seed, octaves=4)
        a = np.clip(1 - d * (0.85 + 0.45 * n), 0, 1) ** 1.6
        light = np.clip(0.78 + 0.35 * (-X * 0.55 - Y * 0.8) + 0.2 * (n - 0.5), 0.45, 1.15)
        rgb = np.stack([205 * light, 203 * light, 200 * light], axis=-1)
        surf = rgba_surface(size, size)
        px = pygame.surfarray.pixels3d(surf)
        px[...] = np.clip(rgb, 0, 255).astype(np.uint8)
        del px
        pa = pygame.surfarray.pixels_alpha(surf)
        pa[...] = (a * 255).astype(np.uint8)
        del pa
        return surf

    def clear(self):
        self.__init__(int(self.rng.integers(1 << 30)))

    def emit(self, origin, direction, rate: float, dt: float, pressure: float,
             ground_y: float, plume_len: float):
        """origin: nozzle exit (world m); direction: unit vector of exhaust flow."""
        if pressure < 0.03 or rate <= 0:
            return
        near_ground = origin[1] - ground_y < plume_len * 1.4
        rate *= pressure * (2.8 if near_ground else 1.0)
        self._carry += rate * dt
        n = int(self._carry)
        self._carry -= n
        if n <= 0:
            return
        n = min(n, self.MAX)
        rng = self.rng
        dx, dy = direction
        along = rng.uniform(0.15, 0.55, n) * plume_len
        pos = np.stack([origin[0] + dx * along, origin[1] + dy * along], axis=1)
        pos += rng.normal(0, 1.2, (n, 2))
        speed = rng.uniform(18, 55, n)
        vel = np.stack([dx * speed, dy * speed], axis=1) + rng.normal(0, 6, (n, 2))
        self.pos = np.concatenate([self.pos, pos])[-self.MAX:]
        self.vel = np.concatenate([self.vel, vel])[-self.MAX:]
        self.age = np.concatenate([self.age, np.zeros(n)])[-self.MAX:]
        self.life = np.concatenate([self.life, rng.uniform(2.5, 5.5, n) * (1.3 if near_ground else 1.0)])[-self.MAX:]
        self.size = np.concatenate([self.size, rng.uniform(2.2, 4.0, n)])[-self.MAX:]
        self.grow = np.concatenate([self.grow, rng.uniform(1.5, 4.0, n)])[-self.MAX:]
        self.shade = np.concatenate([self.shade, rng.uniform(0.75, 1.0, n)])[-self.MAX:]
        self.kind = np.concatenate([self.kind, rng.integers(0, 4, n)])[-self.MAX:]

    def update(self, dt: float, ground_y: float):
        if len(self.age) == 0:
            return
        self.age += dt
        keep = self.age < self.life
        for name in ("pos", "vel", "age", "life", "size", "grow", "shade", "kind"):
            setattr(self, name, getattr(self, name)[keep])
        self.vel *= math.exp(-1.3 * dt)
        self.vel[:, 1] += 1.8 * dt                      # buoyancy
        self.pos += self.vel * dt
        self.size = np.minimum(self.size + self.grow * dt, 11.0)
        # ground deflection: turn vertical momentum into sideways billow
        floor = ground_y + self.size * 0.35
        hit = self.pos[:, 1] < floor
        if hit.any():
            side = np.where(np.abs(self.vel[hit, 0]) > 1, np.sign(self.vel[hit, 0]),
                            self.rng.choice([-1.0, 1.0], hit.sum()))
            self.vel[hit, 0] += side * np.abs(self.vel[hit, 1]) * 0.9
            self.vel[hit, 1] = np.abs(self.vel[hit, 1]) * 0.05
            self.pos[hit, 1] = floor[hit]
            self.grow[hit] = np.maximum(self.grow[hit], 5.0)

    def draw(self, surface: pygame.Surface, camera):
        if len(self.age) == 0:
            return
        sw, sh = surface.get_size()
        order = np.argsort(-self.age)                   # oldest first
        for i in order:
            d_px = int(self.size[i] * 2 * camera.ppm)
            if d_px < 2:
                continue
            sx, sy = camera.world_to_screen(self.pos[i])
            if sx < -d_px or sy < -d_px or sx > sw + d_px or sy > sh + d_px:
                continue
            d_q = max(2, (d_px // 3) * 3)
            key = (int(self.kind[i]), d_q)
            img = self._scaled.get(key)
            if img is None:
                img = pygame.transform.smoothscale(self._puffs[self.kind[i]], (d_q, d_q))
                if len(self._scaled) > 400:
                    self._scaled.clear()
                self._scaled[key] = img
            f = self.age[i] / self.life[i]
            alpha = (min(1.0, self.age[i] * 4) * (1 - f) ** 1.5) * 225 * self.shade[i]
            img.set_alpha(int(alpha))
            surface.blit(img, (sx - d_q / 2, sy - d_q / 2))
