"""Low-level raster helpers: numpy <-> surface, noise, shading, rotated blits."""

import math

import numpy as np
import pygame

# Art is drawn at SUPERSAMPLE x resolution and smooth-scaled down for anti-aliasing.
SUPERSAMPLE = 3

LIGHT_DIR = np.array([-0.55, 0.35, 0.76])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
HALF_VEC = LIGHT_DIR + np.array([0.0, 0.0, 1.0])
HALF_VEC = HALF_VEC / np.linalg.norm(HALF_VEC)


def rgba_surface(w: int, h: int) -> pygame.Surface:
    surf = pygame.Surface((max(1, int(w)), max(1, int(h))), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    return surf


def array_to_surface(rgb: np.ndarray, alpha: np.ndarray) -> pygame.Surface:
    """rgb: (w, h, 3) floats in 0..255, alpha: (w, h) floats in 0..1."""
    w, h = alpha.shape
    surf = rgba_surface(w, h)
    px = pygame.surfarray.pixels3d(surf)
    px[...] = np.clip(rgb, 0, 255).astype(np.uint8)
    del px
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[...] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    del pa
    return surf


def downsample(surf: pygame.Surface, factor: int = SUPERSAMPLE) -> pygame.Surface:
    w, h = surf.get_size()
    return pygame.transform.smoothscale(surf, (max(1, round(w / factor)), max(1, round(h / factor))))


def value_noise(w: int, h: int, cells_x: float, cells_y: float, seed: int) -> np.ndarray:
    """Smooth value noise in [0, 1], shape (w, h). Bilinear over a random lattice."""
    rng = np.random.default_rng(seed)
    gx, gy = max(2, int(cells_x) + 2), max(2, int(cells_y) + 2)
    grid = rng.random((gx, gy))
    xs = np.linspace(0, gx - 1.001, w)
    ys = np.linspace(0, gy - 1.001, h)
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = xs - x0, ys - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = grid[x0][:, y0]
    b = grid[x0 + 1][:, y0]
    c = grid[x0][:, y0 + 1]
    d = grid[x0 + 1][:, y0 + 1]
    top = a + (b - a) * fx[:, None]
    bot = c + (d - c) * fx[:, None]
    return top + (bot - top) * fy[None, :]


def fbm(w: int, h: int, cells_x: float, cells_y: float, seed: int, octaves: int = 4) -> np.ndarray:
    total = np.zeros((w, h))
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(w, h, cells_x * 2 ** o, cells_y * 2 ** o, seed + 97 * o)
        norm += amp
        amp *= 0.5
    return total / norm


def cylinder_lighting(n: np.ndarray, ny: np.ndarray | float = 0.0,
                      ambient: float = 0.30, diffuse: float = 0.80,
                      spec: float = 0.35, shininess: float = 24.0):
    """Lighting terms for a vertical cylinder.

    n  : normalised horizontal position across the cylinder, -1 (left) .. 1 (right)
    ny : upward tilt of the surface normal (for cones), 0 for a straight cylinder
    Returns (diffuse_factor, specular_term) arrays.
    """
    n = np.clip(n, -0.999, 0.999)
    nz = np.sqrt(1.0 - n * n)
    ny = np.asarray(ny, dtype=float)
    norm = np.sqrt(n * n + ny * ny + nz * nz)
    nx_, ny_, nz_ = n / norm, ny / norm, nz / norm
    ndl = np.maximum(0.0, nx_ * LIGHT_DIR[0] + ny_ * LIGHT_DIR[1] + nz_ * LIGHT_DIR[2])
    ndh = np.maximum(0.0, nx_ * HALF_VEC[0] + ny_ * HALF_VEC[1] + nz_ * HALF_VEC[2])
    # soft rim light from the environment on the shadow side
    rim = np.clip(n, 0, 1) ** 6 * 0.12
    return ambient + diffuse * ndl + rim, spec * ndh ** shininess


def shade_scalar(n: float, **kw) -> tuple[float, float]:
    d, s = cylinder_lighting(np.array([n]), **kw)
    return float(d[0]), float(s[0])


def lit_color(base, n: float, spec_color=(255, 255, 255), **kw):
    d, s = shade_scalar(n, **kw)
    return tuple(int(min(255, base[i] * d + spec_color[i] * s)) for i in range(3))


def shaded_beam(surf: pygame.Surface, p0, p1, w0: float, w1: float, base,
                strips: int = 9, spec: float = 0.30, shininess: float = 18.0,
                ambient: float = 0.30):
    """Draw a round, tapered beam from p0 to p1 (pixel coords) with cylindrical shading."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    # perpendicular in screen space; we want "left" in the lighting sense to be the
    # side that faces the upper-left light, so orient by screen x
    px, py = -dy / length, dx / length
    if px > 0 or (abs(px) < 1e-9 and py > 0):
        px, py = -px, -py
    # (px, py) now points toward screen-left; strip parameter -1 = left edge (lit)
    for i in range(strips):
        a = -1 + 2 * i / strips
        b = -1 + 2 * (i + 1) / strips
        col = lit_color(base, (a + b) / 2, spec=spec, shininess=shininess, ambient=ambient)
        quad = [
            (x0 - px * a * w0 / 2, y0 - py * a * w0 / 2),
            (x0 - px * b * w0 / 2, y0 - py * b * w0 / 2),
            (x1 - px * b * w1 / 2, y1 - py * b * w1 / 2),
            (x1 - px * a * w1 / 2, y1 - py * a * w1 / 2),
        ]
        pygame.draw.polygon(surf, col, quad)


def blit_rotated(dst: pygame.Surface, img: pygame.Surface, pivot_in_img, dst_pos,
                 angle_deg: float, zoom: float = 1.0, special_flags: int = 0) -> pygame.Rect:
    """Rotate `img` counter-clockwise by angle_deg around pivot_in_img and place that
    pivot at dst_pos."""
    w, h = img.get_size()
    cx, cy = w / 2, h / 2
    ox, oy = (pivot_in_img[0] - cx) * zoom, (pivot_in_img[1] - cy) * zoom
    a = math.radians(angle_deg)
    # pygame rotates CCW visually; in y-down pixel space that is this matrix:
    rx = ox * math.cos(a) + oy * math.sin(a)
    ry = -ox * math.sin(a) + oy * math.cos(a)
    if abs(angle_deg) < 1e-3 and abs(zoom - 1.0) < 1e-6:
        rot = img
    else:
        rot = pygame.transform.rotozoom(img, angle_deg, zoom)
    rect = rot.get_rect(center=(dst_pos[0] - rx, dst_pos[1] - ry))
    return dst.blit(rot, rect, special_flags=special_flags)
