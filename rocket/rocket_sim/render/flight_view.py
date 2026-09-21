"""On-screen flight view shared by the manual game and the trained-agent viewer:
camera that follows the rocket, world + rocket drawing, crash effects and the HUD."""

import math

import numpy as np
import pygame

from ..physics import RocketPhysics
from .raster import rgba_surface
from .renderer import Camera, RocketRenderer, RocketVisualState
from .scene import SceneRenderer

GOOD, WARN, BAD = (120, 220, 140), (240, 200, 90), (240, 100, 90)


class FlightView:
    """Camera, world, rocket and effects for one RocketPhysics instance."""

    def __init__(self, phys: RocketPhysics, size, pad_radius: float):
        self.phys = phys
        self.size = size
        self.renderer = RocketRenderer(phys.g)
        self.scene = SceneRenderer(*size, pad_radius=pad_radius)
        self.camera = Camera(*size, ppm=6)
        self.zoom = 1.0
        self.reset()

    def reset(self):
        self.renderer.smoke.clear()
        self.cam_pos = None
        self.flash = 0.0
        self.end_time = None

    def visual_state(self) -> RocketVisualState:
        s = self.phys.state
        return RocketVisualState(x=s.x, y=s.y, angle=s.angle, throttle=s.throttle, gimbal=s.gimbal,
                                 engines=s.engines, leg_deploy=s.leg_deploy, fin_deploy=s.fin_deploy,
                                 fin_twist=s.fin_twist)

    def update(self, dt: float, sim_dt: float, just_finished: bool):
        """dt: wall-clock frame time; sim_dt: simulated time that passed this frame."""
        if just_finished:
            self.end_time = 0.0
            if self.phys.status.crashed:
                self.flash = 1.0
                s = self.phys.state
                for _ in range(6):
                    for ang in np.linspace(0, 2 * math.pi, 10, endpoint=False):
                        self.renderer.smoke.emit((s.x, s.y - 8), (math.cos(ang), math.sin(ang) * 0.5 + 0.3),
                                                 400, 0.1, 1.0, 0.0, 30)
        if self.end_time is not None:
            self.end_time += dt
        self.flash = max(0.0, self.flash - dt * 0.8)
        self.renderer.update(self.visual_state(), sim_dt)
        self._update_camera(dt)

    def _update_camera(self, dt):
        s = self.phys.state
        alt = max(0.0, self.phys.altitude())
        # zoom out as altitude grows so the ground comes into view on final approach
        ppm = 9.0 if alt < 30 else max(2.4, min(9.0, 0.42 * self.size[1] / (alt + 25)))
        self.camera.ppm = ppm * self.zoom
        target = np.array([s.x, s.y - 4])
        if self.cam_pos is None:
            self.cam_pos = target
        else:
            self.cam_pos = self.cam_pos + (target - self.cam_pos) * (1 - math.exp(-dt * 8))
        max_off = np.array(self.size) * 0.1 / self.camera.ppm
        self.cam_pos = np.clip(self.cam_pos, target - max_off, target + max_off)
        self.cam_pos[1] = max(self.cam_pos[1], self.size[1] * 0.36 / self.camera.ppm)
        self.camera.center = self.cam_pos

    def draw_world(self, screen):
        self.scene.draw_sky(screen, self.camera, self.camera.center[1])
        self.scene.draw_clouds(screen, self.camera, layer_front=False)
        self.scene.draw_ground(screen, self.camera)
        if self.phys.status.crashed:
            self.renderer.smoke.draw(screen, self.camera)
        else:
            self.renderer.draw(screen, self.camera, self.visual_state())
        self.scene.draw_clouds(screen, self.camera, layer_front=True)
        if self.flash > 0:
            sx, sy = self.camera.world_to_screen((self.phys.state.x, self.phys.state.y - 10))
            r = int((30 + 60 * (1 - self.flash)) * self.camera.ppm)
            f = self.flash
            glow = self.renderer._glow(min(r, 900), (int(255 * f), int(170 * f), int(80 * f)), 1.4)
            screen.blit(glow, glow.get_rect(center=(sx, sy)), special_flags=pygame.BLEND_RGB_ADD)


class Hud:
    def __init__(self):
        self.title = pygame.font.SysFont("helveticaneue,helvetica,arial", 44, bold=True)
        self.big = pygame.font.SysFont("helveticaneue,helvetica,arial", 22, bold=True)
        self.med = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 15)
        self.small = pygame.font.SysFont("helveticaneue,helvetica,arial", 13)

    def panel(self, surface, rect, alpha=160):
        p = rgba_surface(rect.width, rect.height)
        p.fill((8, 12, 20, alpha))
        pygame.draw.rect(p, (255, 255, 255, 40), p.get_rect(), 1)
        surface.blit(p, rect)

    def text(self, surface, font, s, pos, color=(230, 236, 244)):
        surface.blit(font.render(s, True, color), pos)

    def draw(self, screen, view, ctrl, title, help_text, warp=1, paused=False, result_hint=""):
        phys, s, c = view.phys, view.phys.state, ctrl
        p = phys.p
        W, H = screen.get_size()
        alt = max(0.0, phys.altitude())

        # --- flight data
        self.panel(screen, pygame.Rect(16, 16, 330, 330))
        self.text(screen, self.small, title.upper(), (30, 24), (150, 170, 200))
        stop = s.vy * s.vy / (2 * max(0.1, phys.max_decel())) if s.vy < 0 else 0.0
        if phys.status.done:
            burn, burn_col = ("LANDED", GOOD) if phys.status.landed else ("CRASHED", BAD)
        elif s.vy < 0 and phys.max_decel() <= 0:
            burn, burn_col = "CANNOT STOP: more engines", BAD
        elif s.vy < -5 and stop * 1.1 >= alt:
            burn, burn_col = "BURN NOW", BAD
        elif s.vy < -5 and stop * 1.6 >= alt:
            burn, burn_col = "BURN SOON", WARN
        else:
            burn, burn_col = "", GOOD
        self.text(screen, self.big, burn or ("CLIMBING" if s.vy > 1 else "DESCENDING"), (30, 42), burn_col if burn else GOOD)

        def lim_col(v, lim):
            return GOOD if abs(v) <= lim else (WARN if abs(v) <= 2 * lim else BAD)

        rows = [
            ("ALTITUDE", f"{alt / 1000:8.2f} km" if alt >= 1000 else f"{alt:8.1f} m", None),
            ("VERT SPEED", f"{s.vy:+8.1f} m/s", lim_col(s.vy, p.max_touchdown_vy)),
            ("HORIZ SPEED", f"{s.vx:+8.1f} m/s", lim_col(s.vx, p.max_touchdown_vx)),
            ("TILT", f"{math.degrees(s.angle):+8.1f} deg", lim_col(s.angle, p.max_touchdown_angle)),
            ("SPIN", f"{math.degrees(s.omega):+8.1f} deg/s", lim_col(s.omega, p.max_touchdown_omega)),
            ("TO PAD", f"{-s.x:+8.0f} m", GOOD if abs(s.x) < p.pad_radius * 0.6 else WARN),
            ("STOP DIST", f"{stop:8.0f} m", None),
            ("ENGINES", f"{s.engines} x {s.throttle * 100:3.0f}%  TWR {phys.twr():.2f}", None),
            ("GIMBAL", f"{math.degrees(s.gimbal):+8.1f} deg", None),
            ("LEGS", "DEPLOYED" if s.leg_deploy > 0.99 else ("MOVING" if s.leg_deploy > 0 else "STOWED"),
             GOOD if s.leg_deploy > 0.99 else (WARN if alt < 400 else None)),
            ("GRID FINS", "DEPLOYED" if s.fin_deploy > 0.99 else ("MOVING" if s.fin_deploy > 0 else "STOWED"), None),
            ("TIME", f"{s.time:8.1f} s   warp {warp}x", None),
        ]
        y = 78
        for k, v, col in rows:
            self.text(screen, self.small, k, (30, y + 1), (140, 156, 180))
            self.text(screen, self.med, v, (140, y), col or (230, 236, 244))
            y += 21

        # --- fuel and throttle bars
        bx = W - 96
        self.panel(screen, pygame.Rect(bx - 14, 16, 94, 270))
        for i, (label, val, col) in enumerate([("FUEL", s.fuel / phys.fuel_capacity, (110, 190, 255)),
                                               ("THR", s.throttle, (255, 160, 60))]):
            bar = pygame.Rect(bx + i * 42, 30, 16, 210)
            pygame.draw.rect(screen, (50, 58, 72), bar)
            fill = bar.copy()
            fill.height = int(bar.height * max(0.0, min(1.0, val)))
            fill.bottom = bar.bottom
            pygame.draw.rect(screen, col if not (label == "FUEL" and val < 0.15) else BAD, fill)
            if label == "THR" and c.throttle > 0:
                cy = bar.bottom - bar.height * c.throttle
                pygame.draw.line(screen, (255, 255, 255), (bar.left - 4, cy), (bar.right + 4, cy), 2)
            if label == "THR":
                my = bar.bottom - bar.height * p.min_throttle
                pygame.draw.line(screen, (150, 150, 160), (bar.left, my), (bar.right, my), 1)
            lbl = self.small.render(label, True, (150, 170, 200))
            screen.blit(lbl, (bar.centerx - lbl.get_width() / 2, bar.bottom + 8))
        self.text(screen, self.small, f"{s.fuel:,.0f} kg", (bx - 6, 262), (170, 185, 205))

        # --- approach map: position relative to the pad with velocity vector
        m = pygame.Rect(W - 236, 300, 220, 220)
        self.panel(screen, m)
        span = max(300.0, abs(s.x) * 1.3, s.y * 1.1)
        def mp(x, y):
            return (m.centerx + x / span * m.width / 2, m.bottom - 12 - y / span * (m.height - 24))
        pygame.draw.line(screen, (90, 100, 90), (m.left + 6, mp(0, 0)[1]), (m.right - 6, mp(0, 0)[1]), 1)
        pad_w = max(3, p.pad_radius / span * m.width)
        pygame.draw.line(screen, (228, 214, 170), (m.centerx - pad_w / 2, mp(0, 0)[1]), (m.centerx + pad_w / 2, mp(0, 0)[1]), 3)
        rp = mp(max(-span, min(span, s.x)), max(0, min(span, s.y)))
        vs = np.hypot(s.vx, s.vy)
        if vs > 0.5:
            k = 40 / max(vs, 40)
            pygame.draw.line(screen, (255, 200, 110), rp, (rp[0] + s.vx * k, rp[1] - s.vy * k), 2)
        pygame.draw.circle(screen, (240, 244, 250), rp, 4)
        self.text(screen, self.small, f"APPROACH  ({span / 1000:.1f} km)", (m.left + 8, m.top + 6), (150, 170, 200))

        # --- pad direction arrow when the pad is off screen
        px, py = view.camera.world_to_screen((0, p.pad_height))
        if not (0 <= px <= W and 0 <= py <= H):
            cx, cy = W / 2, H / 2
            ang = math.atan2(py - cy, px - cx)
            ax, ay = cx + math.cos(ang) * (min(W, H) / 2 - 40), cy + math.sin(ang) * (min(W, H) / 2 - 40)
            tip = (ax + math.cos(ang) * 14, ay + math.sin(ang) * 14)
            l = (ax + math.cos(ang + 2.5) * 12, ay + math.sin(ang + 2.5) * 12)
            r = (ax + math.cos(ang - 2.5) * 12, ay + math.sin(ang - 2.5) * 12)
            pygame.draw.polygon(screen, (228, 214, 170), [tip, l, r])
            self.text(screen, self.small, "PAD", (ax - 10, ay + 14), (228, 214, 170))

        # --- help / footer
        hs = self.small.render(help_text, True, (170, 182, 200))
        self.panel(screen, pygame.Rect(16, H - 40, hs.get_width() + 24, 28))
        screen.blit(hs, (28, H - 33))
        if paused:
            self.center(screen, "PAUSED", (240, 244, 250), H * 0.4)

        # --- result
        st = phys.status
        if st.done and (view.end_time or 0) > 0.6:
            if st.landed:
                head, col = ("LANDED" if st.reason == "on the pad" else "LANDED OFF PAD"), GOOD
            else:
                head, col = "CRASHED", BAD
            detail = st.reason if st.crashed else f"{st.reason}  |  fuel left {s.fuel:,.0f} kg  |  {s.time:.1f} s"
            box = pygame.Rect(0, 0, max(560, self.med.size(detail)[0] + 60), 150)
            box.center = (W / 2, H * 0.36)
            self.panel(screen, box, 200)
            self.center(screen, head, col, box.top + 20, self.title)
            self.center(screen, detail, (220, 228, 240), box.top + 80, self.med)
            if result_hint:
                self.center(screen, result_hint, (150, 170, 200), box.top + 112, self.small)

    def center(self, screen, s, col, y, font=None):
        img = (font or self.big).render(s, True, col)
        screen.blit(img, (screen.get_width() / 2 - img.get_width() / 2, y))
