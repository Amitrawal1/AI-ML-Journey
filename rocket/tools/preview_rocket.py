"""Visual preview of the rocket asset.

This is NOT the simulation: the flight below is a hand-keyframed timeline used
only to show every visual state the renderer supports. The Gymnasium env will
later drive the same RocketVisualState from real physics.

    python tools/preview_rocket.py                    # interactive viewer
    python tools/preview_rocket.py --contact-sheet out.png
    python tools/preview_rocket.py --export assets/rocket

Keys: SPACE pause | LEFT/RIGHT scrub | 1-7 jump to phase | R restart
      M manual mode (UP/DOWN throttle, A/D gimbal, Q/E tilt, L legs, F fins, G engines)
      +/- zoom | S screenshot | ESC quit
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pygame

from rocket_sim.render import Camera, RocketGeometry, RocketRenderer, RocketVisualState
from rocket_sim.render.exhaust import PlumeFactory
from rocket_sim.render.scene import SceneRenderer

GEOM = RocketGeometry()
PAD_TOP = SceneRenderer.PAD_HEIGHT
LAUNCH_Y = PAD_TOP + 1.6 + GEOM.com_y          # nozzles clear of the pad on the launch mount
LANDED_Y = SceneRenderer(1, 1).rest_height(GEOM.com_y, GEOM.foot_drop)


@dataclass
class Key:
    t: float
    x: float
    y: float
    angle: float = 0.0       # deg, CCW positive
    throttle: float = 0.0
    engines: int = 9
    gimbal: float = 0.0      # deg
    legs: float = 0.0
    fins: float = 0.0
    twist: float = 0.0       # deg
    ppm: float = 7.0
    phase: str = ""


# A compressed, stylised mission profile (altitudes in metres).
TIMELINE = [
    Key(0.0, 0, LAUNCH_Y, 0, 0.0, 9, 0, 0, 0, 0, 7.0, "LAUNCH"),
    Key(2.2, 0, LAUNCH_Y, 0, 1.0, 9, 0, 0, 0, 0, 7.0, "LAUNCH"),
    Key(4.5, 0, 160, 0, 1.0, 9, 1.5, 0, 0, 0, 6.5, "LAUNCH"),
    Key(8.0, 400, 3500, -8, 1.0, 9, -1.0, 0, 0, 0, 6.0, "POWERED ASCENT"),
    Key(12.0, 6000, 24000, -24, 0.9, 9, 0.5, 0, 0, 0, 6.0, "POWERED ASCENT"),
    Key(15.0, 16000, 52000, -32, 0.8, 9, 0, 0, 0, 0, 6.0, "POWERED ASCENT"),
    Key(15.6, 17500, 55000, -32, 0.0, 9, 0, 0, 0, 0, 6.0, "ENGINE CUTOFF"),
    Key(17.5, 22000, 62000, -20, 0.0, 9, 0, 0, 0, 0, 6.0, "ENGINE CUTOFF"),
    Key(21.0, 27000, 68000, 20, 0.0, 9, 0, 0, 0, 0, 6.0, "COAST / APOGEE"),
    Key(24.0, 28000, 64000, 12, 0.0, 9, 0, 0, 1, 0, 6.0, "COAST / APOGEE"),
    Key(26.0, 26000, 48000, 6, 0.0, 3, 0, 0, 1, 8, 6.0, "CONTROLLED DESCENT"),
    Key(26.3, 25500, 45500, 6, 0.9, 3, 2, 0, 1, 0, 6.0, "CONTROLLED DESCENT"),
    Key(28.3, 20000, 32000, 4, 0.9, 3, -2, 0, 1, -6, 6.0, "CONTROLLED DESCENT"),
    Key(28.6, 19000, 30500, 4, 0.0, 3, 0, 0, 1, 12, 6.0, "CONTROLLED DESCENT"),
    Key(32.0, 7000, 9000, 3, 0.0, 1, 0, 0, 1, -14, 6.5, "CONTROLLED DESCENT"),
    Key(34.0, 2500, 3200, 8, 0.0, 1, 0, 0, 1, 10, 7.0, "CONTROLLED DESCENT"),
    Key(34.4, 2000, 2600, 8, 1.0, 1, -4, 0, 1, 6, 7.5, "LANDING BURN"),
    Key(37.0, 350, 700, 5, 0.95, 1, 3, 0, 1, -8, 8.0, "LANDING BURN"),
    Key(39.0, 40, 160, -2, 0.8, 1, -2, 1, 1, 4, 9.0, "LANDING BURN"),
    Key(41.5, 0, LANDED_Y + 6, 0, 0.62, 1, 0.5, 1, 1, 0, 9.5, "FINAL LANDING"),
    Key(42.6, 0, LANDED_Y, 0, 0.45, 1, 0, 1, 1, 0, 9.5, "FINAL LANDING"),
    Key(42.8, 0, LANDED_Y, 0, 0.0, 1, 0, 1, 1, 0, 9.5, "FINAL LANDING"),
    Key(50.0, 0, LANDED_Y, 0, 0.0, 1, 0, 1, 1, 0, 9.5, "FINAL LANDING"),
]
PHASES = ["LAUNCH", "POWERED ASCENT", "ENGINE CUTOFF", "COAST / APOGEE",
          "CONTROLLED DESCENT", "LANDING BURN", "FINAL LANDING"]
PHASE_SNAPSHOT_T = [3.4, 10.0, 16.2, 22.0, 27.0, 36.5, 44.0]
DURATION = TIMELINE[-1].t


def _catmull(p0, p1, p2, p3, u):
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u)


def sample(t: float):
    """Interpolate the timeline -> (RocketVisualState, phase, camera ppm)."""
    t = min(max(t, 0.0), DURATION)
    i = max(0, min(len(TIMELINE) - 2, next((k for k in range(len(TIMELINE) - 1) if TIMELINE[k + 1].t >= t), 0)))
    a, b = TIMELINE[i], TIMELINE[i + 1]
    u = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
    k0, k3 = TIMELINE[max(0, i - 1)], TIMELINE[min(len(TIMELINE) - 1, i + 2)]
    # positions follow a smooth spline; control inputs ease between keys
    x = _catmull(k0.x, a.x, b.x, k3.x, u)
    y = _catmull(k0.y, a.y, b.y, k3.y, u)
    # no spline overshoot past the segment's end points (e.g. below the pad at liftoff)
    x = min(max(x, min(a.x, b.x)), max(a.x, b.x))
    y = min(max(y, min(a.y, b.y)), max(a.y, b.y))
    s = u * u * (3 - 2 * u)

    def mix(f):
        return getattr(a, f) + (getattr(b, f) - getattr(a, f)) * s

    engines = b.engines if b.throttle > a.throttle else a.engines
    wob = math.sin(t * 3.1) * 0.8 + math.sin(t * 7.3) * 0.4
    firing = mix("throttle") > 0.01
    state = RocketVisualState(
        x=x, y=y, angle=math.radians(mix("angle")),
        throttle=mix("throttle"), engines=engines,
        gimbal=math.radians(mix("gimbal") + (wob if firing and y > LANDED_Y + 2 else 0)),
        leg_deploy=mix("legs"), fin_deploy=mix("fins"),
        fin_twist=math.radians(mix("twist")))
    return state, a.phase if u < 1 else b.phase, mix("ppm")


class Hud:
    def __init__(self):
        self.big = pygame.font.SysFont("helveticaneue,helvetica,arial", 26, bold=True)
        self.med = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 15)
        self.small = pygame.font.SysFont("helveticaneue,helvetica,arial", 13)

    def panel(self, surface, rect, alpha=150):
        p = pygame.Surface(rect.size, pygame.SRCALPHA)
        p.fill((8, 12, 20, alpha))
        pygame.draw.rect(p, (255, 255, 255, 40), p.get_rect(), 1)
        surface.blit(p, rect)

    def draw(self, surface, state, phase, t, velocity, manual, paused):
        W, H = surface.get_size()
        self.panel(surface, pygame.Rect(16, 16, 300, 208))
        surface.blit(self.small.render("MISSION PHASE", True, (150, 170, 200)), (30, 26))
        surface.blit(self.big.render(phase if not manual else "MANUAL", True, (240, 244, 250)), (30, 42))
        alt = state.y - LANDED_Y
        rows = [
            ("T+", f"{t:6.1f} s"),
            ("ALTITUDE", f"{alt / 1000:7.2f} km" if abs(alt) >= 1000 else f"{alt:7.1f} m"),
            ("VELOCITY", f"{np.hypot(*velocity):7.0f} m/s"),
            ("ATTITUDE", f"{math.degrees(state.angle):+6.1f} deg"),
            ("GIMBAL", f"{math.degrees(state.gimbal):+6.1f} deg"),
            ("ENGINES", f"{state.engines if state.throttle > 0.01 else 0} lit"),
            ("LEGS / FINS", f"{state.leg_deploy * 100:3.0f}% / {state.fin_deploy * 100:3.0f}%"),
        ]
        y = 80
        for k, v in rows:
            surface.blit(self.small.render(k, True, (140, 156, 180)), (30, y + 1))
            surface.blit(self.med.render(v, True, (230, 236, 244)), (150, y))
            y += 19
        # throttle bar
        bar = pygame.Rect(W - 44, 60, 14, 220)
        self.panel(surface, bar.inflate(18, 50).move(0, 8))
        pygame.draw.rect(surface, (60, 70, 86), bar)
        fill = bar.copy()
        fill.height = int(bar.height * state.throttle)
        fill.bottom = bar.bottom
        pygame.draw.rect(surface, (255, 160, 60), fill)
        lbl = self.small.render("THR", True, (150, 170, 200))
        surface.blit(lbl, (bar.centerx - lbl.get_width() / 2, bar.bottom + 6))
        # footer
        note = "KEYFRAMED VISUAL PREVIEW  -  not a physics simulation"
        if paused:
            note = "PAUSED  -  " + note
        n = self.small.render(note, True, (200, 210, 225))
        help_ = self.small.render("SPACE pause  <-/-> scrub  1-7 phase  M manual  +/- zoom  S screenshot  ESC quit",
                                  True, (160, 172, 190))
        self.panel(surface, pygame.Rect(16, H - 52, max(n.get_width(), help_.get_width()) + 24, 40))
        surface.blit(n, (28, H - 48))
        surface.blit(help_, (28, H - 30))


class Preview:
    def __init__(self, size=(1280, 800)):
        self.size = size
        self.renderer = RocketRenderer(GEOM)
        self.scene = SceneRenderer(*size)
        self.camera = Camera(*size, ppm=7.0)
        self.t = 0.0
        self.zoom = 1.0
        self.prev_pos = None
        self.velocity = (0.0, 0.0)
        self.cam_pos = None

    def step(self, t, dt, state=None):
        if state is None:
            state, phase, ppm = sample(t)
        else:
            phase, ppm = "MANUAL", 8.0
        if self.prev_pos is not None and dt > 0:
            self.velocity = ((state.x - self.prev_pos[0]) / dt, (state.y - self.prev_pos[1]) / dt)
        self.prev_pos = (state.x, state.y)
        self.camera.ppm = ppm * self.zoom
        target = np.array([state.x, state.y - 4])   # bias down so the plume is in frame
        if self.cam_pos is None or dt <= 0:
            self.cam_pos = target
        else:
            k = 1 - math.exp(-dt * 10)
            self.cam_pos = self.cam_pos + (target - self.cam_pos) * k
        # keep the rocket on screen even during very fast phases
        max_off = np.array(self.size) * 0.08 / self.camera.ppm
        self.cam_pos = np.clip(self.cam_pos, target - max_off, target + max_off)
        # never show much below the ground
        self.cam_pos[1] = max(self.cam_pos[1], self.size[1] * 0.36 / self.camera.ppm)
        self.camera.center = self.cam_pos
        self.renderer.update(state, dt)
        return state, phase

    def render(self, surface, state, phase):
        cam_alt = self.camera.center[1]
        self.scene.draw_sky(surface, self.camera, cam_alt)
        self.scene.draw_clouds(surface, self.camera, layer_front=False)
        self.scene.draw_ground(surface, self.camera)
        self.renderer.draw(surface, self.camera, state)
        self.scene.draw_clouds(surface, self.camera, layer_front=True)


def run_interactive(size):
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Reusable booster - visual preview")
    clock = pygame.time.Clock()
    pv = Preview(size)
    hud = Hud()
    paused, manual = False, False
    manual_state = RocketVisualState(x=0, y=600, engines=1, fin_deploy=1)
    legs_target, fins_target = 0.0, 1.0
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 1 / 20)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    pv.t = 0.0
                    pv.renderer.smoke.clear()
                elif ev.key == pygame.K_m:
                    manual = not manual
                    pv.renderer.smoke.clear()
                    pv.cam_pos = None
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    pv.zoom = min(4.0, pv.zoom * 1.2)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    pv.zoom = max(0.25, pv.zoom / 1.2)
                elif ev.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(pv.t * 10):04d}.png")
                elif manual and ev.key == pygame.K_l:
                    legs_target = 1.0 - legs_target
                elif manual and ev.key == pygame.K_f:
                    fins_target = 1.0 - fins_target
                elif manual and ev.key == pygame.K_g:
                    manual_state.engines = {1: 3, 3: 9, 9: 1}[manual_state.engines]
                elif not manual and pygame.K_1 <= ev.key <= pygame.K_7:
                    pv.t = PHASE_SNAPSHOT_T[ev.key - pygame.K_1] - 1.0
                    pv.renderer.smoke.clear()
                    pv.cam_pos = None
                elif not manual and ev.key == pygame.K_RIGHT:
                    pv.t = min(DURATION, pv.t + 2)
                elif not manual and ev.key == pygame.K_LEFT:
                    pv.t = max(0, pv.t - 2)
                    pv.renderer.smoke.clear()
        if manual:
            keys = pygame.key.get_pressed()
            ms = manual_state
            ms.throttle = min(1, max(0, ms.throttle + (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * dt * 0.8))
            lim = math.radians(GEOM.max_gimbal_deg)
            g_in = keys[pygame.K_a] - keys[pygame.K_d]
            ms.gimbal = max(-lim, min(lim, ms.gimbal + g_in * dt * 0.4 if g_in else ms.gimbal * (1 - dt * 4)))
            ms.angle += (keys[pygame.K_q] - keys[pygame.K_e]) * dt * 0.6
            ms.leg_deploy += max(-dt / 2.5, min(dt / 2.5, legs_target - ms.leg_deploy))
            ms.fin_deploy += max(-dt, min(dt, fins_target - ms.fin_deploy))
            ms.fin_twist = math.radians(15) * math.sin(pv.renderer.time * 1.3)
            state, phase = pv.step(pv.t, dt, replace(ms))
        else:
            if not paused:
                pv.t += dt
                if pv.t > DURATION:
                    pv.t = 0.0
                    pv.renderer.smoke.clear()
            state, phase = pv.step(pv.t, dt if not paused else 0.0)
        pv.render(screen, state, phase)
        hud.draw(screen, state, phase, pv.t, pv.velocity, manual, paused)
        pygame.display.flip()
    pygame.quit()


def contact_sheet(path, panel=(340, 620)):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
    font = pygame.font.SysFont("helveticaneue,helvetica,arial", 18, bold=True)
    small = pygame.font.SysFont("helveticaneue,helvetica,arial", 13)
    pw, ph = panel
    sheet = pygame.Surface((pw * len(PHASES) + 8 * (len(PHASES) + 1), ph + 70))
    sheet.fill((10, 13, 19))
    pv = Preview(panel)
    dt = 1 / 30
    t = 0.0
    snaps = list(zip(PHASE_SNAPSHOT_T, PHASES))
    frame = pygame.Surface(panel)
    idx = 0
    while idx < len(snaps):
        state, phase = pv.step(t, dt)
        if t >= snaps[idx][0]:
            pv.render(frame, state, phase)
            x = 8 + idx * (pw + 8)
            sheet.blit(frame, (x, 8))
            sheet.blit(font.render(f"{idx + 1}. {snaps[idx][1]}", True, (235, 240, 248)), (x + 2, ph + 16))
            alt = state.y - LANDED_Y
            info = (f"alt {alt / 1000:.1f} km" if alt > 1000 else f"alt {alt:.0f} m") + \
                   f"  thr {state.throttle * 100:.0f}%  eng {state.engines if state.throttle > 0.01 else 0}" + \
                   f"  legs {state.leg_deploy * 100:.0f}%  fins {state.fin_deploy * 100:.0f}%"
            sheet.blit(small.render(info, True, (150, 165, 190)), (x + 2, ph + 42))
            idx += 1
        t += dt
    pygame.image.save(sheet, path)
    print("wrote", path)


def export_assets(out_dir, ppm=24.0):
    """Write every component as a transparent PNG plus a JSON spec of pivots and dimensions."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
    os.makedirs(out_dir, exist_ok=True)
    r = RocketRenderer(GEOM)
    art = r.art(ppm)
    q = art.ppm
    parts = {
        "body": art.body,
        "nozzle": art.parts.build_nozzle(),
        "nozzle_cluster": art.parts.nozzle_cluster(0.0),
        "landing_leg": art.leg,
        "foot_pad": art.pad,
        "strut_sleeve": art.sleeve,
        "strut_rod": art.rod,
        "grid_fin_edge_on": art.parts.profile_fin(0),
        "grid_fin_twisted_20deg": art.parts.profile_fin(20),
        "grid_fin_front_stowed": art.parts.front_fin(0.0),
    }
    spec = {"pixels_per_metre": q, "geometry": GEOM.summary(), "components": {}}
    for name, sp in parts.items():
        fn = f"{name}.png"
        pygame.image.save(sp.surface, os.path.join(out_dir, fn))
        spec["components"][name] = {"file": fn, "size_px": list(sp.surface.get_size()),
                                    "pivot_px": [round(sp.pivot[0], 2), round(sp.pivot[1], 2)]}
    spec["components"]["body"]["pivot_note"] = "pivot = body-frame origin (nozzle exit centre)"
    # fully assembled configurations, pivot at the centre of mass
    configs = {
        "assembled_ascent": RocketVisualState(),
        "assembled_descent": RocketVisualState(fin_deploy=1, fin_twist=math.radians(15)),
        "assembled_landing": RocketVisualState(fin_deploy=1, leg_deploy=1),
    }
    for name, st in configs.items():
        canvas, pivot, _ = r.compose(st, q)
        crop = canvas.get_bounding_rect()
        img = canvas.subsurface(crop).copy()
        pygame.image.save(img, os.path.join(out_dir, f"{name}.png"))
        spec["components"][name] = {"file": f"{name}.png", "size_px": list(img.get_size()),
                                    "pivot_px": [round(pivot[0] - crop.x, 2), round(pivot[1] - crop.y, 2)],
                                    "pivot_note": "centre of mass"}
    # plume textures as RGBA (alpha = brightness) for use outside additive blending
    pf = PlumeFactory(GEOM, q)
    for name, eng, p in [("plume_sea_level_9eng", 9, 1.0), ("plume_sea_level_1eng", 1, 1.0),
                         ("plume_vacuum_9eng", 9, 0.02)]:
        tex, piv, tppm = pf.texture(eng, p, 0)
        rgb = pygame.surfarray.array3d(tex).astype(float)
        a = rgb.max(axis=2)
        out = pygame.Surface(tex.get_size(), pygame.SRCALPHA)
        px = pygame.surfarray.pixels3d(out)
        px[...] = np.clip(rgb / np.maximum(a[..., None], 1) * 255, 0, 255).astype(np.uint8)
        del px
        pa = pygame.surfarray.pixels_alpha(out)
        pa[...] = a.astype(np.uint8)
        del pa
        pygame.image.save(out, os.path.join(out_dir, f"{name}.png"))
        spec["components"][name] = {"file": f"{name}.png", "size_px": list(out.get_size()),
                                    "pivot_px": [round(piv[0], 2), round(piv[1], 2)],
                                    "pixels_per_metre": tppm, "pivot_note": "nozzle exit centre, flow toward +y (down)"}
    with open(os.path.join(out_dir, "rocket_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    print(f"exported {len(spec['components'])} components to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact-sheet", metavar="PNG")
    ap.add_argument("--export", metavar="DIR")
    ap.add_argument("--size", default="1280x800")
    args = ap.parse_args()
    if args.contact_sheet:
        contact_sheet(args.contact_sheet)
    if args.export:
        export_assets(args.export)
    if not args.contact_sheet and not args.export:
        run_interactive(tuple(int(v) for v in args.size.split("x")))


if __name__ == "__main__":
    main()
