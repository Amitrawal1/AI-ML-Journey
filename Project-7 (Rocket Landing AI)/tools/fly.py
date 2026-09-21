"""Fly and land the booster yourself, on the real physics model.

    python tools/fly.py                  # landing burn scenario (default)
    python tools/fly.py --scenario hover
    python tools/fly.py --pad 500        # landing pad width in metres (default 300)

Controls
    UP / DOWN     throttle up / down (engines run 35..100%, below that they shut off)
    Z / X         full throttle / engine cutoff
    LEFT / RIGHT  steer: engine gimbal + cold-gas thrusters + grid fins together
    L             landing legs          F  grid fins
    G             engine count 1 / 3 / 9
    T             time warp 1x / 2x / 4x    P  pause
    1-4           switch scenario          R  restart
    + / -         zoom                     ESC quit
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame

from rocket_sim.physics import Controls, PhysicsParams, RocketPhysics, RocketState
from rocket_sim.render import RocketGeometry
from rocket_sim.render.flight_view import FlightView, Hud
from rocket_sim.render.scene import SceneRenderer

GEOM = RocketGeometry()

SCENARIOS = {
    "hover": dict(title="Low hover", desc="300 m above the pad, slow descent. Learn the controls.",
                  state=dict(x=-60, y=300, vx=4, vy=-25, angle=0.0), fuel=6000, engines=1, fins=True),
    "landing": dict(title="Landing burn", desc="2.6 km up, falling at 170 m/s, drifting toward the pad.",
                    state=dict(x=-420, y=2600, vx=20, vy=-170, angle=0.04), fuel=6500, engines=1, fins=True),
    "descent": dict(title="High descent", desc="15 km up, falling at 450 m/s. Use 3 engines to brake.",
                    state=dict(x=-2500, y=15000, vx=90, vy=-450, angle=-0.05), fuel=14000, engines=3, fins=True),
    "launch": dict(title="Full mission", desc="Launch, ascend, come back and land. Long and hard.",
                   state=dict(x=0, y=SceneRenderer.PAD_HEIGHT + 1.6 + GEOM.com_y, vx=0, vy=0, angle=0.0),
                   fuel=380_000, engines=9, fins=False, held=True),
}
ORDER = ["hover", "landing", "descent", "launch"]

class Game:
    """Keyboard pilot + physics; all drawing goes through FlightView."""

    HELP = "UP/DOWN throttle  Z full  X cut  LEFT/RIGHT steer  L legs  F fins  G engines  T warp  P pause  R restart  1-4 scenario"

    def __init__(self, size, scenario, pad_width=300.0):
        self.size = size
        self.phys = RocketPhysics(GEOM, PhysicsParams(pad_radius=pad_width / 2))
        self.view = FlightView(self.phys, size, pad_width / 2)
        self.warp = 1
        self.paused = False
        self.load(scenario)

    def load(self, name):
        self.name = name
        sc = SCENARIOS[name]
        st = RocketState(**sc["state"], fuel=sc["fuel"], engines=sc["engines"],
                         fin_deploy=1.0 if sc["fins"] else 0.0, held=sc.get("held", False))
        self.phys.reset(st, fuel_capacity=sc["fuel"])
        self.ctrl = Controls(engines=sc["engines"], fins_deployed=sc["fins"])
        self.view.reset()

    # -------------------------------------------------------------- input
    def handle_key(self, key):
        c = self.ctrl
        if key == pygame.K_r:
            self.load(self.name)
        elif pygame.K_1 <= key <= pygame.K_4:
            self.load(ORDER[key - pygame.K_1])
        elif key == pygame.K_l:
            c.legs = not c.legs
        elif key == pygame.K_f:
            c.fins_deployed = not c.fins_deployed
        elif key == pygame.K_g:
            c.engines = {1: 3, 3: 9, 9: 1}[c.engines]
        elif key == pygame.K_z:
            c.throttle = 1.0
        elif key == pygame.K_x:
            c.throttle = 0.0
        elif key == pygame.K_t:
            self.warp = {1: 2, 2: 4, 4: 1}[self.warp]
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.view.zoom = min(4.0, self.view.zoom * 1.25)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.view.zoom = max(0.25, self.view.zoom / 1.25)

    def read_held_keys(self, dt):
        keys = pygame.key.get_pressed()
        c, p = self.ctrl, self.phys.p
        d = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        if d:
            base = c.throttle if c.throttle > 0 else (p.min_throttle if d > 0 else 0)
            c.throttle = base + d * 0.7 * dt
            if c.throttle < p.min_throttle - 0.02:
                c.throttle = 0.0
            c.throttle = min(1.0, c.throttle)
        steer = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]      # +1 = rotate nose left (CCW)
        lim = math.radians(GEOM.max_gimbal_deg)
        c.gimbal = -steer * lim
        c.rcs = float(steer)
        c.fins = -float(steer)

    # ------------------------------------------------------------- update
    def update(self, dt):
        self.read_held_keys(dt)
        was_done = self.phys.status.done
        if not self.paused:
            for _ in range(self.warp):
                self.phys.step(self.ctrl, dt)
        just_finished = self.phys.status.done and not was_done
        self.view.update(dt, 0.0 if self.paused else dt * self.warp, just_finished)

    def draw(self, screen, hud):
        self.view.draw_world(screen)
        hud.draw(screen, self.view, self.ctrl, SCENARIOS[self.name]["title"], self.HELP,
                 self.warp, self.paused, "R  retry     1-4  other scenario")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=ORDER, default="landing")
    ap.add_argument("--size", default="1280x800")
    ap.add_argument("--pad", type=float, default=300.0, help="landing pad width in metres (default 300)")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Reusable booster - manual flight")
    clock = pygame.time.Clock()
    game = Game(size, args.scenario, args.pad)
    hud = Hud()
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 1 / 20)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                game.handle_key(ev.key)
        game.update(dt)
        game.draw(screen, hud)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
