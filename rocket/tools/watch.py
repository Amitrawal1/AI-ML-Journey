"""Watch a trained agent fly.

    python tools/watch.py                                   # uses runs/ppo_landing/best_model.zip
    python tools/watch.py --model runs/quick/latest.zip --difficulty 0.5
    python tools/watch.py --model runs/ap1/best_model.zip --autopilot
    python tools/watch.py --seed 500                        # a different set of start conditions
"""

import argparse
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
from stable_baselines3 import PPO

from rocket_sim.envs import RocketLandingEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="runs/ppo_landing/best_model.zip")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--autopilot", action="store_true", help="the model was trained with --autopilot")
    ap.add_argument("--pad-width", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0, help="first flight's seed; change it for a new set of flights")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"no model at {args.model} - train one first with: .venv/bin/python tools/train.py")
    model = PPO.load(args.model, device="cpu")
    env = RocketLandingEnv(render_mode="human", difficulty=args.difficulty, autopilot=args.autopilot,
                           pad_width=args.pad_width)
    tally = {}
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    env.close()
                    return
            if term or trunc:
                break
        tally[info["outcome"]] = tally.get(info["outcome"], 0) + 1
        print(f"flight {ep + 1}: {info['outcome']}  {info.get('reason', '')}   totals {tally}")
        for _ in range(45):                     # hold the result on screen ~1.5 s
            env.render()
            pygame.event.pump()
    env.close()


if __name__ == "__main__":
    main()
