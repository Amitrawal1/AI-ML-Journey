"""Train a PPO agent to land the booster.

    python tools/train.py                         # default: 3M steps, ~20-40 min on a laptop
    python tools/train.py --steps 500000 --name quick
    python tools/train.py --resume runs/ppo_landing/latest.zip --steps 2000000
    python tools/train.py --eval runs/ppo_landing/best_model.zip
    python tools/train.py --autopilot --name ap1   # agent picks a lean angle, a PD loop steers
    python tools/train.py --resume runs/ap1/best_model.zip --autopilot --pad-start 300 --pad-width 90
                                                   # shrink the pad from 300 m to 90 m while training

Curriculum: episodes start easy (low, slow, close to the pad) and the
difficulty ramps to 1.0 over the first 60% of training. Every --eval-every
steps the agent is tested on full-difficulty starts; the best one is saved as
runs/<name>/best_model.zip.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rocket_sim.envs import RocketLandingEnv


def evaluate(model, episodes=50, difficulty=1.0, seed=10_000, autopilot=False, pad_width=300.0):
    """Deterministic test flights. Returns outcome counts and mean return."""
    env = RocketLandingEnv(difficulty=difficulty, autopilot=autopilot, pad_width=pad_width)
    outcomes, returns = {}, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += r
            if term or trunc:
                break
        outcomes[info["outcome"]] = outcomes.get(info["outcome"], 0) + 1
        returns.append(total)
    return outcomes, float(np.mean(returns))


class CurriculumAndEval(BaseCallback):
    def __init__(self, total_steps, run_dir, start_difficulty, ramp_fraction, eval_every, eval_episodes,
                 autopilot=False, pad_start=300.0, pad_width=300.0):
        super().__init__()
        self.autopilot = autopilot
        self.pad0, self.pad1 = pad_start, pad_width
        self.total = total_steps
        self.run_dir = run_dir
        self.d0 = start_difficulty
        self.ramp = ramp_fraction
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.next_eval = eval_every
        self.best = -1.0
        self.t0 = time.time()
        self.log = []
        self.start_steps = 0

    def _on_training_start(self):
        self.start_steps = self.num_timesteps

    def difficulty(self):
        frac = (self.num_timesteps - self.start_steps) / max(1, self.total * self.ramp)
        return float(min(1.0, self.d0 + (1.0 - self.d0) * frac))

    def pad_width(self):
        frac = (self.num_timesteps - self.start_steps) / max(1, self.total * self.ramp)
        return float(self.pad0 + (self.pad1 - self.pad0) * min(1.0, frac))

    def _on_rollout_start(self):
        self.training_env.env_method("set_difficulty", self.difficulty())
        self.training_env.env_method("set_pad_width", self.pad_width())

    def _on_step(self):
        if self.num_timesteps - self.start_steps >= self.next_eval:
            self.next_eval += self.eval_every
            # always tested on the final (smallest) pad, so scores stay comparable
            outcomes, mean_ret = evaluate(self.model, self.eval_episodes, autopilot=self.autopilot,
                                          pad_width=self.pad1)
            n = sum(outcomes.values())
            success = outcomes.get("landed", 0) / n
            elapsed = time.time() - self.t0
            row = {"steps": int(self.num_timesteps), "difficulty": round(self.difficulty(), 3),
                   "success_on_pad": round(success, 3), "mean_return": round(mean_ret, 1),
                   "outcomes": outcomes, "minutes": round(elapsed / 60, 1)}
            self.log.append(row)
            with open(os.path.join(self.run_dir, "progress.json"), "w") as f:
                json.dump(self.log, f, indent=1)
            mark = ""
            if success > self.best or (success == self.best and mean_ret > getattr(self, "best_ret", -1e9)):
                self.best, self.best_ret = success, mean_ret
                self.model.save(os.path.join(self.run_dir, "best_model"))
                mark = "  <- new best, saved"
            self.model.save(os.path.join(self.run_dir, "latest"))
            print(f"[{elapsed / 60:5.1f} min] steps {self.num_timesteps:>9,}  difficulty {row['difficulty']:.2f}  "
                  f"pad {self.pad_width():3.0f} m  "
                  f"landed on pad {success * 100:5.1f}%  return {mean_ret:7.1f}  {outcomes}{mark}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=float, default=3e6, help="environment steps to train for")
    ap.add_argument("--envs", type=int, default=8, help="parallel simulations")
    ap.add_argument("--name", default="ppo_landing", help="run folder under runs/")
    ap.add_argument("--resume", help="continue training from a saved .zip")
    ap.add_argument("--start-difficulty", type=float, default=0.1)
    ap.add_argument("--ramp", type=float, default=0.6, help="fraction of training spent ramping difficulty to 1")
    ap.add_argument("--eval-every", type=float, default=100_000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (use ~1e-4 when fine-tuning with --resume)")
    ap.add_argument("--autopilot", action="store_true",
                    help="action 1 is a target lean angle held by a PD controller (models are not "
                         "interchangeable: evaluate/watch an autopilot model with --autopilot too)")
    ap.add_argument("--pad-width", type=float, default=300.0, help="final pad width in m (evals use this)")
    ap.add_argument("--pad-start", type=float, help="pad width at the start, shrinking to --pad-width "
                                                    "over the --ramp fraction (default: no shrinking)")
    ap.add_argument("--eval", metavar="MODEL", help="only evaluate a saved model and exit")
    args = ap.parse_args()

    if args.eval:
        model = PPO.load(args.eval, device="cpu")
        outcomes, mean_ret = evaluate(model, 200, autopilot=args.autopilot, pad_width=args.pad_width)
        n = sum(outcomes.values())
        print(f"200 full-difficulty landings: on pad {outcomes.get('landed', 0) / n * 100:.1f}%  "
              f"mean return {mean_ret:.1f}  {outcomes}")
        return

    torch.set_num_threads(1)
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    total = int(args.steps)
    pad_start = args.pad_start or args.pad_width

    venv = make_vec_env(RocketLandingEnv, n_envs=args.envs, seed=args.seed,
                        env_kwargs={"difficulty": args.start_difficulty, "autopilot": args.autopilot,
                                    "pad_width": pad_start}, vec_env_cls=SubprocVecEnv)
    # rewards range over hundreds; normalising them keeps PPO's value loss well scaled
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.999)

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", custom_objects={"learning_rate": args.lr})
        print(f"resumed from {args.resume}  (learning rate {args.lr:g})")
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=2048, batch_size=2048, n_epochs=10,
            gamma=0.999, gae_lambda=0.95, learning_rate=args.lr, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            seed=args.seed, device="cpu", verbose=0,
        )

    cb = CurriculumAndEval(total, run_dir, args.start_difficulty, args.ramp,
                           int(args.eval_every), args.eval_episodes, args.autopilot, pad_start, args.pad_width)
    print(f"training for {total:,} steps with {args.envs} parallel envs -> {run_dir}/", flush=True)
    try:
        model.learn(total_timesteps=total, callback=cb, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("\nstopped early")
    model.save(os.path.join(run_dir, "latest"))
    print(f"saved {run_dir}/latest.zip  (best: {run_dir}/best_model.zip, "
          f"{cb.best * 100:.1f}% landed on pad at full difficulty)")
    venv.close()


if __name__ == "__main__":
    main()
