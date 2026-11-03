"""Neural Catcher: a simulated organoid learns to catch falling orbs.

Each 20 Hz tick, the orb's position relative to the paddle is stimulated into one
of six sensor channels. Spike counts across the array drive the paddle left or
right through a linear readout trained by reward (REINFORCE). Catches and misses
are also sent to the LIF network as reward stims, so its coupling weights adapt
too (simulator only, see organoid_simulator.reward_lif).

Run from the repo root:

    uv run python experiments/catcher.py                 # pygame window + SDK viewer
    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/catcher.py --headless --seconds 300
"""
from __future__ import annotations

import argparse
import webbrowser
from collections import deque

import numpy as np

import cl

TICKS_PER_SECOND = 20
STIM = cl.StimDesign(200, -1.5, 200, 1.5)
REWARD_STIM = cl.StimDesign(200, -0.5, 200, 0.5)   # amplitude is ignored by the reward bridge
SENSOR_CHANNELS = (25, 26, 27, 28, 29, 30)         # dx bins, far-left .. far-right
REWARD_CHANNEL, PUNISH_CHANNEL = 1, 2

PADDLE_HALF = 0.1
CATCH_RADIUS = 0.12
PADDLE_SPEED = 0.035
FALL_TICKS = 40                                    # 2 s per orb


class Game:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.paddle_x = 0.5
        self.catches = 0
        self.misses = 0
        self.history: deque[int] = deque(maxlen=20)  # rolling catch window
        self.curve: list[float] = []                 # catch rate after each orb
        self.flash = 0                               # ticks of catch/miss feedback left
        self.last_result = 0
        self.new_orb()

    def new_orb(self) -> None:
        self.orb_x = float(self.rng.uniform(0.05, 0.95))
        self.orb_t = 0

    @property
    def orb_y(self) -> float:
        return self.orb_t / FALL_TICKS

    def sensor_channel(self) -> int:
        dx = np.clip(self.orb_x - self.paddle_x, -0.999, 0.999)
        return SENSOR_CHANNELS[int((dx + 1) / 2 * len(SENSOR_CHANNELS))]

    def step(self, action: int) -> int:
        """Move the paddle, advance the orb. Returns +1 catch, -1 miss, 0 otherwise."""
        self.paddle_x = float(np.clip(self.paddle_x + action * PADDLE_SPEED, PADDLE_HALF, 1 - PADDLE_HALF))
        self.orb_t += 1
        self.flash = max(0, self.flash - 1)
        if self.orb_t < FALL_TICKS:
            return 0
        result = 1 if abs(self.orb_x - self.paddle_x) < CATCH_RADIUS else -1
        if result > 0:
            self.catches += 1
        else:
            self.misses += 1
        self.history.append(result > 0)
        self.curve.append(sum(self.history) / len(self.history))
        self.flash, self.last_result = 6, result
        self.new_orb()
        return result


class Readout:
    """Linear policy over per-channel spike counts, trained with REINFORCE.

    Reward arrives every tick (did that move bring the paddle closer to the orb?),
    which assigns credit far faster than one reward per orb."""

    def __init__(self, rng: np.random.Generator, lr: float):
        self.rng = rng
        self.lr = lr
        self.learning = True
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(64)
        self.trace = np.zeros(64)

    def act(self, counts: np.ndarray) -> int:
        p_right = 1 / (1 + np.exp(-np.clip(self.w @ counts, -20, 20)))
        action = 1 if self.rng.random() < p_right else -1
        self.trace += (action - (2 * p_right - 1)) * counts
        return action

    def reward(self, r: float) -> None:
        if self.learning:
            self.w += self.lr * r * self.trace
        self.trace[:] = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this long (default: run until closed)")
    parser.add_argument("--seed", type=int, default=42, help="Simulator and game seed (default 42)")
    parser.add_argument("--coupling", type=float, default=4.0, help="LIF coupling_gain_mV (default 4.0)")
    parser.add_argument("--no-plasticity", action="store_true", help="Freeze the LIF coupling weights")
    parser.add_argument("--lr", type=float, default=0.002, help="Readout learning rate; lower = slower, more visible learning (default 0.002)")
    parser.add_argument("--headless", action="store_true", help="No window; print catch rate per block")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the SDK viewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.reward_lif:make_reward_lif_source",
            config={"random_seed": args.seed, "coupling_gain_mV": args.coupling,
                    "plasticity": not args.no_plasticity},
        )

    rng = np.random.default_rng(args.seed)
    game = Game(rng)
    readout = Readout(rng, args.lr)
    view = None if args.headless else View()
    counts = np.zeros(64)

    with cl.open() as neurons:
        server = getattr(neurons, "_websocket_server", None)
        if server is not None and view is not None and not args.no_browser:
            webbrowser.open(server.web_url)

        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_seconds=args.seconds):
            counts[:] = 0
            for spike in tick.analysis.spikes:
                counts[spike.channel] += 1
            counts[[REWARD_CHANNEL, PUNISH_CHANNEL]] = 0

            before = abs(game.orb_x - game.paddle_x)
            result = game.step(readout.act(counts))
            if not result:
                readout.reward(1.0 if abs(game.orb_x - game.paddle_x) < before else -1.0)
            else:
                readout.trace[:] = 0
                neurons.stim(cl.ChannelSet(REWARD_CHANNEL if result > 0 else PUNISH_CHANNEL), REWARD_STIM)
                if args.headless and (game.catches + game.misses) % 10 == 0:
                    n = game.catches + game.misses
                    print(f"orbs {n - 9:>4}-{n:<4} catch rate (last 20) {game.curve[-1]:.0%}")
            neurons.stim(cl.ChannelSet(game.sensor_channel()), STIM)

            if view is not None and not view.draw(game, readout, counts):
                break

    total = game.catches + game.misses
    print(f"\n{game.catches}/{total} caught" + (f", last 20: {game.curve[-1]:.0%}" if game.curve else ""))


class View:
    W, H, GAME_W = 960, 600, 480

    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Neural Catcher")
        self.font = pygame.font.SysFont("menlo", 16)
        self.big = pygame.font.SysFont("menlo", 28, bold=True)
        self.heat = np.zeros(64)

    def draw(self, game: Game, readout: Readout, counts: np.ndarray) -> bool:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                readout.learning = not readout.learning
            if event.type == pg.KEYDOWN and event.key == pg.K_r:
                readout.reset()
                game.history.clear()

        s = self.screen
        s.fill((12, 14, 22))
        gw, h = self.GAME_W, self.H

        # Game area
        if game.flash:
            tint = (20, 70, 40) if game.last_result > 0 else (80, 20, 25)
            pg.draw.rect(s, tint, (0, 0, gw, h))
        pg.draw.circle(s, (255, 200, 60), (int(game.orb_x * gw), int(40 + game.orb_y * (h - 100))), 12)
        px = int(game.paddle_x * gw)
        pg.draw.rect(s, (90, 200, 255), (px - int(PADDLE_HALF * gw), h - 50, int(2 * PADDLE_HALF * gw), 14), border_radius=6)
        pg.draw.line(s, (40, 44, 60), (gw, 0), (gw, h), 2)
        s.blit(self.big.render(f"{game.catches} caught  {game.misses} missed", True, (230, 230, 240)), (16, 12))

        x0 = gw + 24
        # Live spikes, 8x8
        self.heat = 0.8 * self.heat + counts
        self._grid(x0, 40, "live spikes", lambda ch: self._heat_color(self.heat[ch]))
        # Learned readout: orange = pushes right, blue = pushes left
        wmax = max(np.abs(readout.w).max(), 1e-6)
        self._grid(x0 + 220, 40, "readout map", lambda ch: self._weight_color(readout.w[ch] / wmax))

        # Catch-rate curve
        cx, cy, cw, ch_ = x0, 290, 420, 200
        pg.draw.rect(s, (24, 27, 40), (cx, cy, cw, ch_))
        pg.draw.line(s, (60, 64, 85), (cx, cy + ch_ * 0.75), (cx + cw, cy + ch_ * 0.75))
        curve = game.curve[-200:]
        if len(curve) > 1:
            pts = [(cx + i * cw / (len(curve) - 1), cy + ch_ * (1 - v)) for i, v in enumerate(curve)]
            pg.draw.lines(s, (120, 230, 140), False, pts, 2)
        rate = f"{game.curve[-1]:.0%}" if game.curve else "--"
        s.blit(self.font.render(f"catch rate (last 20): {rate}   chance ~25%", True, (200, 200, 210)), (cx, cy - 22))

        status = "LEARNING" if readout.learning else "FROZEN"
        s.blit(self.font.render(f"[space] {status}   [r] reset readout   [esc] quit", True, (150, 150, 170)), (x0, h - 40))
        pg.display.flip()
        return True

    def _grid(self, x0, y0, title, color_of):
        pg = self.pg
        self.screen.blit(self.font.render(title, True, (200, 200, 210)), (x0, y0 - 22))
        for ch in range(64):
            r, c = divmod(ch, 8)
            color = (35, 35, 45) if ch in (0, 4, 7, 56, 63) else color_of(ch)
            rect = (x0 + c * 24, y0 + r * 24, 22, 22)
            pg.draw.rect(self.screen, color, rect, border_radius=3)
            if ch in SENSOR_CHANNELS:
                pg.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=3)

    @staticmethod
    def _heat_color(v):
        t = min(v / 4.0, 1.0)
        return (int(30 + 225 * t), int(30 + 180 * t ** 2), int(50 + 40 * t))

    @staticmethod
    def _weight_color(v):
        if v >= 0:
            return (int(30 + 225 * v), int(30 + 120 * v), 40)
        return (40, int(30 + 120 * -v), int(30 + 225 * -v))


if __name__ == "__main__":
    main()
