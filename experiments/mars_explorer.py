"""Mars Explorer: a simulated organoid flies a drone over Mars hunting for beacons, in 3D.

Mars is 4-24 light-minutes from Earth, so a drone there cannot be flown by remote
control; it has to fly itself. The drone patrols a search pattern over the hills.
When a beacon comes within detection range it homes in on it; reaching it counts
as a find and a new beacon appears elsewhere.

Each 20 Hz tick, each axis's predicted error (goal minus where the drone is
heading) is binned into one of six sensor channels: row 1 east/west, row 4
altitude, row 6 north/south. Spike counts across the array drive three thruster
readouts, each trained every tick by whether its error shrank (REINFORCE). Finds
and crashes are also sent to the LIF network as reward stims, so its coupling
weights adapt too (simulator only, see organoid_simulator.reward_lif).

The 3D view is a three.js page served from this script and fed over a websocket.

Run from the repo root:

    uv run python experiments/mars_explorer.py           # opens the 3D view in your browser
    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/mars_explorer.py --headless --seconds 300
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import queue
import threading
import webbrowser
from pathlib import Path

import numpy as np

import cl

TICKS_PER_SECOND = 20
DT = 1 / TICKS_PER_SECOND
STIM = cl.StimDesign(200, -2.5, 200, 2.5)
REWARD_STIM = cl.StimDesign(200, -0.5, 200, 0.5)   # amplitude is ignored by the reward bridge
REWARD_CHANNEL, PUNISH_CHANNEL = 1, 2
SENSORS = ((9, 10, 11, 12, 13, 14),                # row 1: east/west error bins
           (33, 34, 35, 36, 37, 38),               # row 4: altitude
           (49, 50, 51, 52, 53, 54))               # row 6: north/south
ERROR_EDGES = (-6.0, -2.0, 0.0, 2.0, 6.0)          # metres; 6 bins

WORLD = 100.0          # world spans -WORLD..WORLD on x and z
GRID = 101             # heightmap resolution sent to the browser
THRUST = 2.5           # m/s^2 per axis on top of the hover baseline
DRAG = 0.6             # 1/s
LEAD = 1.0             # s of velocity folded into the sensed error
CRUISE = 12.0          # search altitude above the terrain, m
DETECT = 40.0          # beacon detection range (horizontal), m
FIND = 4.0             # reach this close to a beacon to find it, m
N_BEACONS = 4

HERE = Path(__file__).parent / "mars_explorer"


def terrain(x, z):
    """Rolling Martian hills plus two craters. Works on scalars and arrays."""
    h = (4 * np.sin(x / 23) * np.cos(z / 29) + 2.5 * np.sin(x / 11 + 1.3) * np.sin(z / 13 + 0.7)
         + 1.2 * np.cos((x + z) / 7))
    for cx, cz, r, depth in ((40, -30, 18, 7), (-45, 35, 14, 5)):
        d2 = ((x - cx) ** 2 + (z - cz) ** 2) / r ** 2
        h = h - depth * np.exp(-d2 * 2) + 0.6 * depth * np.exp(-((np.sqrt(d2) - 1) ** 2) * 12)
    return h


def search_pattern() -> list[np.ndarray]:
    """Expanding square spiral over the map, as (x, z) waypoints."""
    points, step, x, z = [], 30.0, 0.0, 0.0
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for leg in range(12):
        dx, dz = directions[leg % 4]
        length = step * (leg // 2 + 1)
        x, z = np.clip(x + dx * length, -85, 85), np.clip(z + dz * length, -85, 85)
        points.append(np.array([x, z]))
    return points


class World:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.waypoints = search_pattern()
        self.waypoint = 0
        self.beacons = [self.new_beacon() for _ in range(N_BEACONS)]
        self.found = 0
        self.crashes = 0
        self.approach_times: list[float] = []   # seconds from detection to find
        self.target = -1                       # index of the beacon being homed on
        self.detected_at = 0
        self.ticks = 0
        self.thrust = np.zeros(3)
        self.respawn(np.array([0.0, 0.0]))

    def new_beacon(self, avoid: np.ndarray | None = None) -> np.ndarray:
        while True:
            x, z = self.rng.uniform(-85, 85, 2)
            if avoid is None or np.hypot(x - avoid[0], z - avoid[2]) > 60:
                return np.array([x, terrain(x, z), z])

    def respawn(self, xz: np.ndarray) -> None:
        self.pos = np.array([xz[0], terrain(*xz) + CRUISE, xz[1]])
        self.vel = np.zeros(3)

    def goal(self) -> np.ndarray:
        x, _, z = self.pos
        if self.target < 0:
            near = [(np.hypot(b[0] - x, b[2] - z), i) for i, b in enumerate(self.beacons)]
            dist, i = min(near)
            if dist < DETECT:
                self.target, self.detected_at = i, self.ticks
        if self.target >= 0:
            return self.beacons[self.target] + np.array([0.0, 2.0, 0.0])
        wx, wz = self.waypoints[self.waypoint]
        return np.array([wx, terrain(x, z) + CRUISE, wz])   # follow the terrain while searching

    def errors(self) -> np.ndarray:
        return self.goal() - (self.pos + LEAD * self.vel)

    def step(self, thrust: np.ndarray) -> int:
        """Advance one tick. Returns +1 beacon found, -1 crash, 0 otherwise."""
        self.ticks += 1
        self.thrust = thrust
        self.vel += (THRUST * thrust - DRAG * self.vel) * DT
        self.pos += self.vel * DT
        self.pos[[0, 2]] = np.clip(self.pos[[0, 2]], -WORLD + 1, WORLD - 1)

        x, y, z = self.pos
        if y <= terrain(x, z):
            self.crashes += 1
            self.target = -1
            self.respawn(np.array([x, z]))
            return -1
        if self.target >= 0 and np.linalg.norm(self.goal() - self.pos) < FIND:
            self.found += 1
            self.approach_times.append((self.ticks - self.detected_at) / TICKS_PER_SECOND)
            self.beacons[self.target] = self.new_beacon(avoid=self.pos)
            self.target = -1
            return 1
        if self.target < 0 and np.hypot(*(self.waypoints[self.waypoint] - self.pos[[0, 2]])) < 8:
            self.waypoint = (self.waypoint + 1) % len(self.waypoints)
        return 0

    def state(self, counts: np.ndarray, sensors: tuple[int, ...], learning: bool) -> dict:
        r = lambda v: [round(float(c), 2) for c in v]
        return {"type": "state", "t": self.ticks / TICKS_PER_SECOND, "pos": r(self.pos), "vel": r(self.vel),
                "thrust": r(self.thrust), "goal": r(self.goal()), "target": self.target,
                "beacons": [r(b) for b in self.beacons], "found": self.found, "crashes": self.crashes,
                "times": self.approach_times[-40:], "spikes": counts.astype(int).tolist(),
                "sensors": list(sensors), "learning": learning}


class Readout:
    """Binary thruster policy over spike counts, trained every tick with REINFORCE."""

    def __init__(self, rng: np.random.Generator, lr: float):
        self.rng = rng
        self.lr = lr
        self.learning = True
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(64)
        self.b = 0.0
        self.grad = np.zeros(64)
        self.grad_b = 0.0

    def act(self, counts: np.ndarray) -> int:
        p_pos = 1 / (1 + np.exp(-np.clip(self.w @ counts + self.b, -20, 20)))
        action = 1 if self.rng.random() < p_pos else -1
        self.grad_b = action - (2 * p_pos - 1)
        self.grad = self.grad_b * counts
        return action

    def reward(self, r: float) -> None:
        if self.learning:
            self.w += self.lr * r * self.grad
            self.b += self.lr * r * self.grad_b


class Viewer:
    """Serves the three.js page over HTTP and streams world state over a websocket."""

    def __init__(self, http_port: int, ws_port: int):
        from websockets.asyncio.server import broadcast, serve
        self.broadcast = broadcast
        self.clients: set = set()
        self.commands: queue.Queue[str] = queue.Queue()
        self.init_message = json.dumps(self.terrain_message())
        self.loop = asyncio.new_event_loop()

        async def handler(ws):
            self.clients.add(ws)
            try:
                await ws.send(self.init_message)
                async for message in ws:
                    self.commands.put(json.loads(message).get("cmd", ""))
            finally:
                self.clients.discard(ws)

        async def run():
            async with serve(handler, "127.0.0.1", ws_port):
                await asyncio.Future()

        threading.Thread(target=self.loop.run_until_complete, args=(run(),), daemon=True).start()
        handler_cls = functools.partial(QuietHandler, directory=str(HERE))
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", http_port), handler_cls)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{http_port}/?ws={ws_port}"

    @staticmethod
    def terrain_message() -> dict:
        axis = np.linspace(-WORLD, WORLD, GRID)
        xs, zs = np.meshgrid(axis, axis)          # row = z, column = x
        heights = terrain(xs, zs)
        return {"type": "init", "world": WORLD, "grid": GRID, "heights": np.round(heights, 2).ravel().tolist(),
                "sensors": [list(s) for s in SENSORS], "detect": DETECT}

    def send(self, state: dict) -> None:
        if self.clients:
            self.loop.call_soon_threadsafe(self.broadcast, set(self.clients), json.dumps(state))


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def relax_accelerated_timeout() -> None:
    """In accelerated time the SDK waits only 0.1 s of wall time for each tick's
    data, which the LIF source can miss under load; wait up to 60 s instead.
    Patches a private SDK class, so it is skipped if that class moves."""
    try:
        from cl._sim._data_buffer import SharedDataBuffer
    except ImportError:
        return
    original = SharedDataBuffer.wait_for_timestamp

    def wait_for_timestamp(self, target_timestamp, timeout_seconds=None):
        return original(self, target_timestamp, timeout_seconds=60.0)

    SharedDataBuffer.wait_for_timestamp = wait_for_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this long (default: run until Ctrl-C)")
    parser.add_argument("--seed", type=int, default=42, help="Simulator and world seed (default 42)")
    parser.add_argument("--coupling", type=float, default=4.0, help="LIF coupling_gain_mV (default 4.0)")
    parser.add_argument("--lr", type=float, default=0.01, help="Readout learning rate (default 0.01)")
    parser.add_argument("--no-plasticity", action="store_true", help="Freeze the LIF coupling weights")
    parser.add_argument("--headless", action="store_true", help="No 3D view; print progress every 20 s")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port for the 3D view; websocket uses port+1")
    parser.add_argument("--no-browser", action="store_true", help="Print the 3D view URL instead of opening it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.reward_lif:make_reward_lif_source",
            config={"random_seed": args.seed, "coupling_gain_mV": args.coupling,
                    "plasticity": not args.no_plasticity,
                    "reward_channel": REWARD_CHANNEL, "punish_channel": PUNISH_CHANNEL},
        )
        if os.getenv("CL_SDK_ACCELERATED_TIME") == "1":
            relax_accelerated_timeout()

    rng = np.random.default_rng(args.seed)
    world = World(rng)
    readouts = [Readout(rng, args.lr) for _ in range(3)]   # east/west, altitude, north/south
    viewer = None
    if not args.headless:
        viewer = Viewer(args.port, args.port + 1)
        print(f"3D view: {viewer.url}")
        if not args.no_browser:
            webbrowser.open(viewer.url)
    counts = np.zeros(64)

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_seconds=args.seconds):
            counts[:] = 0
            for spike in tick.analysis.spikes:
                counts[spike.channel] += 1
            counts[[REWARD_CHANNEL, PUNISH_CHANNEL]] = 0

            while viewer is not None and not viewer.commands.empty():
                cmd = viewer.commands.get()
                if cmd == "toggle_learning":
                    for readout in readouts:
                        readout.learning = not readout.learning
                elif cmd == "reset":
                    for readout in readouts:
                        readout.reset()
                    world.approach_times.clear()
                    world.found = world.crashes = 0

            # Spikes this tick respond to last tick's sensor stims.
            before = np.abs(world.errors())
            thrust = np.array([readout.act(counts) for readout in readouts], dtype=float)
            event = world.step(thrust)
            if event == 0:
                after = np.abs(world.errors())
                for readout, b, a in zip(readouts, before, after):
                    readout.reward(1.0 if a < b else -1.0)
            else:
                neurons.stim(cl.ChannelSet(REWARD_CHANNEL if event > 0 else PUNISH_CHANNEL), REWARD_STIM)

            sensors = tuple(chs[int(np.searchsorted(ERROR_EDGES, e))] for chs, e in zip(SENSORS, world.errors()))
            neurons.stim(cl.ChannelSet(*sensors), STIM)

            if viewer is not None:
                viewer.send(world.state(counts, sensors, readouts[0].learning))
            elif tick.iteration % (20 * TICKS_PER_SECOND) == 0 and tick.iteration:
                recent = world.approach_times[-5:]
                mean = f"{np.mean(recent):.1f}s" if recent else "--"
                print(f"t={tick.iteration // TICKS_PER_SECOND:>4}s  found {world.found}  crashes {world.crashes}  "
                      f"approach time (last 5) {mean}", flush=True)

    print(f"\nbeacons found: {world.found}, crashes: {world.crashes}")


if __name__ == "__main__":
    main()
