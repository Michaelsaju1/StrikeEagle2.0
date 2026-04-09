"""Compare Python sim vs Rust sim on identical matches.

Runs Rust CLI to get a replay JSON, then replays the same match in Python
and compares fighter state at every replay frame (every 4 ticks).
"""
import json
import subprocess
import sys
import os
import math
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import SimState, SimConfig, Action, CONTROL_PERIOD
from opponents import get_opponent

RUST_CLI = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dogfight-challenge", "target", "release", "dogfight.exe"
)
FRAME_INTERVAL = 4


def run_rust_match(p0, p1, seed, randomize=False):
    """Run a match in Rust and return replay frames."""
    import tempfile
    replay_path = os.path.join(tempfile.gettempdir(), f"parity_{seed}.json")
    cmd = [RUST_CLI, "run", "--p0", p0, "--p1", p1, "--seed", str(seed),
           "--output", replay_path]
    if randomize:
        cmd.append("--randomize")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Rust CLI failed: {result.stderr}")
        return None
    with open(replay_path, "r") as f:
        replay = json.load(f)
    os.unlink(replay_path)
    return replay


def run_python_match(p0_name, p1_name, seed, randomize=False):
    """Run a match in Python, recording state every FRAME_INTERVAL ticks."""
    cfg = SimConfig()
    state = SimState(config=cfg, seed=seed, randomize=randomize)
    opp0 = get_opponent(p0_name, config=cfg)
    opp1 = get_opponent(p1_name, config=cfg)

    frames = []

    def record_frame():
        f0, f1 = state.fighters[0], state.fighters[1]
        frames.append({
            "tick": state.tick,
            "fighters": [
                {"x": f0.x, "y": f0.y, "yaw": f0.yaw, "speed": f0.speed,
                 "hp": f0.hp, "alive": f0.alive, "stalled": f0.stall_ticks > 0},
                {"x": f1.x, "y": f1.y, "yaw": f1.yaw, "speed": f1.speed,
                 "hp": f1.hp, "alive": f1.alive, "stalled": f1.stall_ticks > 0},
            ],
        })

    # Record initial frame
    record_frame()

    # Current actions (updated every CONTROL_PERIOD ticks)
    action0 = Action.none()
    action1 = Action.none()

    while not state.is_terminal():
        # Update actions at decision points
        if state.tick % CONTROL_PERIOD == 0:
            obs0 = state.observe(0)
            obs1 = state.observe(1)
            action0 = opp0.act(obs0)
            action1 = opp1.act(obs1)

        state.step([action0, action1])

        # Record at frame intervals
        if state.tick % FRAME_INTERVAL == 0:
            record_frame()

    # Record final frame if not already captured
    if state.tick % FRAME_INTERVAL != 0:
        record_frame()

    return frames


def compare_frames(rust_frames, python_frames):
    """Compare frames from Rust and Python sims. Returns max divergences."""
    max_div = {"x": 0, "y": 0, "yaw": 0, "speed": 0}
    first_hp_mismatch = None
    first_alive_mismatch = None
    first_stall_mismatch = None

    # Build tick -> frame lookup for Python
    py_by_tick = {f["tick"]: f for f in python_frames}

    compared = 0
    for rf in rust_frames:
        tick = rf["tick"]
        pf = py_by_tick.get(tick)
        if pf is None:
            continue

        compared += 1
        for i in range(2):
            r = rf["fighters"][i]
            p = pf["fighters"][i]

            # Position / speed / yaw divergence
            dx = abs(r["x"] - p["x"])
            dy = abs(r["y"] - p["y"])
            # Angle difference (handle wraparound)
            dyaw = abs(r["yaw"] - p["yaw"])
            if dyaw > math.pi:
                dyaw = 2 * math.pi - dyaw
            dspeed = abs(r["speed"] - p["speed"])

            max_div["x"] = max(max_div["x"], dx)
            max_div["y"] = max(max_div["y"], dy)
            max_div["yaw"] = max(max_div["yaw"], dyaw)
            max_div["speed"] = max(max_div["speed"], dspeed)

            # Discrete state mismatches
            if r["hp"] != p["hp"] and first_hp_mismatch is None:
                first_hp_mismatch = (tick, i, r["hp"], p["hp"])
            if r["alive"] != p["alive"] and first_alive_mismatch is None:
                first_alive_mismatch = (tick, i, r["alive"], p["alive"])
            if r["stalled"] != p["stalled"] and first_stall_mismatch is None:
                first_stall_mismatch = (tick, i, r["stalled"], p["stalled"])

    return {
        "compared_frames": compared,
        "max_divergence": max_div,
        "first_hp_mismatch": first_hp_mismatch,
        "first_alive_mismatch": first_alive_mismatch,
        "first_stall_mismatch": first_stall_mismatch,
    }


def test_match(p0, p1, seed, randomize=False):
    """Run one parity test."""
    label = f"{p0} vs {p1} seed={seed} rand={randomize}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    replay = run_rust_match(p0, p1, seed, randomize)
    if replay is None:
        print("  SKIP: Rust CLI failed")
        return None

    rust_frames = replay["frames"]
    rust_outcome = replay.get("outcome", {})
    print(f"  Rust: {len(rust_frames)} frames, final tick={rust_frames[-1]['tick']}")

    py_frames = run_python_match(p0, p1, seed, randomize)
    print(f"  Python: {len(py_frames)} frames, final tick={py_frames[-1]['tick']}")

    result = compare_frames(rust_frames, py_frames)
    div = result["max_divergence"]

    print(f"  Compared {result['compared_frames']} frames")
    print(f"  Max divergence: x={div['x']:.6f}  y={div['y']:.6f}  "
          f"yaw={div['yaw']:.6f}  speed={div['speed']:.6f}")

    if result["first_hp_mismatch"]:
        t, i, rh, ph = result["first_hp_mismatch"]
        print(f"  HP MISMATCH at tick {t}, fighter {i}: rust={rh} python={ph}")
    if result["first_alive_mismatch"]:
        t, i, ra, pa = result["first_alive_mismatch"]
        print(f"  ALIVE MISMATCH at tick {t}, fighter {i}: rust={ra} python={pa}")
    if result["first_stall_mismatch"]:
        t, i, rs, ps = result["first_stall_mismatch"]
        print(f"  STALL MISMATCH at tick {t}, fighter {i}: rust={rs} python={ps}")

    # Pass/fail
    threshold = 1e-2  # Allow some float divergence
    passed = (div["x"] < threshold and div["y"] < threshold and
              div["yaw"] < threshold and div["speed"] < threshold and
              result["first_hp_mismatch"] is None and
              result["first_alive_mismatch"] is None)

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def main():
    if not os.path.exists(RUST_CLI):
        print(f"Rust CLI not found at {RUST_CLI}")
        print("Run: cd dogfight-challenge && cargo build --release")
        sys.exit(1)

    print("=== Sim Parity Test: Python vs Rust ===")
    print(f"CLI: {RUST_CLI}\n")

    tests = [
        # Default spawns (deterministic)
        ("dogfighter", "chaser", 42, False),
        ("chaser", "dogfighter", 42, False),
        ("ace", "brawler", 42, False),
        ("dogfighter", "brawler", 42, False),
        # Different seeds
        ("dogfighter", "chaser", 123, False),
        ("dogfighter", "chaser", 999, False),
    ]

    results = []
    for p0, p1, seed, rand in tests:
        r = test_match(p0, p1, seed, rand)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {sum(1 for r in results if r)}/{len(results)} passed")
    print(f"{'='*60}")

    sys.exit(0 if all(r for r in results) else 1)


if __name__ == "__main__":
    main()
