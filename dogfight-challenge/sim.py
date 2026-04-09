"""Python port of the Rust dogfight simulator physics.
Matches crates/sim/src/physics.rs and crates/sim/src/observation.rs exactly.
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Constants (from shared/src/constants.rs) ──────────────────────────
TICK_RATE = 120
DT = 1.0 / TICK_RATE
MATCH_DURATION_SECS = 90
MAX_TICKS = TICK_RATE * MATCH_DURATION_SECS  # 10800

ARENA_RADIUS = 500.0
ARENA_DIAMETER = ARENA_RADIUS * 2.0
GRAVITY = 130.0
MAX_ALTITUDE = 600.0
ALT_BOUNDARY_LOW = 50.0
ALT_BOUNDARY_HIGH = 550.0
GROUND_DEATH_ALTITUDE = 5.0
CEILING_SPEED_DRAIN = 80.0

FIGHTER_RADIUS = 8.0
MAX_SPEED = 250.0
MIN_SPEED = 20.0
MAX_THRUST = 180.0
DRAG_COEFF = 0.9
MAX_TURN_RATE = 4.0
MIN_TURN_RATE = 0.8
TURN_BLEED_COEFF = 0.25
MAX_HP = 5

STALL_SPEED = 30.0
STALL_RECOVERY_TICKS = 36
STALL_NOSE_DOWN_RATE = 2.5

DAMAGE_SPEED_PENALTY = 0.03
DAMAGE_TURN_PENALTY = 0.02

BULLET_SPEED = 400.0
BULLET_LIFETIME_TICKS = 60
BULLET_RADIUS = 3.0
GUN_COOLDOWN_TICKS = 90
REAR_ASPECT_CONE = 0.785  # PI/4

SINGLE_FRAME_OBS_SIZE = 56
FRAME_STACK_SIZE = 4
OBS_SIZE = SINGLE_FRAME_OBS_SIZE * FRAME_STACK_SIZE  # 224
ACTION_SIZE = 3
MAX_BULLET_SLOTS = 8
CONTROL_PERIOD = 10

MAX_ENERGY = MAX_SPEED * MAX_SPEED + 2.0 * GRAVITY * MAX_ALTITUDE


# ── Helper functions ──────────────────────────────────────────────────
def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def wrap_x(x: float) -> float:
    while x > ARENA_RADIUS:
        x -= ARENA_DIAMETER
    while x < -ARENA_RADIUS:
        x += ARENA_DIAMETER
    return x


def wrapped_rel(ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    dx = ax - bx
    if dx > ARENA_RADIUS:
        dx -= ARENA_DIAMETER
    elif dx < -ARENA_RADIUS:
        dx += ARENA_DIAMETER
    return dx, ay - by


def wrapped_distance(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = wrapped_rel(ax, ay, bx, by)
    return math.sqrt(dx * dx + dy * dy)


# ── Data classes ──────────────────────────────────────────────────────
@dataclass
class SimConfig:
    gravity: float = GRAVITY
    drag_coeff: float = DRAG_COEFF
    turn_bleed_coeff: float = TURN_BLEED_COEFF
    max_speed: float = MAX_SPEED
    min_speed: float = MIN_SPEED
    max_thrust: float = MAX_THRUST
    bullet_speed: float = BULLET_SPEED
    gun_cooldown_ticks: int = GUN_COOLDOWN_TICKS
    bullet_lifetime_ticks: int = BULLET_LIFETIME_TICKS
    max_hp: int = MAX_HP
    max_turn_rate: float = MAX_TURN_RATE
    min_turn_rate: float = MIN_TURN_RATE
    rear_aspect_cone: float = REAR_ASPECT_CONE


@dataclass
class FighterState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    hp: int = MAX_HP
    gun_cooldown_ticks: int = 0
    alive: bool = True
    stall_ticks: int = 0

    def forward(self) -> tuple[float, float]:
        return math.cos(self.yaw), math.sin(self.yaw)


@dataclass
class Bullet:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    owner: int = 0
    ticks_remaining: int = 0


@dataclass
class Action:
    yaw_input: float = 0.0
    throttle: float = 0.0
    shoot: bool = False

    @staticmethod
    def none():
        return Action()

    @staticmethod
    def from_raw(raw: np.ndarray) -> "Action":
        return Action(
            yaw_input=float(np.clip(raw[0], -1.0, 1.0)),
            throttle=float(np.clip(raw[1], 0.0, 1.0)),
            shoot=bool(raw[2] > 0.0),
        )


@dataclass
class MatchStats:
    p0_hp: int = MAX_HP
    p1_hp: int = MAX_HP
    p0_hits: int = 0
    p1_hits: int = 0
    p0_shots: int = 0
    p1_shots: int = 0


# ── Physics helpers ───────────────────────────────────────────────────
def cfg_turn_rate_at_speed(cfg: SimConfig, speed: float) -> float:
    t = max(0.0, min(1.0, (speed - cfg.min_speed) / (cfg.max_speed - cfg.min_speed)))
    return cfg.max_turn_rate + t * (cfg.min_turn_rate - cfg.max_turn_rate)


def cfg_effective_max_speed(cfg: SimConfig, hp: int) -> float:
    return cfg.max_speed * (1.0 - DAMAGE_SPEED_PENALTY * (cfg.max_hp - hp))


def cfg_effective_turn_rate(cfg: SimConfig, speed: float, hp: int) -> float:
    return cfg_turn_rate_at_speed(cfg, speed) * (1.0 - DAMAGE_TURN_PENALTY * (cfg.max_hp - hp))


def apply_boundaries(f: FighterState) -> bool:
    """Returns True if fighter hit the ground (death)."""
    f.x = wrap_x(f.x)
    if f.y <= GROUND_DEATH_ALTITUDE:
        return True
    if f.y > ALT_BOUNDARY_HIGH:
        penetration = min(1.0, max(0.0, (f.y - ALT_BOUNDARY_HIGH) / (MAX_ALTITUDE - ALT_BOUNDARY_HIGH)))
        f.speed -= penetration * penetration * CEILING_SPEED_DRAIN * DT
        if f.speed < MIN_SPEED:
            f.speed = MIN_SPEED
    if f.y > MAX_ALTITUDE:
        f.y = MAX_ALTITUDE
    return False


# ── SimState ──────────────────────────────────────────────────────────
SPAWN_SPEED = 50.0


class SimState:
    def __init__(self, config: Optional[SimConfig] = None, seed: int = 0, randomize: bool = False):
        self.config = config or SimConfig()
        hp = self.config.max_hp
        self.tick = 0
        self.stats = MatchStats(p0_hp=hp, p1_hp=hp)
        self.bullets: list[Bullet] = []

        if randomize:
            rng = np.random.default_rng(seed)
            x_offset = rng.uniform(100.0, 300.0)
            alt0 = rng.uniform(150.0, 450.0)
            alt1 = rng.uniform(150.0, 450.0)
            yaw0 = rng.uniform(-math.pi, math.pi)
            yaw1 = rng.uniform(-math.pi, math.pi)
            speed0 = rng.uniform(self.config.min_speed + 15.0, 80.0)
            speed1 = rng.uniform(self.config.min_speed + 15.0, 80.0)
            self.fighters = [
                FighterState(x=-x_offset, y=alt0, yaw=yaw0, speed=speed0, hp=hp),
                FighterState(x=x_offset, y=alt1, yaw=yaw1, speed=speed1, hp=hp),
            ]
        else:
            self.fighters = [
                FighterState(x=-200.0, y=300.0, yaw=0.0, speed=SPAWN_SPEED, hp=hp),
                FighterState(x=200.0, y=300.0, yaw=math.pi, speed=SPAWN_SPEED, hp=hp),
            ]

        self.prev_fighters = [FighterState(**f.__dict__) for f in self.fighters]

        # Observation frame history (last 3 frames per player)
        self.obs_history = [[np.zeros(SINGLE_FRAME_OBS_SIZE, dtype=np.float32) for _ in range(3)] for _ in range(2)]
        self.obs_history_count = [0, 0]

    def is_terminal(self) -> bool:
        return self.tick >= MAX_TICKS or not self.fighters[0].alive or not self.fighters[1].alive

    def outcome(self) -> tuple[str, str]:
        """Returns (outcome, reason). outcome in {'p0_win','p1_win','draw'}."""
        p0_alive = self.fighters[0].alive
        p1_alive = self.fighters[1].alive
        if not p0_alive and not p1_alive:
            return "draw", "elimination"
        elif not p1_alive:
            return "p0_win", "elimination"
        elif not p0_alive:
            return "p1_win", "elimination"
        elif self.tick >= MAX_TICKS:
            p0_hp = self.fighters[0].hp
            p1_hp = self.fighters[1].hp
            if p0_hp > p1_hp:
                return "p0_win", "timeout"
            elif p1_hp > p0_hp:
                return "p1_win", "timeout"
            else:
                return "draw", "timeout"
        return "draw", "timeout"

    def step(self, actions: list[Action]):
        self.prev_fighters = [FighterState(**f.__dict__) for f in self.fighters]

        for i, action in enumerate(actions):
            if not self.fighters[i].alive:
                continue
            self._step_fighter(i, action)

        self._step_bullets()
        self._check_collisions()
        self.bullets = [b for b in self.bullets if b.ticks_remaining > 0]

        self.stats.p0_hp = self.fighters[0].hp
        self.stats.p1_hp = self.fighters[1].hp
        self.tick += 1

    def _step_fighter(self, idx: int, action: Action):
        cfg = self.config
        f = self.fighters[idx]

        # ── Stall state ──
        if f.stall_ticks > 0:
            f.stall_ticks -= 1

            # Rotate nose toward straight down (-PI/2)
            target = -math.pi / 2.0
            diff = normalize_angle(target - f.yaw)
            max_rot = STALL_NOSE_DOWN_RATE * DT
            f.yaw += max(-max_rot, min(max_rot, diff))
            f.yaw = normalize_angle(f.yaw)

            # Only gravity and drag during stall
            f.speed += (-cfg.gravity * math.sin(f.yaw)) * DT
            f.speed -= cfg.drag_coeff * f.speed * DT
            f.speed = max(cfg.min_speed, min(cfg_effective_max_speed(cfg, f.hp), f.speed))

            # Early recovery
            if f.speed > STALL_SPEED + 10.0:
                f.stall_ticks = 0

            # Integrate position
            fx, fy = f.forward()
            f.x += fx * f.speed * DT
            f.y += fy * f.speed * DT
            if apply_boundaries(f):
                f.alive = False
                return

            # Cooldown still ticks
            if f.gun_cooldown_ticks > 0:
                f.gun_cooldown_ticks -= 1
            return

        # ── Check stall entry ──
        if f.speed < STALL_SPEED:
            f.stall_ticks = STALL_RECOVERY_TICKS
            return

        # ── Normal flight ──
        turn_rate = cfg_effective_turn_rate(cfg, f.speed, f.hp)
        yaw_input = max(-1.0, min(1.0, action.yaw_input))
        yaw_delta = yaw_input * turn_rate * DT
        f.yaw += yaw_delta
        f.yaw = normalize_angle(f.yaw)

        # Energy bleed from turning
        f.speed -= cfg.turn_bleed_coeff * abs(yaw_delta) * f.speed

        # Thrust and drag
        throttle = max(0.0, min(1.0, action.throttle))
        f.speed += (throttle * cfg.max_thrust - cfg.drag_coeff * f.speed) * DT

        # Gravity
        f.speed += (-cfg.gravity * math.sin(f.yaw)) * DT

        # Clamp speed
        f.speed = max(cfg.min_speed, min(cfg_effective_max_speed(cfg, f.hp), f.speed))

        # Integrate position
        fx, fy = f.forward()
        f.x += fx * f.speed * DT
        f.y += fy * f.speed * DT

        if apply_boundaries(f):
            f.alive = False
            return

        # Gun cooldown
        if f.gun_cooldown_ticks > 0:
            f.gun_cooldown_ticks -= 1

        # Spawn bullet
        if action.shoot and f.gun_cooldown_ticks == 0:
            bvx = fx * cfg.bullet_speed
            bvy = fy * cfg.bullet_speed
            spawn_dist = FIGHTER_RADIUS + BULLET_RADIUS + 1.0
            raw_spawn_x = f.x + fx * spawn_dist
            raw_spawn_y = f.y + fy * spawn_dist
            self.bullets.append(Bullet(
                x=wrap_x(raw_spawn_x), y=raw_spawn_y,
                vx=bvx, vy=bvy,
                owner=idx, ticks_remaining=cfg.bullet_lifetime_ticks,
            ))
            f.gun_cooldown_ticks = cfg.gun_cooldown_ticks
            if idx == 0:
                self.stats.p0_shots += 1
            else:
                self.stats.p1_shots += 1

    def _step_bullets(self):
        for b in self.bullets:
            b.x += b.vx * DT
            b.y += b.vy * DT
            b.x = wrap_x(b.x)
            if b.ticks_remaining > 0:
                b.ticks_remaining -= 1

    def _check_collisions(self):
        collision_dist_sq = (FIGHTER_RADIUS + BULLET_RADIUS) ** 2
        rear_aspect_cos = math.cos(self.config.rear_aspect_cone)

        for b in self.bullets:
            if b.ticks_remaining == 0:
                continue
            for i, f in enumerate(self.fighters):
                if b.owner == i or not f.alive:
                    continue
                dx, dy = wrapped_rel(b.x, b.y, f.x, f.y)
                dist_sq = dx * dx + dy * dy
                if dist_sq <= collision_dist_sq:
                    # Rear-aspect armor check
                    bspeed = math.sqrt(b.vx * b.vx + b.vy * b.vy)
                    if bspeed > 0:
                        bdx, bdy = b.vx / bspeed, b.vy / bspeed
                    else:
                        bdx, bdy = 0.0, 0.0
                    ffx, ffy = f.forward()
                    dot = bdx * ffx + bdy * ffy

                    b.ticks_remaining = 0  # consume bullet regardless

                    if dot > rear_aspect_cos:
                        break  # glanced off

                    f.hp = max(0, f.hp - 1)
                    if f.hp == 0:
                        f.alive = False

                    if b.owner == 0:
                        self.stats.p0_hits += 1
                    else:
                        self.stats.p1_hits += 1
                    break

    # ── Observation ───────────────────────────────────────────────────
    def observe_single_frame(self, player: int) -> np.ndarray:
        data = np.zeros(SINGLE_FRAME_OBS_SIZE, dtype=np.float32)
        cfg = self.config
        opp = 1 - player
        me = self.fighters[player]
        them = self.fighters[opp]
        prev_me = self.prev_fighters[player]
        prev_them = self.prev_fighters[opp]

        # Self state [0..8)
        data[0] = me.speed / cfg.max_speed
        data[1] = math.cos(me.yaw)
        data[2] = math.sin(me.yaw)
        data[3] = me.hp / cfg.max_hp
        data[4] = me.gun_cooldown_ticks / cfg.gun_cooldown_ticks if cfg.gun_cooldown_ticks > 0 else 0.0
        data[5] = me.y / MAX_ALTITUDE
        data[6] = me.x / ARENA_RADIUS
        my_energy = me.speed * me.speed + 2.0 * cfg.gravity * me.y
        data[7] = my_energy / MAX_ENERGY

        # Opponent state [8..19)
        rel_x, rel_y = wrapped_rel(them.x, them.y, me.x, me.y)
        distance = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        data[8] = rel_x / ARENA_DIAMETER
        data[9] = rel_y / ARENA_DIAMETER
        data[10] = them.speed / cfg.max_speed
        data[11] = math.cos(them.yaw)
        data[12] = math.sin(them.yaw)
        data[13] = them.hp / cfg.max_hp
        data[14] = distance / ARENA_DIAMETER

        prev_rel_x, prev_rel_y = wrapped_rel(prev_them.x, prev_them.y, prev_me.x, prev_me.y)
        prev_distance = math.sqrt(prev_rel_x * prev_rel_x + prev_rel_y * prev_rel_y)
        closure_rate = (prev_distance - distance) / DT if self.tick > 0 else 0.0
        data[15] = max(-1.0, min(1.0, closure_rate / cfg.max_speed))

        angular_velocity = normalize_angle(them.yaw - prev_them.yaw) / DT if self.tick > 0 else 0.0
        data[16] = max(-1.0, min(1.0, angular_velocity / MAX_TURN_RATE))

        opp_energy = them.speed * them.speed + 2.0 * cfg.gravity * them.y
        data[17] = opp_energy / MAX_ENERGY

        angle_opp_to_me = math.atan2(-rel_y, -rel_x)
        angle_off_tail = normalize_angle(angle_opp_to_me - them.yaw)
        data[18] = angle_off_tail / math.pi

        # Bullets [19..51) — 8 nearest bullets × 4 floats
        bullet_entries = []
        for b in self.bullets:
            dist = wrapped_distance(b.x, b.y, me.x, me.y)
            bullet_entries.append((dist, b))
        bullet_entries.sort(key=lambda e: e[0])

        for slot, (_, bullet) in enumerate(bullet_entries[:MAX_BULLET_SLOTS]):
            base = 19 + slot * 4
            brel_x, brel_y = wrapped_rel(bullet.x, bullet.y, me.x, me.y)
            data[base] = brel_x / ARENA_DIAMETER
            data[base + 1] = brel_y / ARENA_DIAMETER
            data[base + 2] = 1.0 if bullet.owner == player else 0.0
            angle = math.atan2(brel_y, brel_x)
            data[base + 3] = angle / math.pi

        # Relative geometry [51..55)
        angle_to_opp = math.atan2(rel_y, rel_x)
        angle_off_nose = normalize_angle(angle_to_opp - me.yaw)
        data[51] = angle_off_nose / math.pi

        opp_angle_to_me_dir = math.atan2(-rel_y, -rel_x)
        opp_angle_off_nose = normalize_angle(opp_angle_to_me_dir - them.yaw)
        data[52] = opp_angle_off_nose / math.pi

        my_vx = me.speed * math.cos(me.yaw)
        my_vy = me.speed * math.sin(me.yaw)
        opp_vx = them.speed * math.cos(them.yaw)
        opp_vy = them.speed * math.sin(them.yaw)
        data[53] = (my_vx - opp_vx) / cfg.max_speed
        data[54] = (my_vy - opp_vy) / cfg.max_speed

        # Meta [55]
        ticks_remaining = max(0, MAX_TICKS - self.tick)
        data[55] = ticks_remaining / MAX_TICKS

        return data

    def observe(self, player: int) -> np.ndarray:
        current = self.observe_single_frame(player)
        data = np.zeros(OBS_SIZE, dtype=np.float32)
        data[:SINGLE_FRAME_OBS_SIZE] = current

        count = self.obs_history_count[player]
        for i in range(3):
            dest_start = (i + 1) * SINGLE_FRAME_OBS_SIZE
            if i < count:
                data[dest_start:dest_start + SINGLE_FRAME_OBS_SIZE] = self.obs_history[player][i]

        # Shift history
        self.obs_history[player][2] = self.obs_history[player][1].copy()
        self.obs_history[player][1] = self.obs_history[player][0].copy()
        self.obs_history[player][0] = current.copy()
        if self.obs_history_count[player] < 3:
            self.obs_history_count[player] += 1

        return data
