"""Python port of the 4 built-in opponent policies + shared tactics.
Matches crates/sim/src/opponents/ exactly.
"""
import math
from dataclasses import dataclass
from sim import (
    Action, SimConfig, ARENA_DIAMETER, MAX_ALTITUDE, MAX_ENERGY,
    MAX_TURN_RATE, STALL_SPEED, ALT_BOUNDARY_HIGH,
    SINGLE_FRAME_OBS_SIZE, MAX_BULLET_SLOTS, normalize_angle,
)
import numpy as np

PI = math.pi


# ══════════════════════════════════════════════════════════════════════
# Shared Tactics (from tactics.rs)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TacticalState:
    my_speed: float = 0.0
    my_yaw: float = 0.0
    my_hp: float = 0.0
    gun_cooldown: float = 0.0
    altitude: float = 0.0
    rel_x: float = 0.0
    rel_y: float = 0.0
    opp_speed: float = 0.0
    opp_yaw: float = 0.0
    opp_hp: float = 0.0
    distance: float = 0.0
    angle_to_opp: float = 0.0
    angle_off_nose: float = 0.0
    opp_angle_to_me: float = 0.0
    closing_rate: float = 0.0
    my_energy: float = 0.0
    opp_energy: float = 0.0
    energy_advantage: float = 0.0
    altitude_advantage: float = 0.0
    am_behind_opponent: bool = False
    opponent_behind_me: bool = False
    would_be_rear_aspect_shot: bool = False
    nearest_enemy_bullet_dist: float = float('inf')
    nearest_enemy_bullet_angle: float = 0.0
    enemy_bullet_threat_count: int = 0
    ticks_remaining_frac: float = 1.0


def extract_tactical_state(obs: np.ndarray, config: SimConfig) -> TacticalState:
    d = obs
    my_speed = d[0] * config.max_speed
    my_yaw = math.atan2(d[2], d[1])
    my_hp = d[3]
    gun_cooldown = d[4]
    altitude = d[5] * MAX_ALTITUDE
    my_energy = d[7] * MAX_ENERGY

    rel_x = d[8] * ARENA_DIAMETER
    rel_y = d[9] * ARENA_DIAMETER
    opp_speed = d[10] * config.max_speed
    opp_yaw = math.atan2(d[12], d[11])
    opp_hp = d[13]
    distance = d[14] * ARENA_DIAMETER
    closing_rate = d[15] * config.max_speed
    opp_energy = d[17] * MAX_ENERGY

    angle_to_opp = math.atan2(rel_y, rel_x)
    angle_off_nose = d[51] * PI
    opp_angle_to_me = d[52] * PI

    energy_advantage = my_energy / opp_energy if opp_energy > 1.0 else 2.0
    opp_altitude = altitude + rel_y
    altitude_advantage = altitude - opp_altitude

    am_behind_opponent = abs(opp_angle_to_me) > 2.0
    opponent_behind_me = abs(angle_off_nose) > 2.0

    would_be_rear_aspect = _is_rear_aspect_shot(
        rel_x, rel_y, opp_yaw, my_yaw, config.rear_aspect_cone
    )

    nearest_dist, nearest_angle, threat_count = _compute_bullet_threats(d)

    return TacticalState(
        my_speed=my_speed, my_yaw=my_yaw, my_hp=my_hp,
        gun_cooldown=gun_cooldown, altitude=altitude,
        rel_x=rel_x, rel_y=rel_y, opp_speed=opp_speed,
        opp_yaw=opp_yaw, opp_hp=opp_hp, distance=distance,
        angle_to_opp=angle_to_opp, angle_off_nose=angle_off_nose,
        opp_angle_to_me=opp_angle_to_me, closing_rate=closing_rate,
        my_energy=my_energy, opp_energy=opp_energy,
        energy_advantage=energy_advantage, altitude_advantage=altitude_advantage,
        am_behind_opponent=am_behind_opponent, opponent_behind_me=opponent_behind_me,
        would_be_rear_aspect_shot=would_be_rear_aspect,
        nearest_enemy_bullet_dist=nearest_dist,
        nearest_enemy_bullet_angle=nearest_angle,
        enemy_bullet_threat_count=threat_count,
        ticks_remaining_frac=d[55],
    )


def _is_rear_aspect_shot(rel_x, rel_y, opp_yaw, my_yaw, rear_aspect_cone):
    angle_to_opp = math.atan2(rel_y, rel_x)
    angle_off = abs(angle_diff(angle_to_opp, my_yaw))
    if angle_off > 0.5:
        return False
    bullet_dir_x = math.cos(my_yaw)
    bullet_dir_y = math.sin(my_yaw)
    opp_fwd_x = math.cos(opp_yaw)
    opp_fwd_y = math.sin(opp_yaw)
    dot = bullet_dir_x * opp_fwd_x + bullet_dir_y * opp_fwd_y
    return dot > math.cos(rear_aspect_cone)


def _compute_bullet_threats(obs):
    nearest_dist = float('inf')
    nearest_angle = 0.0
    threat_count = 0
    for slot in range(MAX_BULLET_SLOTS):
        base = 19 + slot * 4
        is_friendly = obs[base + 2]
        if is_friendly > 0.5:
            continue
        bx, by = obs[base], obs[base + 1]
        if bx == 0.0 and by == 0.0:
            continue
        dist = math.sqrt(bx * bx + by * by) * ARENA_DIAMETER
        angle = obs[base + 3] * PI
        if dist < 150.0:
            threat_count += 1
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_angle = angle
    return nearest_dist, nearest_angle, threat_count


def angle_diff(target, current):
    d = target - current
    while d > PI:
        d -= 2 * PI
    while d < -PI:
        d += 2 * PI
    return d


def yaw_toward(desired, current, gain):
    d = angle_diff(desired, current)
    return max(-1.0, min(1.0, d * gain))


def altitude_safety(altitude, yaw):
    pull_up = 1.0 if math.cos(yaw) > 0.0 else -1.0
    if altitude < 50.0 and math.sin(yaw) < 0.5:
        return pull_up
    if altitude < 100.0 and math.sin(yaw) < 0.0:
        return pull_up
    if altitude < 150.0 and math.sin(yaw) < -0.3:
        return pull_up
    if altitude > 570.0 and math.sin(yaw) > 0.3:
        push_down = -1.0 if math.cos(yaw) > 0.0 else 1.0
        return push_down
    return None


def can_shoot(ts, angle_threshold, distance_threshold):
    return (
        abs(ts.angle_off_nose) < angle_threshold
        and ts.distance < distance_threshold
        and ts.gun_cooldown < 0.01
        and not ts.would_be_rear_aspect_shot
    )


def lead_aim(ts, config, lead_factor):
    time_to_target = ts.distance / config.bullet_speed
    opp_fwd_x = math.cos(ts.opp_yaw)
    opp_fwd_y = math.sin(ts.opp_yaw)
    lead_x = ts.rel_x + opp_fwd_x * ts.opp_speed * time_to_target * lead_factor
    lead_y = ts.rel_y + opp_fwd_y * ts.opp_speed * time_to_target * lead_factor
    return math.atan2(lead_y, lead_x)


def crossing_aim(ts, config, lead_factor):
    time_to_target = ts.distance / config.bullet_speed
    opp_fwd_x = math.cos(ts.opp_yaw)
    opp_fwd_y = math.sin(ts.opp_yaw)
    lead_x = ts.rel_x + opp_fwd_x * ts.opp_speed * time_to_target * lead_factor
    lead_y = ts.rel_y + opp_fwd_y * ts.opp_speed * time_to_target * lead_factor
    perp_x = -opp_fwd_y
    perp_y = opp_fwd_x
    side = 1.0 if (ts.rel_x * perp_x + ts.rel_y * perp_y) > 0.0 else -1.0
    offset = 60.0
    return math.atan2(lead_y + perp_y * offset * side, lead_x + perp_x * offset * side)


def smart_aim(ts, config, lead_factor):
    if ts.am_behind_opponent:
        return crossing_aim(ts, config, lead_factor)
    return lead_aim(ts, config, lead_factor)


def stall_avoidance(speed, yaw_input):
    if speed < STALL_SPEED + 15.0:
        urgency = max(0.0, min(1.0, (STALL_SPEED + 15.0 - speed) / 15.0))
        max_yaw = 1.0 - urgency * 0.7
        min_throttle = urgency * 0.8
        return max(-max_yaw, min(max_yaw, yaw_input)), min_throttle
    return yaw_input, 0.0


# ══════════════════════════════════════════════════════════════════════
# Chaser Policy
# ══════════════════════════════════════════════════════════════════════

class ChaserPolicy:
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self.evade_timer = 0
        self.evade_dir = 1.0
        self.yo_yo_timer = 0
        self.yo_yo_phase = 0.0

    def act(self, obs: np.ndarray) -> Action:
        ts = extract_tactical_state(obs, self.config)

        yaw_override = altitude_safety(ts.altitude, ts.my_yaw)
        if yaw_override is not None:
            return Action(yaw_input=yaw_override, throttle=1.0, shoot=False)

        if self.evade_timer > 0:
            self.evade_timer -= 1
        if self.yo_yo_timer > 0:
            self.yo_yo_timer -= 1

        if ts.nearest_enemy_bullet_dist < 80.0 and self.evade_timer == 0:
            self.evade_timer = 15
            self.evade_dir = -self.evade_dir

        if self.evade_timer > 0:
            return Action(
                yaw_input=self.evade_dir, throttle=0.8,
                shoot=can_shoot(ts, 0.3, 320.0),
            )

        if self.yo_yo_timer == 0:
            if ts.distance < 120.0 and ts.closing_rate > 50.0 and ts.altitude < 500.0:
                self.yo_yo_timer = 40
                self.yo_yo_phase = 1.0
            elif ts.distance > 300.0 and ts.closing_rate < -20.0 and ts.altitude > 150.0:
                self.yo_yo_timer = 30
                self.yo_yo_phase = -1.0

        desired_yaw = smart_aim(ts, self.config, 1.0)
        yaw_d = angle_diff(desired_yaw, ts.my_yaw)

        if self.yo_yo_timer > 0:
            yaw_d += self.yo_yo_phase * 0.4

        if ts.altitude < 120.0 and math.sin(ts.my_yaw) < 0.1:
            yaw_d += 0.25
        elif ts.altitude > 520.0 and math.sin(ts.my_yaw) > -0.1:
            yaw_d -= 0.25

        yaw_input = max(-1.0, min(1.0, yaw_d * 2.5))

        throttle = 0.7 if ts.my_speed > 120.0 and abs(yaw_input) > 0.7 else 1.0

        shoot = can_shoot(ts, 0.22, 320.0)

        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)

        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)


# ══════════════════════════════════════════════════════════════════════
# Ace Policy
# ══════════════════════════════════════════════════════════════════════

class AcePolicy:
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self.evade_timer = 0
        self.evade_dir = 1.0

    def act(self, obs: np.ndarray) -> Action:
        ts = extract_tactical_state(obs, self.config)

        yaw_override = altitude_safety(ts.altitude, ts.my_yaw)
        if yaw_override is not None:
            return Action(yaw_input=yaw_override, throttle=1.0, shoot=False)

        if self.evade_timer > 0:
            self.evade_timer -= 1

        if ts.nearest_enemy_bullet_dist < 130.0 and self.evade_timer == 0:
            self.evade_timer = 18
            self.evade_dir = -self.evade_dir

        if self.evade_timer > 0:
            return self._act_evade(ts)

        if ts.opponent_behind_me and ts.distance < 250.0:
            return self._act_defend(ts)

        return self._act_pursue(ts)

    def _act_evade(self, ts):
        evade_yaw = self.evade_dir
        evading_downward = evade_yaw * math.cos(ts.my_yaw) < 0.0
        if ts.altitude < 120.0 and evading_downward and math.sin(ts.my_yaw) < 0.1:
            evade_yaw = 1.0 if math.cos(ts.my_yaw) > 0.0 else -1.0
        evading_upward = evade_yaw * math.cos(ts.my_yaw) > 0.0
        if ts.altitude > 540.0 and evading_upward and math.sin(ts.my_yaw) > 0.0:
            evade_yaw = -1.0 if math.cos(ts.my_yaw) > 0.0 else 1.0

        yaw_input, min_thr = stall_avoidance(ts.my_speed, evade_yaw)
        throttle = max(0.7, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=can_shoot(ts, 0.30, 380.0))

    def _act_defend(self, ts):
        break_dir = 1.0 if ts.angle_off_nose > 0.0 else -1.0
        perp_yaw = ts.angle_to_opp + (PI / 2.0) * break_dir
        target_yaw = perp_yaw + 0.15
        yaw_input = yaw_toward(target_yaw, ts.my_yaw, 3.5)
        throttle = 0.3 if ts.my_speed > 100.0 else 0.5
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=can_shoot(ts, 0.35, 350.0))

    def _act_pursue(self, ts):
        desired_yaw = smart_aim(ts, self.config, 1.0)
        yaw_d = angle_diff(desired_yaw, ts.my_yaw)

        if ts.altitude < 250.0:
            urgency = min(1.0, (250.0 - ts.altitude) / 100.0)
            if math.sin(ts.my_yaw) < 0.2:
                yaw_d += urgency * 0.35
        elif ts.altitude > 480.0:
            urgency = min(1.0, (ts.altitude - 480.0) / 80.0)
            if math.sin(ts.my_yaw) > -0.2:
                yaw_d -= urgency * 0.3

        if ts.my_speed < 90.0 and ts.altitude > 300.0 and math.sin(ts.my_yaw) > -0.1:
            yaw_d -= 0.08

        yaw_input = max(-1.0, min(1.0, yaw_d * 3.0))

        if ts.my_speed < 80.0:
            throttle = 1.0
        elif abs(yaw_input) > 0.7:
            throttle = 0.5
        elif ts.distance > 250.0:
            throttle = 1.0
        else:
            throttle = 0.7

        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=can_shoot(ts, 0.20, 380.0))


# ══════════════════════════════════════════════════════════════════════
# Dogfighter Policy
# ══════════════════════════════════════════════════════════════════════

def _altitude_bias(altitude, yaw, speed):
    bias = 0.0
    if altitude > ALT_BOUNDARY_HIGH:
        urgency = min(1.0, (altitude - ALT_BOUNDARY_HIGH) / 50.0)
        if math.sin(yaw) > 0.0:
            bias -= urgency * 0.3
    elif altitude > 450.0 and math.sin(yaw) > 0.3:
        bias -= 0.1

    if altitude < 130.0:
        urgency = min(1.0, (130.0 - altitude) / 60.0)
        if math.sin(yaw) < 0.1:
            bias += urgency * 0.4
    elif altitude < 180.0 and math.sin(yaw) < -0.2:
        bias += 0.15

    if speed < 100.0 and altitude > 250.0 and math.sin(yaw) > -0.2:
        bias -= 0.08
    return bias


class DogfighterPolicy:
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self.mode = "attack"
        self.mode_timer = 0
        self.attack_patience = 0
        self.last_distance = 400.0
        self.evade_timer = 0
        self.evade_dir = 1.0

    def act(self, obs: np.ndarray) -> Action:
        ts = extract_tactical_state(obs, self.config)

        yaw_override = altitude_safety(ts.altitude, ts.my_yaw)
        if yaw_override is not None:
            return Action(yaw_input=yaw_override, throttle=1.0, shoot=False)

        if self.mode_timer > 0:
            self.mode_timer -= 1
        if self.evade_timer > 0:
            self.evade_timer -= 1

        if ts.nearest_enemy_bullet_dist < 90.0 and self.evade_timer == 0:
            self.evade_timer = 18
            self.evade_dir = -self.evade_dir

        if self.evade_timer > 0:
            return self._act_evade(ts)

        self._update_mode(ts)

        if self.mode == "attack":
            self.attack_patience += 1
        else:
            self.attack_patience = 0
        self.last_distance = ts.distance

        if self.mode == "attack":
            return self._act_attack(ts)
        elif self.mode == "defend":
            return self._act_defend(ts)
        elif self.mode == "energy":
            return self._act_energy(ts)
        else:  # disengage → immediately attack
            return self._act_attack(ts)

    def _update_mode(self, ts):
        if self.mode_timer > 0:
            return
        new_mode = None
        if self.mode == "attack":
            if ts.opponent_behind_me and ts.distance < 200.0:
                new_mode = "defend"
            elif ts.energy_advantage < 0.5 and ts.altitude > 100.0:
                new_mode = "energy"
        elif self.mode == "defend":
            if not ts.opponent_behind_me or ts.distance > 300.0:
                new_mode = "attack"
            elif ts.energy_advantage < 0.4:
                new_mode = "energy"
        elif self.mode == "energy":
            if ts.energy_advantage > 0.7 or ts.distance < 200.0:
                new_mode = "attack"
            elif ts.opponent_behind_me and ts.distance < 150.0:
                new_mode = "defend"
        else:  # disengage
            new_mode = "attack"

        if new_mode is not None:
            self.mode = new_mode
            timers = {"attack": 30, "defend": 45, "energy": 30, "disengage": 0}
            self.mode_timer = timers.get(new_mode, 0)

    def _act_evade(self, ts):
        shoot = can_shoot(ts, 0.3, 350.0)
        evading_downward = self.evade_dir * math.cos(ts.my_yaw) < 0.0
        if ts.altitude < 80.0 and evading_downward and math.sin(ts.my_yaw) < 0.1:
            evade_yaw = 1.0 if math.cos(ts.my_yaw) > 0.0 else -1.0
        else:
            evade_yaw = self.evade_dir
        yaw_input, min_thr = stall_avoidance(ts.my_speed, evade_yaw)
        throttle = max(0.8, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_attack(self, ts):
        desired_yaw = smart_aim(ts, self.config, 1.0)
        yaw_d = angle_diff(desired_yaw, ts.my_yaw)
        yaw_d += _altitude_bias(ts.altitude, ts.my_yaw, ts.my_speed)
        yaw_input = max(-1.0, min(1.0, yaw_d * 3.0))

        if ts.my_speed < 80.0:
            throttle = 1.0
        elif abs(yaw_input) > 0.7:
            throttle = 0.5
        elif ts.distance > 250.0:
            throttle = 1.0
        else:
            throttle = 0.7

        shoot = can_shoot(ts, 0.22, 350.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_defend(self, ts):
        break_dir = 1.0 if ts.angle_off_nose > 0.0 else -1.0
        perp_yaw = ts.angle_to_opp + (PI / 2.0) * break_dir
        yaw_input = yaw_toward(perp_yaw, ts.my_yaw, 3.5)
        throttle = 0.3 if ts.my_speed > 100.0 else 0.6
        shoot = can_shoot(ts, 0.3, 250.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_energy(self, ts):
        if ts.altitude < 400.0:
            away_x = -ts.rel_x
            climb_angle = math.atan2(0.5, 1.0 if away_x > 0 else -1.0)
            desired_yaw = climb_angle if away_x > 0 else PI - climb_angle
        else:
            desired_yaw = 0.0 if math.cos(ts.my_yaw) > 0.0 else PI

        yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 2.0)
        shoot = can_shoot(ts, 0.25, 300.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(1.0, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)


# ══════════════════════════════════════════════════════════════════════
# Brawler Policy
# ══════════════════════════════════════════════════════════════════════

class BrawlerPolicy:
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self.phase = "close"
        self.phase_timer = 0
        self.jink_timer = 0
        self.jink_dir = 1.0
        self.overshoot_timer = 0

    def act(self, obs: np.ndarray) -> Action:
        ts = extract_tactical_state(obs, self.config)

        yaw_override = altitude_safety(ts.altitude, ts.my_yaw)
        if yaw_override is not None:
            return Action(yaw_input=yaw_override, throttle=1.0, shoot=False)

        if self.phase_timer > 0:
            self.phase_timer -= 1
        if self.jink_timer > 0:
            self.jink_timer -= 1
        if self.overshoot_timer > 0:
            self.overshoot_timer -= 1

        if ts.nearest_enemy_bullet_dist < 100.0 and self.jink_timer == 0:
            self.jink_timer = 8
            self.jink_dir = -self.jink_dir

        self._update_phase(ts)

        if self.phase == "close":
            return self._act_close(ts)
        elif self.phase == "brawl":
            return self._act_brawl(ts)
        elif self.phase == "overshoot_bait":
            return self._act_overshoot_bait(ts)
        elif self.phase == "overshoot_punish":
            return self._act_overshoot_punish(ts)
        else:  # retreat
            return self._act_retreat(ts)

    def _update_phase(self, ts):
        if self.phase_timer > 0:
            return
        new_phase = None
        if self.phase == "close":
            if ts.altitude < 130.0:
                new_phase = "retreat"
            elif ts.distance < 200.0:
                new_phase = "brawl"
        elif self.phase == "brawl":
            if ts.altitude < 120.0:
                new_phase = "retreat"
            elif ts.distance > 300.0:
                new_phase = "close"
            elif ts.opponent_behind_me and ts.distance < 180.0 and ts.closing_rate > 30.0:
                new_phase = "overshoot_bait"
        elif self.phase == "overshoot_bait":
            if ts.altitude < 140.0:
                new_phase = "retreat"
            elif ts.am_behind_opponent or (abs(ts.angle_off_nose) < 1.0 and ts.distance < 200.0):
                new_phase = "overshoot_punish"
            elif ts.distance > 250.0:
                new_phase = "close"
            elif not ts.opponent_behind_me:
                new_phase = "brawl"
        elif self.phase == "overshoot_punish":
            if self.overshoot_timer == 0:
                new_phase = "brawl"
        elif self.phase == "retreat":
            if ts.altitude > 220.0 and ts.my_speed > 80.0:
                new_phase = "close"

        if new_phase is not None:
            self.phase = new_phase
            self.phase_timer = 20
            if new_phase == "overshoot_punish":
                self.overshoot_timer = 60

    def _act_close(self, ts):
        desired_yaw = smart_aim(ts, self.config, 1.0)
        yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 3.0)
        if ts.altitude < 120.0 and math.sin(ts.my_yaw) < 0.0:
            yaw_input = max(-1.0, min(1.0, yaw_input + 0.3))
        shoot = can_shoot(ts, 0.25, 300.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(1.0, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_brawl(self, ts):
        desired_yaw = smart_aim(ts, self.config, 0.7)
        yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 4.0)
        if self.jink_timer > 0:
            yaw_input = self.jink_dir
        if ts.my_speed > 120.0:
            throttle = 0.0
        elif ts.my_speed < 70.0:
            throttle = 0.6
        else:
            throttle = 0.2
        shoot = can_shoot(ts, 0.30, 250.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(throttle, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_overshoot_bait(self, ts):
        perp_yaw = ts.angle_to_opp + (PI / 2.0) * self.jink_dir
        yaw_input = yaw_toward(perp_yaw, ts.my_yaw, 2.0)
        shoot = can_shoot(ts, 0.35, 200.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        return Action(yaw_input=yaw_input, throttle=min_thr, shoot=shoot)

    def _act_overshoot_punish(self, ts):
        desired_yaw = smart_aim(ts, self.config, 0.8)
        yaw_input = yaw_toward(desired_yaw, ts.my_yaw, 4.0)
        shoot = can_shoot(ts, 0.35, 300.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(0.5, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=shoot)

    def _act_retreat(self, ts):
        climb_yaw = 0.5 if math.cos(ts.my_yaw) > 0.0 else PI - 0.5
        yaw_input = yaw_toward(climb_yaw, ts.my_yaw, 2.0)
        yaw_input, min_thr = stall_avoidance(ts.my_speed, yaw_input)
        throttle = max(1.0, min_thr)
        return Action(yaw_input=yaw_input, throttle=throttle, shoot=False)


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def get_opponent(name: str, config=None):
    opponents = {
        "dogfighter": DogfighterPolicy,
        "chaser": ChaserPolicy,
        "ace": AcePolicy,
        "brawler": BrawlerPolicy,
    }
    return opponents[name](config=config)
