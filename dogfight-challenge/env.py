"""Gymnasium-compatible dogfight environment with reward shaping."""
import numpy as np
from sim import SimState, SimConfig, Action, CONTROL_PERIOD, MAX_TICKS, OBS_SIZE
from opponents import get_opponent


class DogfightEnv:
    """Single dogfight environment. Agent is player 0, opponent is player 1."""

    def __init__(self, opponent_name="dogfighter", randomize_spawns=True, seed=None,
                 domain_rand=False, domain_rand_pct=0.05):
        self.opponent_name = opponent_name
        self.randomize_spawns = randomize_spawns
        self.domain_rand = domain_rand
        self.domain_rand_pct = domain_rand_pct
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.opponent = None
        self.prev_agent_hp = 0
        self.prev_opp_hp = 0
        self.obs_shape = (OBS_SIZE,)
        self.action_size = 3

    def _make_config(self):
        cfg = SimConfig()
        if self.domain_rand:
            p = self.domain_rand_pct
            cfg.gravity *= self.rng.uniform(1 - p, 1 + p)
            cfg.drag_coeff *= self.rng.uniform(1 - p, 1 + p)
            cfg.turn_bleed_coeff *= self.rng.uniform(1 - p, 1 + p)
            cfg.max_thrust *= self.rng.uniform(1 - p, 1 + p)
            cfg.bullet_speed *= self.rng.uniform(1 - p, 1 + p)
        return cfg

    def reset(self):
        config = self._make_config()
        seed = int(self.rng.integers(0, 2**31))
        self.state = SimState(config=config, seed=seed, randomize=self.randomize_spawns)
        self.opponent = get_opponent(self.opponent_name, config=config)
        self.prev_agent_hp = self.state.fighters[0].hp
        self.prev_opp_hp = self.state.fighters[1].hp
        obs = self.state.observe(0)
        # Also compute initial opponent obs (so frame stacking starts)
        self.state.observe(1)
        return obs

    def step(self, action_raw: np.ndarray):
        """Step the environment by one decision period (CONTROL_PERIOD=10 physics ticks).

        Args:
            action_raw: np.ndarray of shape (3,) — [yaw, throttle, shoot_logit]

        Returns:
            obs, reward, done, truncated, info
        """
        agent_action = Action.from_raw(action_raw)

        # Get opponent action
        opp_obs = self.state.observe(1)
        opp_action = self.opponent.act(opp_obs)

        self.prev_agent_hp = self.state.fighters[0].hp
        self.prev_opp_hp = self.state.fighters[1].hp

        # Step CONTROL_PERIOD physics ticks
        for _ in range(CONTROL_PERIOD):
            self.state.step([agent_action, opp_action])
            if self.state.is_terminal():
                break
        done = self.state.is_terminal()
        truncated = False

        # Compute observation for agent
        obs = self.state.observe(0)

        # Compute reward
        reward = self._compute_reward(done)

        info = {
            "agent_hp": self.state.fighters[0].hp,
            "opp_hp": self.state.fighters[1].hp,
            "agent_alive": self.state.fighters[0].alive,
            "opp_alive": self.state.fighters[1].alive,
            "tick": self.state.tick,
            "p0_hits": self.state.stats.p0_hits,
            "p0_shots": self.state.stats.p0_shots,
            "p1_hits": self.state.stats.p1_hits,
            "ground_death": not self.state.fighters[0].alive and self.state.fighters[0].y <= 5.0,
        }

        if done:
            outcome, reason = self.state.outcome()
            info["outcome"] = outcome
            info["reason"] = reason

        return obs, reward, done, truncated, info

    def _compute_reward(self, done):
        reward = 0.0
        me = self.state.fighters[0]
        opp = self.state.fighters[1]

        # ── Terminal rewards ──
        if done:
            if not me.alive:
                reward += -10.0
                if me.y <= 5.0:  # ground crash
                    reward += -5.0
                return reward
            if not opp.alive:
                reward += 10.0
                return reward
            # Timeout
            hp_diff = me.hp - opp.hp
            reward += hp_diff * 2.0
            return reward

        # ── Survival bonus ──
        reward += 0.05

        # ── Combat rewards ──
        opp_hp_lost = self.prev_opp_hp - opp.hp
        reward += opp_hp_lost * 3.0

        my_hp_lost = self.prev_agent_hp - me.hp
        reward += my_hp_lost * -2.0

        # ── Stall penalty ──
        if me.stall_ticks > 0:
            reward += -0.3

        # ── Ground proximity penalty ──
        altitude = me.y
        if altitude < 150.0:
            reward += -0.5 * (1.0 - altitude / 150.0)

        return reward


class VecDogfightEnv:
    """Vectorized environment running N independent matches in parallel."""

    def __init__(self, num_envs=64, opponent_names=None, randomize_spawns=True,
                 domain_rand=False, domain_rand_pct=0.05, seed=None):
        self.num_envs = num_envs
        self.opponent_names = opponent_names or ["dogfighter"]
        self.randomize_spawns = randomize_spawns
        self.domain_rand = domain_rand
        self.domain_rand_pct = domain_rand_pct
        self.rng = np.random.default_rng(seed)

        self.envs = []
        for i in range(num_envs):
            opp = self.opponent_names[i % len(self.opponent_names)]
            env_seed = int(self.rng.integers(0, 2**31))
            self.envs.append(DogfightEnv(
                opponent_name=opp,
                randomize_spawns=randomize_spawns,
                seed=env_seed,
                domain_rand=domain_rand,
                domain_rand_pct=domain_rand_pct,
            ))

        self.obs_shape = (OBS_SIZE,)

    def reset(self):
        obs = np.zeros((self.num_envs, OBS_SIZE), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs[i] = env.reset()
        return obs

    def step(self, actions: np.ndarray):
        """Step all envs. Auto-resets on done.

        Args:
            actions: (num_envs, 3) array

        Returns:
            obs, rewards, dones, infos
        """
        obs = np.zeros((self.num_envs, OBS_SIZE), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        for i, env in enumerate(self.envs):
            o, r, d, _, info = env.step(actions[i])
            if d:
                # Store terminal info before auto-reset
                info["terminal_observation"] = o
                infos[i] = info
                # Auto-reset with new random opponent
                opp = self.opponent_names[int(self.rng.integers(0, len(self.opponent_names)))]
                env.opponent_name = opp
                o = env.reset()
            obs[i] = o
            rewards[i] = r
            dones[i] = d
            infos[i] = info

        return obs, rewards, dones, infos

    def set_opponents(self, opponent_names):
        """Change the opponent pool for future resets."""
        self.opponent_names = opponent_names
