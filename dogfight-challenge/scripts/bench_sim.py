"""Benchmark sim throughput: raw physics steps/sec and PPO iteration time."""
import time
import numpy as np
import torch
from env import DogfightEnv, VecDogfightEnv
from model import PolicyNetwork, ValueNetwork
from sim import SimState, SimConfig, Action, CONTROL_PERIOD


def bench_raw_sim(n_ticks=100_000):
    """Benchmark raw physics step throughput (no observations, no NN)."""
    cfg = SimConfig()
    state = SimState(config=cfg, seed=42, randomize=True)
    a0 = Action(yaw_input=0.5, throttle=0.8, shoot=False)
    a1 = Action(yaw_input=-0.3, throttle=0.6, shoot=True)

    t0 = time.perf_counter()
    for _ in range(n_ticks):
        state.step([a0, a1])
        if state.is_terminal():
            state = SimState(config=cfg, seed=42, randomize=True)
    elapsed = time.perf_counter() - t0
    print(f"Raw sim:       {n_ticks / elapsed:,.0f} ticks/sec ({elapsed:.2f}s for {n_ticks:,} ticks)")


def bench_vec_env(num_envs, n_decisions=1000):
    """Benchmark vectorized env step throughput (with observations + opponent AI)."""
    env = VecDogfightEnv(num_envs=num_envs, opponent_names=["dogfighter"], seed=42)
    env.reset()
    actions = np.random.randn(num_envs, 3).astype(np.float32)
    actions[:, 0] = np.clip(actions[:, 0], -1, 1)
    actions[:, 1] = np.clip(actions[:, 1], 0, 1)

    t0 = time.perf_counter()
    for _ in range(n_decisions):
        obs, rewards, dones, infos = env.step(actions)
    elapsed = time.perf_counter() - t0

    total_ticks = n_decisions * num_envs * CONTROL_PERIOD
    decision_steps = n_decisions * num_envs
    print(f"VecEnv({num_envs:>3}):  {decision_steps / elapsed:,.0f} decisions/sec "
          f"({total_ticks / elapsed:,.0f} ticks/sec, {elapsed:.2f}s)")
    return decision_steps / elapsed


def bench_ppo_iteration(num_envs=64, n_steps=512):
    """Benchmark one full PPO rollout + update cycle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPPO benchmark (device={device}, num_envs={num_envs}, n_steps={n_steps})")

    actor = PolicyNetwork().to(device)
    critic = ValueNetwork().to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=3e-4, eps=1e-5,
    )

    env = VecDogfightEnv(num_envs=num_envs, opponent_names=["dogfighter"], seed=42)
    batch_size = num_envs * n_steps

    # Rollout phase
    obs = env.reset()
    obs_tensor = torch.FloatTensor(obs)

    obs_buf = torch.zeros((n_steps, num_envs, 224), dtype=torch.float32)
    act_buf = torch.zeros((n_steps, num_envs, 3), dtype=torch.float32)
    logprob_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    reward_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    done_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    value_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)

    t0 = time.perf_counter()

    # Rollout
    actor.eval()
    for step in range(n_steps):
        obs_buf[step] = obs_tensor
        with torch.no_grad():
            obs_gpu = obs_tensor.to(device)
            action, logprob, _, value = actor.get_action_and_value(obs_gpu, critic)
            value_buf[step] = value.cpu()
            act_buf[step] = action.cpu()
            logprob_buf[step] = logprob.cpu()

        action_np = action.cpu().numpy()
        obs, rewards, dones, infos = env.step(action_np)
        obs_tensor = torch.FloatTensor(obs)
        reward_buf[step] = torch.FloatTensor(rewards)
        done_buf[step] = torch.FloatTensor(dones.astype(np.float32))

    t_rollout = time.perf_counter() - t0

    # GAE
    t1 = time.perf_counter()
    with torch.no_grad():
        next_value = critic(obs_tensor.to(device)).cpu()
    advantages = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    lastgae = 0
    gamma, gae_lambda = 0.995, 0.95
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_value
            next_done = torch.FloatTensor(dones.astype(np.float32))
        else:
            next_val = value_buf[t + 1]
            next_done = done_buf[t + 1]
        delta = reward_buf[t] + gamma * next_val * (1 - next_done) - value_buf[t]
        advantages[t] = lastgae = delta + gamma * gae_lambda * (1 - next_done) * lastgae
    returns = advantages + value_buf
    t_gae = time.perf_counter() - t1

    # PPO update (4 epochs)
    t2 = time.perf_counter()
    b_obs = obs_buf.reshape(-1, 224).to(device)
    b_act = act_buf.reshape(-1, 3).to(device)
    b_logprob = logprob_buf.reshape(-1).to(device)
    b_adv = advantages.reshape(-1).to(device)
    b_ret = returns.reshape(-1).to(device)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    actor.train()
    for epoch in range(4):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, 2048):
            end = start + 2048
            mb_idx = indices[start:end]
            _, new_logprob, entropy, new_value = actor.get_action_and_value(
                b_obs[mb_idx], critic, b_act[mb_idx]
            )
            logratio = new_logprob - b_logprob[mb_idx]
            ratio = logratio.exp()
            mb_adv = b_adv[mb_idx]
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((new_value - b_ret[mb_idx]) ** 2).mean()
            loss = pg_loss + 0.5 * v_loss + 0.01 * (-entropy)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 0.5
            )
            optimizer.step()

    t_update = time.perf_counter() - t2
    t_total = time.perf_counter() - t0

    sps = batch_size / t_total
    print(f"  Rollout:  {t_rollout:.2f}s ({batch_size / t_rollout:,.0f} steps/sec)")
    print(f"  GAE:      {t_gae:.3f}s")
    print(f"  Update:   {t_update:.2f}s")
    print(f"  Total:    {t_total:.2f}s")
    print(f"  SPS:      {sps:,.0f} steps/sec")
    print(f"  At this rate, 3M steps = {3_000_000 / sps / 60:.1f} min, "
          f"10M steps = {10_000_000 / sps / 60:.1f} min")


if __name__ == "__main__":
    print("=== Sim Throughput Benchmark ===\n")

    bench_raw_sim(100_000)
    print()

    for n in [1, 16, 64]:
        bench_vec_env(n, n_decisions=500)
    print()

    bench_ppo_iteration(num_envs=64, n_steps=512)
