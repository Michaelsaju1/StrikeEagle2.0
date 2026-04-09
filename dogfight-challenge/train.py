"""CleanRL-style PPO training for dogfight agent.
Usage: python train.py [--steps 3000000] [--resume checkpoint.pt]
"""
import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from env import VecDogfightEnv
from model import PolicyNetwork, ValueNetwork, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", type=str, default="dogfighter", help="Opponent name")
    parser.add_argument("--steps", type=int, default=3_000_000, help="Total training steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout length per env")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--ent-coeff", type=float, default=0.01)
    parser.add_argument("--vf-coeff", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=250_000, help="Evaluate every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (default: off)")
    return parser.parse_args()



def evaluate(actor, device, num_matches=20):
    """Run evaluation matches against all 4 opponents. Returns win rates."""
    from env import DogfightEnv
    actor.eval()
    results = {}

    for opp_name in ["dogfighter", "chaser", "ace", "brawler"]:
        wins = 0
        for seed in range(num_matches):
            env = DogfightEnv(opponent_name=opp_name, randomize_spawns=True, seed=seed + 10000)
            obs = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = actor(obs_t).cpu().numpy()[0]
                obs, _, done, _, info = env.step(action)

            if info.get("outcome") == "p0_win":
                wins += 1

        results[opp_name] = wins / num_matches

    actor.train()
    return results


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment — single opponent, no domain randomization
    print(f"Opponent: {args.opponent}")
    env = VecDogfightEnv(
        num_envs=args.num_envs,
        opponent_names=[args.opponent],
        randomize_spawns=True,
        domain_rand=False,
        domain_rand_pct=0.0,
        seed=args.seed,
    )

    # Networks
    actor = PolicyNetwork().to(device)
    critic = ValueNetwork().to(device)
    print(f"Actor params: {count_parameters(actor):,}")
    print(f"Critic params: {count_parameters(critic):,}")

    # Load checkpoint if resuming
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume} at step {global_step}")

    # Optimizer
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr, eps=1e-5,
    )

    # Wandb (off by default)
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="dogfight-rl", config=vars(args),
                       name=f"{args.opponent}_s{args.seed}")
        except Exception:
            print("wandb init failed, disabling")
            use_wandb = False

    # Rollout storage
    num_envs = args.num_envs
    n_steps = args.n_steps
    batch_size = num_envs * n_steps

    obs_buf = torch.zeros((n_steps, num_envs, 224), dtype=torch.float32)
    act_buf = torch.zeros((n_steps, num_envs, 3), dtype=torch.float32)
    logprob_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    reward_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    done_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)
    value_buf = torch.zeros((n_steps, num_envs), dtype=torch.float32)

    # Episode tracking
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_wins = deque(maxlen=100)
    ep_ground_deaths = deque(maxlen=100)

    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training Loop ──
    obs = env.reset()
    obs_tensor = torch.FloatTensor(obs)
    start_time = time.time()
    num_updates = args.steps // batch_size

    print(f"Training for {args.steps:,} steps ({num_updates} updates, batch_size={batch_size:,})")
    print()

    for update in range(num_updates):
        # ── Anneal learning rate ──
        frac = 1.0 - update / num_updates
        lr_now = args.lr_end + frac * (args.lr - args.lr_end)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # ── Rollout ──
        actor.eval()
        for step in range(n_steps):
            obs_buf[step] = obs_tensor

            with torch.no_grad():
                obs_gpu = obs_tensor.to(device)
                action, logprob, _, value = actor.get_action_and_value(obs_gpu, critic)
                value_buf[step] = value.cpu()
                act_buf[step] = action.cpu()
                logprob_buf[step] = logprob.cpu()

            # Convert action to numpy
            action_np = action.cpu().numpy()

            # Step env
            obs, rewards, dones, infos = env.step(action_np)
            obs_tensor = torch.FloatTensor(obs)

            reward_buf[step] = torch.FloatTensor(rewards)
            done_buf[step] = torch.FloatTensor(dones.astype(np.float32))

            # Track episodes
            for i, info in enumerate(infos):
                if dones[i]:
                    outcome = info.get("outcome", "draw")
                    ep_wins.append(1.0 if outcome == "p0_win" else 0.0)
                    ep_ground_deaths.append(1.0 if info.get("ground_death", False) else 0.0)

        # ── Compute advantages (GAE) ──
        with torch.no_grad():
            next_value = critic(obs_tensor.to(device)).cpu()

        advantages = torch.zeros((n_steps, num_envs), dtype=torch.float32)
        lastgae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
                next_done = torch.FloatTensor(dones.astype(np.float32))
            else:
                next_val = value_buf[t + 1]
                next_done = done_buf[t + 1]

            delta = reward_buf[t] + args.gamma * next_val * (1 - next_done) - value_buf[t]
            advantages[t] = lastgae = delta + args.gamma * args.gae_lambda * (1 - next_done) * lastgae

        returns = advantages + value_buf

        # ── Flatten batch ──
        b_obs = obs_buf.reshape(-1, 224).to(device)
        b_act = act_buf.reshape(-1, 3).to(device)
        b_logprob = logprob_buf.reshape(-1).to(device)
        b_adv = advantages.reshape(-1).to(device)
        b_ret = returns.reshape(-1).to(device)
        b_val = value_buf.reshape(-1).to(device)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ── PPO Update ──
        actor.train()
        clipfracs = []
        for epoch in range(args.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = actor.get_action_and_value(
                    b_obs[mb_idx], critic, b_act[mb_idx]
                )

                # Policy loss
                logratio = new_logprob - b_logprob[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_eps).float().mean().item())

                mb_adv = b_adv[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - b_ret[mb_idx]) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy

                loss = pg_loss + args.vf_coeff * v_loss + args.ent_coeff * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

        global_step += batch_size

        # ── Logging ──
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        if update % 5 == 0:
            mean_win = np.mean(ep_wins) if ep_wins else 0
            mean_gd = np.mean(ep_ground_deaths) if ep_ground_deaths else 0
            print(
                f"Step {global_step:>10,} | "
                f"SPS {sps:,.0f} | "
                f"Win {mean_win:.2f} | "
                f"GndDeath {mean_gd:.2f} | "
                f"LR {lr_now:.1e} | "
                f"Ent {args.ent_coeff:.4f} | "
                f"ClipFrac {np.mean(clipfracs):.3f} | "
                f"VLoss {v_loss.item():.3f}"
            )

        if use_wandb:
            log_data = {
                "global_step": global_step,
                "sps": sps,
                "lr": lr_now,
                "entropy_coeff": args.ent_coeff,
                "pg_loss": pg_loss.item(),
                "v_loss": v_loss.item(),
                "entropy": entropy.item(),
                "clipfrac": np.mean(clipfracs),
            }
            if ep_wins:
                log_data["win_rate"] = np.mean(ep_wins)
            if ep_ground_deaths:
                log_data["ground_death_rate"] = np.mean(ep_ground_deaths)
            try:
                import wandb
                wandb.log(log_data)
            except Exception:
                pass

        # ── Save checkpoint ──
        if global_step % args.save_every < batch_size:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_{global_step}.pt")
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

        # ── Evaluation ──
        if global_step % args.eval_every < batch_size:
            print("  Evaluating...")
            win_rates = evaluate(actor, device, num_matches=20)
            composite = (
                win_rates["dogfighter"] * 0.15 +
                win_rates["chaser"] * 0.25 +
                win_rates["ace"] * 0.30 +
                win_rates["brawler"] * 0.30
            )
            print(f"  Win rates: df={win_rates['dogfighter']:.0%} ch={win_rates['chaser']:.0%} "
                  f"ac={win_rates['ace']:.0%} br={win_rates['brawler']:.0%} | "
                  f"Composite={composite:.0%}")

            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "eval/dogfighter": win_rates["dogfighter"],
                        "eval/chaser": win_rates["chaser"],
                        "eval/ace": win_rates["ace"],
                        "eval/brawler": win_rates["brawler"],
                        "eval/composite": composite,
                        "global_step": global_step,
                    })
                except Exception:
                    pass

    # ── Final save ──
    final_path = os.path.join(args.save_dir, f"final_{args.opponent}.pt")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    # Final eval
    print("\nFinal evaluation:")
    win_rates = evaluate(actor, device, num_matches=50)
    composite = (
        win_rates["dogfighter"] * 0.15 +
        win_rates["chaser"] * 0.25 +
        win_rates["ace"] * 0.30 +
        win_rates["brawler"] * 0.30
    )
    print(f"Win rates: df={win_rates['dogfighter']:.0%} ch={win_rates['chaser']:.0%} "
          f"ac={win_rates['ace']:.0%} br={win_rates['brawler']:.0%} | "
          f"Composite={composite:.0%}")


if __name__ == "__main__":
    main()
