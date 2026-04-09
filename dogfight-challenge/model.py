"""Policy and value networks for PPO training."""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Beta, Bernoulli


class PolicyNetwork(nn.Module):
    """Actor network. Flat MLP matching starter architecture (Phase A).
    Output: [yaw, throttle, shoot_logit] with proper clamping.
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(224, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.cont_head = nn.Linear(256, 2)  # yaw_mean, throttle_mean
        self.shoot_head = nn.Linear(256, 1)  # shoot logit

        # Learnable log-std for continuous actions
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        """Deterministic forward for ONNX export.
        Returns [yaw, throttle, shoot_logit] with clamping.
        """
        x = self.backbone(obs)
        cont = self.cont_head(x)
        yaw = torch.clamp(cont[:, 0:1], -1.0, 1.0)
        throttle = torch.clamp(cont[:, 1:2], 0.0, 1.0)
        shoot = self.shoot_head(x)
        return torch.cat([yaw, throttle, shoot], dim=-1)

    def get_action_and_value(self, obs, value_net, action=None):
        """Sample action and compute log_prob, entropy, value.

        For training — NOT used in ONNX export.
        Distributions: Gaussian+tanh for yaw, Beta for throttle, Bernoulli for shoot.
        """
        x = self.backbone(obs)
        cont_mean = self.cont_head(x)  # (batch, 2)
        shoot_logit = self.shoot_head(x)  # (batch, 1)

        # Yaw: Gaussian + tanh squash to [-1, 1]
        yaw_std = self.log_std[0].exp().clamp(min=0.01, max=1.0)
        yaw_dist = Normal(cont_mean[:, 0], yaw_std)

        # Throttle: Beta distribution on [0, 1]
        # Convert network output to alpha/beta > 1 for unimodal Beta
        throttle_alpha = torch.nn.functional.softplus(cont_mean[:, 1]) + 1.0
        throttle_beta_param = torch.nn.functional.softplus(self.log_std[1]) + 1.0
        throttle_dist = Beta(throttle_alpha, throttle_beta_param)

        shoot_dist = Bernoulli(logits=shoot_logit.squeeze(-1))

        if action is None:
            yaw_raw = yaw_dist.sample()
            throttle = throttle_dist.sample()
            shoot_sample = shoot_dist.sample()

            yaw = torch.tanh(yaw_raw)
            action = torch.stack([yaw, throttle, shoot_sample], dim=-1)
        else:
            yaw = action[:, 0]
            throttle = action[:, 1]
            shoot_sample = action[:, 2]

            # Inverse tanh for yaw log_prob
            yaw_raw = torch.atanh(yaw.clamp(-0.999, 0.999))
            # Clamp throttle for Beta log_prob numerical stability
            throttle = throttle.clamp(1e-6, 1 - 1e-6)

        # Log probs
        yaw_log_prob = yaw_dist.log_prob(yaw_raw) - torch.log(1 - yaw.pow(2) + 1e-6)
        throttle_log_prob = throttle_dist.log_prob(throttle)
        shoot_log_prob = shoot_dist.log_prob(shoot_sample)

        log_prob = yaw_log_prob + throttle_log_prob + shoot_log_prob
        entropy = yaw_dist.entropy().mean() + throttle_dist.entropy().mean() + shoot_dist.entropy().mean()

        value = value_net(obs)

        return action, log_prob, entropy, value


class ValueNetwork(nn.Module):
    """Critic network. Separate from actor — not exported to ONNX."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(224, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    actor = PolicyNetwork()
    critic = ValueNetwork()
    print(f"Actor params: {count_parameters(actor):,}")
    print(f"Critic params: {count_parameters(critic):,}")

    # Test forward
    obs = torch.randn(4, 224)
    out = actor(obs)
    print(f"Actor output shape: {out.shape}, values: {out[0].detach()}")

    action, log_prob, entropy, value = actor.get_action_and_value(obs, critic)
    print(f"Sampled action: {action[0].detach()}")
    print(f"Log prob: {log_prob[0].item():.3f}")
    print(f"Entropy: {entropy.item():.3f}")
    print(f"Value: {value[0].item():.3f}")
