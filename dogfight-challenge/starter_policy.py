import torch
import torch.nn as nn
import numpy as np


class StarterPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(224, 256),
            nn.LayerNorm(256, eps=1e-05),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256, eps=1e-05),
            nn.ReLU(),
        )
        self.cont_head = nn.Linear(256, 2)
        self.shoot_head = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.backbone(obs)
        cont = self.cont_head(x)
        steer = torch.clamp(cont[:, 0:1], -1.0, 1.0)
        throttle = torch.clamp(cont[:, 1:2], 0.0, 1.0)
        shoot = self.shoot_head(x)
        return torch.cat([steer, throttle, shoot], dim=-1)


def load_from_weights(path="starter_weights.npz"):
    model = StarterPolicy()
    data = np.load(path)
    state_dict = {k: torch.tensor(data[k]) for k in data.files}
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = load_from_weights()
    model.eval()
    obs = torch.zeros(1, 224)
    with torch.no_grad():
        action = model(obs)
    print(f"Input shape:  {obs.shape}")
    print(f"Output shape: {action.shape}")
    print(f"Output values: {action}")
