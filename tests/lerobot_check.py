import time

import torch

import lerobot
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config

print(f"lerobot     {lerobot.__version__}")
print(f"torch       {torch.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device      {device}")

STATE_DIM, ACTION_DIM, IMG = 6, 6, (3, 96, 128)
B = 4

cfg = make_policy_config(
    "act",
    device=device,
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=IMG),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    },
    pretrained_backbone_weights=None,  # skip ImageNet download
)
policy = get_policy_class("act")(cfg).to(device)
n_params = sum(p.numel() for p in policy.parameters()) / 1e6
print(f"policy      ACT ({n_params:.1f}M params)")

chunk = cfg.chunk_size
batch = {
    "observation.state": torch.randn(B, STATE_DIM, device=device),
    "observation.images.front": torch.rand(B, *IMG, device=device),
    "action": torch.randn(B, chunk, ACTION_DIM, device=device),
    "action_is_pad": torch.zeros(B, chunk, dtype=torch.bool, device=device),
}

# One training step
policy.train()
opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
t0 = time.perf_counter()
loss, _ = policy.forward(batch)
loss.backward()
opt.step()
if device == "cuda":
    torch.cuda.synchronize()
print(f"train step  loss={loss.item():.3f} ({(time.perf_counter() - t0) * 1e3:.0f} ms)")

# One inference step
policy.eval()
policy.reset()
obs = {k: v[:1] for k, v in batch.items() if k.startswith("observation")}
with torch.no_grad():
    action = policy.select_action(obs)
assert action.shape == (1, ACTION_DIM), action.shape
assert action.device.type == device
print(f"inference   action shape {tuple(action.shape)} ok")

if device == "cuda":
    print(f"gpu mem     {torch.cuda.max_memory_allocated() / 2**20:.0f} MB peak")
print("PASS")
