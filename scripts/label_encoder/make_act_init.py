"""Build outputs/act_labelenc_init/: a fresh ACT (same config as Kamel's all-data checkpoint) whose ResNet18 is the label-pretrained one.

    /path/to/python -I scripts/label_encoder/make_act_init.py --backbone outputs/label_encoder/resnet18_backbone.pt
Everything except model.backbone.* is re-initialised, so training from this folder tests the new encoder alone
(see decisions/act-label-encoder.md).
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label_data as L

SOURCE = ("brouthen/act_so101_pick_striped_all_pretrain_50k", "050000")
CHUNK_SIZE = 100


def download_source(work: Path) -> Path:
    snapshot_download(SOURCE[0], allow_patterns=[f"{SOURCE[1]}/*"], local_dir=work)
    return work / SOURCE[1]


def build_init(source: Path, backbone_weights: Path, out: Path) -> None:
    config = PreTrainedConfig.from_pretrained(source)
    torch.manual_seed(0)
    policy = ACTPolicy(config)  # random transformer; the ResNet gets ImageNet weights, replaced just below
    # FrozenBatchNorm2d has no num_batches_tracked buffer, so torchvision's counters are dropped.
    weights = {k: v for k, v in torch.load(backbone_weights, map_location="cpu").items() if not k.endswith("num_batches_tracked") and not k.startswith("fc.")}
    missing, unexpected = policy.model.backbone.load_state_dict(weights, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    # PreTrainedConfig swaps cuda for mps on this Mac; the folder must keep the source's device.
    shutil.copy(source / "config.json", out / "config.json")
    for f in source.iterdir():
        if f.name not in ("config.json", "model.safetensors"):
            shutil.copy(f, out / f.name)


def check_forward(out: Path) -> float:
    config = PreTrainedConfig.from_pretrained(out)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(out, config=config)
    delta = {"action": [i / 30 for i in range(CHUNK_SIZE)]}
    ds = LeRobotDataset("molnrb/so101_pick_rb_mixed", root=L.DS, episodes=[0], delta_timestamps=delta)
    pre, _ = make_pre_post_processors(
        config, pretrained_path=out,
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "normalizer_processor": {"stats": ds.meta.stats}},
    )
    batch = pre(next(iter(torch.utils.data.DataLoader(ds, batch_size=2))))
    policy.train()
    loss, _ = policy.forward(batch)
    return float(loss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=Path, default=Path("outputs/label_encoder/resnet18_backbone.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/act_labelenc_init"))
    ap.add_argument("--work", type=Path, default=Path("outputs/_act_source"))
    args = ap.parse_args()
    source = download_source(args.work)
    build_init(source, args.backbone, args.out)
    print("forward loss on a real batch (CPU):", check_forward(args.out))


if __name__ == "__main__":
    main()
