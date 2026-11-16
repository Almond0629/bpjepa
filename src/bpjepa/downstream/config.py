"""Load downstream configurations and bind data paths from the command line."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def load_downstream_config(path: Path, *, frames_dir: Path, label_cache_dir: Path, soccernet_labels_dir: Path,
                           num_workers: int | None = None, devices: int | None = None) -> dict:
    """frames_dir holds train.json, valid.json, train/, valid/; soccernet_labels_dir holds <match>/Labels-v2.json of the validation split."""
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    for split, stage in (("train", "train"), ("valid", "val")):
        dataset = config["datamodule"][stage]["dataset"]
        dataset["datadir"] = str(Path(frames_dir) / f"{split}.json")
        dataset["video_path_prefix"] = str(Path(frames_dir) / split)
        dataset["label_args"]["cache_dir"] = str(Path(label_cache_dir) / split)
        if num_workers is not None:
            config["datamodule"][stage]["loader"]["num_workers"] = num_workers
    config["model"]["evaluation_args"]["SoccerNet_path"] = str(soccernet_labels_dir)
    if devices is not None:
        config["trainer"]["devices"] = devices
    for required in (Path(frames_dir) / "train.json", Path(frames_dir) / "valid.json", Path(soccernet_labels_dir)):
        if not required.exists():
            raise FileNotFoundError(f"required input is missing: {required}")
    return config


def apply_overrides(config: dict, items: list[str] | None) -> dict:
    """Smoke runs only: --override a.b.c=value sets one config value; the resolved config records it."""
    for item in items or ():
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"expected key=value, got {item!r}")
        *parents, last = key.split(".")
        node = config
        for part in parents:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return config


def write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(config))), resolve=True))
