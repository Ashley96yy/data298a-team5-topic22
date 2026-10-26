"""Validate Part D settings and enumerate configurations without training a model."""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path

GRID = {"hidden_size": [64, 128], "dropout": [0.2, 0.5],
        "learning_rate": [0.001, 0.0005]}
INTEGER_SETTINGS = ("batch_size", "epochs", "num_layers", "num_heads", "early_stopping_patience")


def validate_config(config, require_ready=False):
    """Allow explicit nulls in a planning template; reject them before execution."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(isinstance(config, dict), "Configuration must be an object")
    require(type(config.get("schema_version")) is int and config["schema_version"] == 1,
            "schema_version must be 1")
    for section in ("data", "training", "search", "evaluation"):
        require(isinstance(config.get(section), dict), f"Missing object: {section}")
    require(set(config["search"]) == set(GRID), "Search must contain exactly the three assigned parameters")
    for key, expected in GRID.items():
        values = config["search"][key]
        require(isinstance(values, list) and len(values) == 2 and
                all(type(x) in (int, float) for x in values) and
                set(values) == set(expected), f"search.{key} must contain {expected} exactly once")
    data, training = config["data"], config["training"]
    require(type(data.get("max_seq_len")) is int and data["max_seq_len"] == 50,
            "data.max_seq_len must match the frozen dataset: 50")
    for key in ("expected_dataset", "expected_subset"):
        require(isinstance(data.get(key), str) and bool(data[key].strip()), f"Missing data.{key}")
    for key in INTEGER_SETTINGS:
        require(key in training, f"Missing training.{key}")
        value = training[key]
        require(value is None or (type(value) is int and value > 0), f"training.{key} must be a positive integer or null")
    require("weight_decay" in training, "Missing training.weight_decay")
    value = training["weight_decay"]
    require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            "weight_decay must be finite and nonnegative or null")
    require("seeds" in training, "Missing training.seeds")
    seeds = training["seeds"]
    require(seeds is None or (isinstance(seeds, list) and len(seeds) > 0 and
            all(type(s) is int and 0 <= s < 2**32 for s in seeds) and len(set(seeds)) == len(seeds)),
            "seeds must be unique integers in [0, 2**32), or null")
    heads = training["num_heads"]
    require(heads is None or all(size % heads == 0 for size in GRID["hidden_size"]),
            "num_heads must divide both 64 and 128")
    require(config["evaluation"] == {"selection_split": "val", "selection_metric": "NDCG@10", "sampled": False},
            "Selection must use full-catalog validation NDCG@10")
    require(isinstance(config.get("output_dir"), str) and bool(config["output_dir"].strip()), "output_dir is required")
    require("processed_dir" in data and (data["processed_dir"] is None or
            isinstance(data["processed_dir"], str) and bool(data["processed_dir"].strip())),
            "data.processed_dir must be a nonempty path or null")
    pending = [f"training.{k}" for k in (*INTEGER_SETTINGS, "weight_decay", "seeds") if training[k] is None]
    if data["processed_dir"] is None:
        pending.append("data.processed_dir")
    if require_ready:
        require(not pending, "Unconfirmed settings: " + ", ".join(pending))
    return pending


def load_config(path, require_ready=False):
    with Path(path).open() as stream:
        config = json.load(stream)
    validate_config(config, require_ready)
    return config


def expand_grid(config):
    validate_config(config)
    runs = []
    for values in itertools.product(*(config["search"][k] for k in GRID)):
        run = copy.deepcopy(config)
        run.pop("search")
        run["model_hyperparameters"] = dict(zip(GRID, values))
        runs.append(run)
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.require_ready)
        print(json.dumps({"configuration_count": len(expand_grid(config)),
                          "pending_settings": validate_config(config),
                          "grid": [r["model_hyperparameters"] for r in expand_grid(config)]}, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Configuration error: {error}\n")
