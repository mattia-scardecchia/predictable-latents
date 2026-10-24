"""Compose the published recipes and explicit command-line overrides."""

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).parent / "configs"


def recipe_names():
    return sorted(p.stem for p in (CONFIG_ROOT / "recipes").glob("*.yaml"))


def read_recipe(name):
    if name not in recipe_names():
        raise ValueError(f"Unknown recipe {name!r}; choose from {', '.join(recipe_names())}")
    recipe = OmegaConf.load(CONFIG_ROOT / "recipes" / f"{name}.yaml")
    parent = recipe.pop("extends", None)
    return OmegaConf.merge(read_recipe(parent), recipe) if parent else recipe


def load_config(recipe, stage="ae", *, data_root=None, assets_root=None,
                output_dir=None, seed=None, ae_checkpoint=None, overrides=()):
    """Return a complete configuration; departures from the recipe must be explicit overrides."""
    if stage not in {"ae", "wm"}:
        raise ValueError("stage must be 'ae' or 'wm'")
    specification = read_recipe(recipe)
    if stage == "ae" and specification.get("ae") is None:
        raise ValueError(f"Recipe {recipe!r} uses a pretrained autoencoder; start at stage wm")
    cfg = OmegaConf.load(CONFIG_ROOT / ("autoencoder.yaml" if stage == "ae" else "world_model.yaml"))
    if stage == "ae":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_ROOT / "methods" / f"{specification.method}.yaml"))
    cfg = OmegaConf.merge(cfg, specification.get(stage, {}))
    cfg.id = recipe
    cfg.seed = int(seed if seed is not None else specification.seeds[0])
    if data_root is not None:
        cfg.data.params.root_path = str(Path(data_root).expanduser().resolve())
    if assets_root is not None:
        cfg.assets_root = str(Path(assets_root).expanduser().resolve())
    if output_dir is not None:
        cfg.output_dir = str(Path(output_dir).expanduser().resolve())
    if ae_checkpoint is not None:
        cfg.repr.ckpt = str(Path(ae_checkpoint).expanduser().resolve())
    if cfg.repr.get("type") == "rae":
        cfg.repr.rae.decoder_config_path = str(CONFIG_ROOT / "decoder" / "ViTXL")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
