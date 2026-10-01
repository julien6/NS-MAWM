"""Optional plugin entry points; no core edits are needed for new strategies."""
from importlib.metadata import entry_points

STRATEGY_PLUGINS = {}


def register_strategy(name, *, loss, predict):
    if name in {"none", "projection", "residual", "regularization", "feature_weighting"} or name in STRATEGY_PLUGINS:
        raise ValueError(f"Strategy name already registered: {name}")
    STRATEGY_PLUGINS[name] = {"loss": loss, "predict": predict}


def load_plugins():
    for group, register in (("strategies", register_strategy), ("backbones", register_backbone), ("environments", register_environment)):
        for plugin in entry_points(group="ns_mawm." + group):
            if plugin.value not in _LOADED:
                plugin.load()(register)
                _LOADED.add(plugin.value)


def register_backbone(name, factory):
    from .models import BACKBONES
    if name in BACKBONES:
        raise ValueError(f"Backbone already registered: {name}")
    BACKBONES[name] = factory


def register_environment(name, factory, library_factory, *, options=()):
    from .environments import ENVIRONMENTS
    if name in ENVIRONMENTS:
        raise ValueError(f"Environment already registered: {name}")
    ENVIRONMENTS[name] = factory
    ENVIRONMENT_LIBRARIES[name] = library_factory
    ENVIRONMENT_OPTIONS[name] = set(options)


ENVIRONMENT_LIBRARIES = {}
ENVIRONMENT_OPTIONS = {}

_LOADED = set()
