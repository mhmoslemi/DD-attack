"""
Config loading.

A config is a nested YAML file. We parse it into nested SimpleNamespace objects
so that the rest of the code can read hyper-parameters with dotted attribute
access, e.g. `cfg.condensation.ipc`, `cfg.poison.eps`.

Command-line overrides use dotted keys and YAML-typed values, e.g.

    --override poison.eps=0.501961 experiment.attack=dmpoison condensation.ipc=10

Each override value is parsed with `yaml.safe_load`, so ints/floats/bools/lists
are all handled correctly.
"""
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _ns_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_ns_to_dict(v) for v in ns]
    return ns


def _set_dotted(d, dotted_key, value):
    keys = dotted_key.split('.')
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_config(path, overrides=None):
    """Load YAML at `path`, apply dotted `overrides` (list of 'a.b=value'),
    return a nested SimpleNamespace."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            if '=' not in ov:
                raise ValueError('override must be key=value, got %r' % ov)
            key, raw = ov.split('=', 1)
            value = yaml.safe_load(raw)  # type-aware parse
            _set_dotted(cfg, key.strip(), value)

    return _to_ns(cfg)


def config_to_dict(ns):
    """Inverse of load_config, for serialising the resolved config to JSON."""
    return _ns_to_dict(ns)
