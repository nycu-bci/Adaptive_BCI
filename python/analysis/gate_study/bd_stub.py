"""Stand-ins for optional packages of the online system that the lab conda env does not have (braindecode, pylsl).
main/Utils/config.py and the pretraining script import ShallowFBCSPNet / EEGNetv4 / EEGConformer / ATCNet at module level,
and main/__init__.py imports main.Utils.LSL (pylsl). The gate study only uses SCCNet and no LSL stream, so these names
only have to exist. Using one of them raises. Each stand-in is installed only if the real package is missing."""

import importlib.util
import sys
import types


def _placeholder(pkg_name, name):
    def __init__(self, *a, **k):
        raise RuntimeError(f"{pkg_name} is not installed: {name} is only a placeholder (bd_stub.py)")

    return type(name, (), {"__init__": __init__})


def _placeholder_func(pkg_name, name):
    def f(*a, **k):
        raise RuntimeError(f"{pkg_name} is not installed: {name} is only a placeholder (bd_stub.py)")

    return f


def _missing(name):
    if name in sys.modules:  # 已經裝過 (真的或替身)
        return False
    return importlib.util.find_spec(name) is None


def install():
    installed = []
    if _missing("braindecode"):
        pkg = types.ModuleType("braindecode")
        models = types.ModuleType("braindecode.models")
        for name in ("ShallowFBCSPNet", "EEGNetv4", "EEGConformer", "ATCNet"):
            setattr(models, name, _placeholder("braindecode", name))
        pkg.models = models
        sys.modules["braindecode"] = pkg
        sys.modules["braindecode.models"] = models
        installed.append("braindecode")
    if _missing("pylsl"):
        lsl = types.ModuleType("pylsl")
        for name in ("StreamInfo", "StreamOutlet", "StreamInlet"):
            setattr(lsl, name, _placeholder("pylsl", name))
        for name in ("resolve_byprop", "resolve_streams"):
            setattr(lsl, name, _placeholder_func("pylsl", name))
        sys.modules["pylsl"] = lsl
        installed.append("pylsl")
    return installed
