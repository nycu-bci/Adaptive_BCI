"""Stand-in for braindecode.models when braindecode is not installed (the lab conda env has no braindecode).
main/Utils/config.py and the pretraining script import ShallowFBCSPNet / EEGNetv4 / EEGConformer / ATCNet at module level;
the gate study only uses SCCNet, so these names only have to exist. Building one of them raises."""

import importlib.util
import sys
import types


def install():
    if importlib.util.find_spec("braindecode") is not None:
        return False
    pkg = types.ModuleType("braindecode")
    models = types.ModuleType("braindecode.models")

    def make(name):
        def __init__(self, *a, **k):
            raise RuntimeError(f"braindecode is not installed: {name} is only a placeholder (bd_stub.py)")

        return type(name, (), {"__init__": __init__})

    for name in ("ShallowFBCSPNet", "EEGNetv4", "EEGConformer", "ATCNet"):
        setattr(models, name, make(name))
    pkg.models = models
    sys.modules["braindecode"] = pkg
    sys.modules["braindecode.models"] = models
    return True
