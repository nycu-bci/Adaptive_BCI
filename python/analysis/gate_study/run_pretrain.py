# python python/analysis/gate_study/run_pretrain.py <pretrain_cross_subject.py arguments>
"""Run else/pretraining/pretrain_cross_subject.py with the braindecode placeholder (bd_stub.py); arguments pass through."""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bd_stub  # noqa: E402

bd_stub.install()
script = os.path.join(os.path.dirname(os.path.dirname(HERE)), "else", "pretraining", "pretrain_cross_subject.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
