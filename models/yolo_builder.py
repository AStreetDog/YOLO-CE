"""Custom YOLO model builder that registers GASE and DSG modules.

This module extends the Ultralytics YOLO model parser to recognize
custom module names (GASEConv, P2GuidedP3DN) in YAML config files.
"""

from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn import tasks
from ultralytics.utils import RANK, LOGGER, DEFAULT_CFG_DICT, YAML
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from models.gase import GASEConv
from models.dsg import P2GuidedP3DN


# Modules that follow Conv-like channel scaling
CUSTOM_BASE_MODULES = {
    "GASEConv": GASEConv,
}

# Modules that receive multi-input (list of tensors)
CUSTOM_MULTI_INPUT_MODULES = {
    "P2GuidedP3DN": P2GuidedP3DN,
}

ALL_CUSTOM_MODULES = {**CUSTOM_BASE_MODULES, **CUSTOM_MULTI_INPUT_MODULES}


def parse_model_custom(d, ch, verbose=True):
    """Parse a YOLO model.yaml with support for GASE and DSG custom blocks."""
    import ast
    import math

    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")

    if scales and scale:
        depth, width, max_channels = scales[scale]

    if act:
        from ultralytics.nn.modules.conv import Conv
        Conv.default_act = eval(act)

    ch = [ch[-1]]
    layers, save, c2 = [], [], ch[-1]

    from ultralytics.nn.modules import (
        C3k2, Conv, SPPF, C2PSA, Concat, Detect, nn,
    )

    MODULE_MAP = {
        "Conv": Conv, "C3k2": C3k2, "SPPF": SPPF,
        "C2PSA": C2PSA, "Concat": Concat, "Detect": Detect,
        "nn.Upsample": nn.Upsample,
        **ALL_CUSTOM_MODULES,
    }

    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        m = MODULE_MAP.get(m_name)
        if m is None:
            m = getattr(nn, m_name.split(".")[-1]) if "nn." in m_name else None
        if m is None:
            raise ValueError(f"Unknown module: {m_name}")

        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = ast.literal_eval(a)
                except (ValueError, SyntaxError):
                    pass

        n = max(round(n * depth), 1) if n > 1 else n

        if m in (Conv, GASEConv):
            c1, c2 = ch[f if isinstance(f, int) else f[0]], args[0]
            c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * width / 8) * 8 if c2 != nc else c2
            args = [c1, c2, *args[1:]]
        elif m is C3k2:
            c1, c2 = ch[f if isinstance(f, int) else f[0]], args[0]
            c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * width / 8) * 8 if c2 != nc else c2
            args = [c1, c2, *args[1:]]
            if n > 1:
                args.insert(2, n)
                n = 1
        elif m is SPPF:
            c1 = ch[f if isinstance(f, int) else f[0]]
            c2 = args[0]
            c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * width / 8) * 8
            args = [c1, c2, *args[1:]]
        elif m is C2PSA:
            c1 = ch[f if isinstance(f, int) else f[0]]
            c2 = args[0]
            c2 = min(c2, max_channels)
            c2 = math.ceil(c2 * width / 8) * 8
            args = [c1, c2, *args[1:]]
            if n > 1:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is P2GuidedP3DN:
            c2 = ch[f[-1]]
            c1_list = [ch[x] for x in f]
            args = [c1_list, *args]
        elif m is Detect:
            args = [[ch[x] for x in f]]
            if end2end:
                args.append(nc)
        elif m is nn.Upsample:
            c2 = ch[f if isinstance(f, int) else f[0]]
        else:
            c2 = ch[f if isinstance(f, int) else f[0]]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        m_.i, m_.f, m_.type = i, f, m_name
        save.extend(x % (i + 1) for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class CustomDetectionModel(DetectionModel):
    """Detection model with custom module support."""

    def _parse_model(self, d, ch, verbose=True):
        return parse_model_custom(d, ch, verbose)


class CustomYOLO(YOLO):
    """YOLO wrapper that uses the custom model parser."""

    @property
    def task_map(self):
        return {
            "detect": {
                "model": CustomDetectionModel,
                "trainer": DetectionTrainer,
            }
        }

