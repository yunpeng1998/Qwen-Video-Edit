# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    'i2v-A14B': i2v_A14B,
    'ti2v-5B': ti2v_5B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    # H*W entries for ~0.25MP videos (dims must be divisible by 16:
    # VAE stride 8 x patch 2). Added for enhancing 384x640 outputs.
    '384*640': (384, 640),
    '640*384': (640, 384),
}

MAX_AREA_CONFIGS = {
    '384*640': 384 * 640,
    '640*384': 640 * 384,
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '384*640', '640*384'),
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '384*640', '640*384'),
    'ti2v-5B': ('704*1280', '1280*704'),
}
