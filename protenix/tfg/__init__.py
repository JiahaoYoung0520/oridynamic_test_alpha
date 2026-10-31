
# 推理发行版：按符号摘取；来源哈希见runtime_sources.json。
"""
This code provides a modular guidance engine that can be plugged into
`protenix.model.generator.sample_diffusion`.
"""
from .config import parse_tfg_config
from .engine import TFGEngine
