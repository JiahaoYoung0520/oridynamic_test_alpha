
"""
This code provides a modular guidance engine that can be plugged into
`protenix.model.generator.sample_diffusion`.
"""
from .config import parse_tfg_config
from .engine import TFGEngine
