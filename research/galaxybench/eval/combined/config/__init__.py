from .base import EvalConfig, get_eval_config, list_eval_types
from . import galaxyzoo  # noqa: F401 - registers GalaxyZooConfig

__all__ = ['EvalConfig', 'get_eval_config', 'list_eval_types']
