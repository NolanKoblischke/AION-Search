"""Galaxy Zoo evaluation configuration."""

from pathlib import Path
from .base import EvalConfig


class GalaxyZooConfig(EvalConfig):
    """Configuration for Galaxy Zoo evaluation."""
    
    @property
    def name(self) -> str:
        return "galaxyzoo"
    
    @property
    def display_name(self) -> str:
        return "Galaxy Zoo"
    
    @property
    def output_prefix(self) -> str:
        return "galaxy"
    
    @property
    def hdf5_path(self) -> str:
        return "galaxybench/dataprep/galaxyzoo/gz5_selected_galaxies.hdf5"
    
    @property
    def hdf5_table_name(self) -> str:
        return "__astropy_table__"
    
    def get_loader_class(self):
        from ..loaders.galaxyzoo import GalaxyZooLoader
        return GalaxyZooLoader
    
    def get_judge_class(self):
        from ..judges.galaxyzoo import GalaxyZooJudge
        return GalaxyZooJudge
    
    def get_display_title(self) -> str:
        return "Galaxy Viewer"
    
    def get_galaxy_icon(self) -> str:
        return "fa-star"
    
    def get_score_field_name(self) -> str:
        return "decision_tree_score"
    
    def get_sort_options(self) -> list:
        return [
            {"value": "volunteer-path-length", "label": "Path Length ↓"},
            {"value": "volunteer-path-length-short", "label": "Path Length ↑"}
        ]

    def get_comparison_function_name(self) -> str:
        """Get the name of the JavaScript function used to create comparison displays."""
        return "createDecisionTreeComparison"
    
    @property
    def js_file(self) -> str:
        """Path to the external JavaScript file for Galaxy Zoo config."""
        return "display_config/galaxyzoo.js"
    
    def get_comparison_javascript(self) -> str:
        """Get JavaScript code for creating Galaxy Zoo decision tree comparison displays."""
        # This method is kept for backward compatibility but will be replaced
        # by loading from the external JS file
        return ""


# Create an instance to register this config
GalaxyZooConfig() 