"""Base configuration classes for the unified evaluation framework."""

from abc import ABC, abstractmethod
from typing import Dict, Type, List, Optional, Any
from pathlib import Path

# Global registry for evaluation configurations
_EVAL_CONFIGS: Dict[str, 'EvalConfig'] = {}


class EvalConfig(ABC):
    """Abstract base class for evaluation configurations."""
    
    def __init__(self):
        # Register this config when instantiated
        _EVAL_CONFIGS[self.name] = self
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this evaluation type."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this evaluation type."""
        pass
    
    @property
    @abstractmethod
    def output_prefix(self) -> str:
        """Prefix for output files."""
        pass
    
    @property
    @abstractmethod
    def hdf5_path(self) -> str:
        """Path to the HDF5 data file."""
        pass
    
    @property
    @abstractmethod
    def hdf5_table_name(self) -> str:
        """Name of the table within the HDF5 file."""
        pass
    
    @abstractmethod
    def get_loader_class(self):
        """Get the loader class for this evaluation type."""
        pass
    
    @abstractmethod
    def get_judge_class(self):
        """Get the judge class for this evaluation type."""
        pass
    
    def get_primary_color(self) -> str:
        """Get the primary color for this eval type's display."""
        return "#667eea"  # Default purple-blue
    
    def get_galaxy_icon(self) -> str:
        """Get the FontAwesome icon for galaxies in this eval type."""
        return "fa-star"  # Default star icon
    
    def get_display_title(self) -> str:
        """Get the title for the HTML display."""
        return f"{self.display_name} Explorer"
    
    def get_score_field_name(self) -> str:
        """Get the name of the score field in judging results."""
        return "score"  # Default score field name
    
    def format_galaxy_header_info(self, galaxy_data: Dict[str, Any]) -> str:
        """Format additional info to display in galaxy card headers."""
        ra = galaxy_data.get('ra', 'N/A')
        dec = galaxy_data.get('dec', 'N/A')
        
        if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
            return f'<div style="margin-left: auto; display: flex; gap: 15px; align-items: center; font-size: 0.9em; color: #666;"><div><strong>RA:</strong> {ra:.6f}</div><div><strong>Dec:</strong> {dec:.6f}</div></div>'
        return ""
    
    def format_galaxy_metadata(self, galaxy_data: Dict[str, Any]) -> str:
        """Format metadata section for galaxy cards."""
        return ""  # Default: no metadata section
    
    def get_sort_options(self) -> List[Dict[str, str]]:
        """Get additional sort options specific to this eval type."""
        return []  # Default: no additional sort options

    def get_comparison_function_name(self) -> str:
        """Get the name of the JavaScript function used to create comparison displays."""
        return "createGenericComparison"  # Default function name
    
    @property
    def js_file(self) -> str:
        """Path to the external JavaScript file for this config."""
        return "display_config/base.js"
    
    def get_comparison_javascript(self) -> str:
        """Get JavaScript code for creating eval-specific comparison displays."""
        # This method is kept for backward compatibility but will be replaced
        # by loading from the external JS file
        return ""


class AllEvalConfig(EvalConfig):
    """Special configuration for displaying all evaluation types together."""
    
    @property
    def name(self) -> str:
        return "all"
    
    @property
    def display_name(self) -> str:
        return "All Evaluations"
    
    @property
    def output_prefix(self) -> str:
        return "all_evals"
    
    @property
    def hdf5_path(self) -> str:
        return ""  # Not used for combined display
    
    @property
    def hdf5_table_name(self) -> str:
        return ""  # Not used for combined display
    
    def get_loader_class(self):
        return None  # Not used for combined display
    
    def get_judge_class(self):
        return None  # Not used for combined display
    
    def get_primary_color(self) -> str:
        return "#6f42c1"  # Purple for combined view
    
    def get_galaxy_icon(self) -> str:
        return "fa-layer-group"  # Multi-layer icon for combined
    
    def get_display_title(self) -> str:
        return "Multi-Evaluation Galaxy Explorer"
    
    def get_score_field_name(self) -> str:
        # Return a generic field name - the actual logic will need to 
        # dynamically determine the score field based on record content
        return "score"
    
    def get_sort_options(self) -> List[Dict[str, str]]:
        return [
            {"value": "eval-type", "label": "Evaluation Type"},
            {"value": "volunteer-path-length", "label": "Path Length ↓"},
            {"value": "volunteer-path-length-short", "label": "Path Length ↑"}
        ]
    
    def get_comparison_function_name(self) -> str:
        return "createAllEvalComparison"
    
    @property
    def js_file(self) -> str:
        """Path to the external JavaScript file for combined display."""
        # For AllEvalConfig, we'll need to load multiple JS files
        return None  # Special handling needed for multiple files
    
    def get_comparison_javascript(self) -> str:
        """Get JavaScript code for creating eval-specific comparison displays."""
        # This method is kept for backward compatibility but will be replaced
        # by loading from the external JS files
        return ""


def get_eval_config(eval_type: str) -> EvalConfig:
    """Get configuration for the specified evaluation type."""
    if eval_type == "all" and "all" not in _EVAL_CONFIGS:
        # Create the AllEvalConfig if it doesn't exist
        AllEvalConfig()
    
    if eval_type not in _EVAL_CONFIGS:
        raise ValueError(f"Unknown evaluation type: {eval_type}")
    
    return _EVAL_CONFIGS[eval_type]


def list_eval_types() -> List[str]:
    """List all available evaluation types."""
    # Return all registered types except 'all' (which is special)
    return [name for name in _EVAL_CONFIGS.keys() if name != "all"] 