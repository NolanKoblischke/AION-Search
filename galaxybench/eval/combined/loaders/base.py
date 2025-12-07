"""Base data loader class for the unified evaluation framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import h5py


class BaseLoader(ABC):
    """Base class for loading galaxy data from HDF5 files."""
    
    def __init__(self, hdf5_path: Path, table_name: str):
        self.hdf5_path = hdf5_path
        self.table_name = table_name
    
    @abstractmethod
    def load_galaxy_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load galaxy data from HDF5 file.
        
        Returns:
            Dictionary mapping object_id to galaxy data dict containing:
            - index: int
            - object_id: str
            - image_array: numpy array
            - ra: float (optional)
            - dec: float (optional)
            - Additional eval-specific fields
        """
        pass
    
    def _safe_decode(self, value):
        """Safely decode bytes to string."""
        if isinstance(value, bytes):
            return value.decode()
        return str(value) if value is not None else ""
    
    def _safe_float(self, value):
        """Safely convert to float."""
        try:
            return float(value) if value is not None else None
        except:
            return None
    
    def _safe_int(self, value):
        """Safely convert to int."""
        try:
            return int(value) if value is not None else None
        except:
            return None 