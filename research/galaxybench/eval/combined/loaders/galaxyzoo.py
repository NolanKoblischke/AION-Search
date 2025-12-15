"""Galaxy Zoo data loader."""

import json
import h5py
from typing import Dict, Any
from .base import BaseLoader


class GalaxyZooLoader(BaseLoader):
    """Loader for Galaxy Zoo data."""
    
    def __init__(self, hdf5_path, table_name, config=None):
        super().__init__(hdf5_path, table_name)
        self.config = config
    
    def load_galaxy_data(self) -> Dict[str, Dict[str, Any]]:
        """Load galaxy data from Galaxy Zoo HDF5 file."""
        galaxy_lookup = {}
        
        with h5py.File(self.hdf5_path, "r") as f:
            data_table = f[self.table_name]
            
            for i, row in enumerate(data_table):
                # Get object ID
                obj_id_raw = row["object_id"]
                obj_id = self._safe_decode(obj_id_raw)
                
                # Skip if object ID is empty
                if not obj_id or obj_id.strip() == "":
                    continue
                
                # Get decision tree data
                decision_tree_str = row["decision_tree"]
                if isinstance(decision_tree_str, bytes):
                    decision_tree_str = decision_tree_str.decode()
                
                # Parse decision tree
                try:
                    decision_tree_data = json.loads(decision_tree_str)
                except json.JSONDecodeError:
                    decision_tree_data = []
                
                # Store galaxy data
                galaxy_info = {
                    'index': i,
                    'object_id': obj_id,
                    'decision_tree': decision_tree_data,
                    'image_array': row["image_array"]
                }
                
                # Add optional fields if they exist
                field_names = row.dtype.names
                
                if "ra" in field_names:
                    galaxy_info['ra'] = self._safe_float(row["ra"])
                else:
                    galaxy_info['ra'] = None
                    
                if "dec" in field_names:
                    galaxy_info['dec'] = self._safe_float(row["dec"])
                else:
                    galaxy_info['dec'] = None
                    
                if "iauname" in field_names:
                    galaxy_info['iauname'] = self._safe_decode(row["iauname"])
                else:
                    galaxy_info['iauname'] = ""
                
                galaxy_lookup[obj_id] = galaxy_info
        
        return galaxy_lookup 