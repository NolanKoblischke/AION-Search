import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import h5py
import numpy as np
from collections import defaultdict

# File paths
INPUT_FILE = Path("data/gz5_legacysurvey_images.hdf5")
OUTPUT_FILE = Path("gz5_selected_galaxies.hdf5")
OUTPUT_JSON = Path("selected_galaxies_summary.json")
NOTES_DIR = Path("dataprep/galaxyzoo/notes")

# Constants
MAX_PER_UNIQUE_PATH = 5  # Maximum 5 galaxies per unique decision tree path
VOTE_THRESHOLD = 10
AGREEMENT_THRESHOLD = 0.7
CHUNK_SIZE = 30_000

# Define the decision tree flow
# This maps each question to its possible answers and follow-up questions
DECISION_FLOW = {
    "smooth-or-featured": {
        "smooth": "how-rounded",
        "featured-or-disk": "disk-edge-on",
        "artifact": None  # End of path
    },
    "how-rounded": {
        "round": "merging",
        "in-between": "merging",
        "cigar-shaped": "merging"
    },
    "disk-edge-on": {
        "yes": "edge-on-bulge",
        "no": "has-spiral-arms"  # First of multiple questions
    },
    "edge-on-bulge": {
        "boxy": "merging",
        "none": "merging",
        "rounded": "merging"
    },
    "has-spiral-arms": {
        "yes": "spiral-winding",
        "no": "bar"  # Skip to bar question
    },
    "spiral-winding": {
        "tight": "spiral-arm-count",
        "medium": "spiral-arm-count",
        "loose": "spiral-arm-count"
    },
    "spiral-arm-count": {
        "1": "bar",
        "2": "bar",
        "3": "bar",
        "4": "bar",
        "more-than-4": "bar",
        "cant-tell": "bar"
    },
    "bar": {
        "strong": "bulge-size",
        "weak": "bulge-size",
        "no": "bulge-size"
    },
    "bulge-size": {
        "dominant": "merging",
        "large": "merging",
        "moderate": "merging",
        "small": "merging",
        "none": "merging"
    },
    "merging": {
        "none": None,
        "minor-disturbance": None,
        "major-disturbance": None,
        "merger": None
    }
}


def get_galaxy_full_path(row) -> Tuple[bool, List[Tuple[str, str, int, float]], str]:
    """
    Trace the full decision tree path for a galaxy.
    
    Returns:
        (is_complete, path_nodes, path_string)
        where:
        - is_complete: True if all nodes in path meet criteria
        - path_nodes: List of (question, answer, votes, agreement)
        - path_string: Unique string representation of the path
    """
    path_nodes = []
    current_question = "smooth-or-featured"
    
    while current_question:
        # Find which answer has the highest fraction for this question
        best_answer = None
        best_fraction = 0.0
        best_votes = 0
        
        for answer, next_question in DECISION_FLOW.get(current_question, {}).items():
            col_name = f"{current_question}_{answer}"
            fraction_col = f"{col_name}_fraction"
            
            if col_name in row.dtype.names and fraction_col in row.dtype.names:
                votes = int(row[col_name])
                fraction = float(row[fraction_col])
                
                if fraction > best_fraction:
                    best_fraction = fraction
                    best_answer = answer
                    best_votes = votes
        
        if best_answer is None:
            # Can't continue path
            return False, path_nodes, ""
        
        # Check if this node meets our criteria
        if best_votes < VOTE_THRESHOLD or best_fraction < AGREEMENT_THRESHOLD:
            return False, path_nodes, ""
        
        # Add to path
        path_nodes.append((current_question, best_answer, best_votes, best_fraction))
        
        # Move to next question
        next_question = DECISION_FLOW.get(current_question, {}).get(best_answer)
        
        # Special handling for featured galaxies that are not edge-on
        # They need to answer multiple questions in sequence
        if current_question == "disk-edge-on" and best_answer == "no":
            # Will go through: has-spiral-arms -> (possibly spiral questions) -> bar -> bulge-size -> merging
            current_question = "has-spiral-arms"
        elif current_question == "has-spiral-arms" and best_answer == "no":
            # Skip spiral questions, go to bar
            current_question = "bar"
        else:
            current_question = next_question
    
    # Create unique path string
    path_string = " -> ".join([f"{q}:{a}" for q, a, _, _ in path_nodes])
    
    return True, path_nodes, path_string


def create_decision_tree_json(path_nodes: List[Tuple[str, str, int, float]]) -> str:
    """Create a JSON string with the full decision tree path."""
    tree_info = []
    for question, answer, votes, fraction in path_nodes:
        tree_info.append({
            "question": question,
            "answer": answer,
            "node": f"{question}_{answer}",
            "votes": votes,
            "agreement": round(fraction, 3)
        })
    return json.dumps(tree_info)


def load_excluded_galaxy_ids() -> Set[str]:
    """Load galaxy IDs from .txt filenames in the notes directory."""
    excluded_ids = set()
    
    if not NOTES_DIR.exists():
        print(f"Notes directory {NOTES_DIR} does not exist, no galaxies will be excluded.")
        return excluded_ids
    
    txt_files = list(NOTES_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {NOTES_DIR}, no galaxies will be excluded.")
        return excluded_ids
    
    print(f"Loading excluded galaxy IDs from {len(txt_files)} .txt filenames in {NOTES_DIR}...")
    
    for txt_file in txt_files:
        galaxy_id = txt_file.stem
        excluded_ids.add(galaxy_id)
    
    print(f"Loaded {len(excluded_ids)} galaxy IDs to exclude.")
    return excluded_ids


def print_progress(path_galaxies: Dict[str, List], chunk_num: int, total_paths: int):
    """Print progress after processing a chunk."""
    print(f"\n--- Progress after chunk {chunk_num} ---")
    print(f"Total unique paths found: {total_paths}")
    
    # Show top 10 paths by galaxy count
    sorted_paths = sorted(path_galaxies.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("Top 10 paths by galaxy count:")
    for path, galaxies in sorted_paths:
        # Shorten path for display
        path_parts = path.split(" -> ")
        if len(path_parts) > 3:
            short_path = f"{path_parts[0]} -> ... -> {path_parts[-1]}"
        else:
            short_path = path
        print(f"  {short_path}: {len(galaxies)} galaxies")
    
    # Count how many paths have reached the limit
    paths_at_limit = sum(1 for galaxies in path_galaxies.values() if len(galaxies) >= MAX_PER_UNIQUE_PATH)
    print(f"Paths at limit ({MAX_PER_UNIQUE_PATH} galaxies): {paths_at_limit}")
    print()


def main():
    print("Starting Galaxy Zoo fully confident decision tree extraction...")
    print(f"Looking for galaxies with complete confident paths (>={AGREEMENT_THRESHOLD} agreement, >={VOTE_THRESHOLD} votes)")
    print(f"Selecting up to {MAX_PER_UNIQUE_PATH} galaxies per unique decision tree path")
    
    # Load excluded galaxy IDs
    excluded_galaxy_ids = load_excluded_galaxy_ids()
    
    # Dictionary to store galaxies grouped by their unique path
    path_galaxies: Dict[str, List[Tuple[int, str, List[Tuple[str, str, int, float]]]]] = defaultdict(list)
    
    # Set to track all selected galaxy indices
    all_selected_indices: Set[int] = set()
    
    # Track statistics
    total_confident_galaxies = 0
    skipped_excluded = 0
    skipped_incomplete = 0
    
    print(f"Opening HDF5 file: {INPUT_FILE}")
    with h5py.File(INPUT_FILE, "r") as f:
        data_table = f["__astropy_table__"]
        total_rows = len(data_table)
        print(f"Total galaxies in dataset: {total_rows:,}")
        print(f"Processing in chunks of {CHUNK_SIZE:,}")
        
        chunk_num = 0
        
        # Process data in chunks
        for chunk_start in range(0, total_rows, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
            chunk_num += 1
            print(f"\nProcessing chunk {chunk_num} ({chunk_start:,} to {chunk_end-1:,})...")
            
            # Load chunk
            data_chunk = data_table[chunk_start:chunk_end]
            
            # Process each galaxy in chunk
            for idx, row in enumerate(data_chunk):
                global_idx = chunk_start + idx
                
                # Get galaxy ID
                obj_id = row["object_id"]
                if isinstance(obj_id, bytes):
                    obj_id = obj_id.decode()
                else:
                    obj_id = str(obj_id)
                
                # Skip if in exclusion list
                if obj_id in excluded_galaxy_ids:
                    skipped_excluded += 1
                    continue
                
                # Get the full decision path for this galaxy
                is_complete, path_nodes, path_string = get_galaxy_full_path(row)
                
                if not is_complete:
                    skipped_incomplete += 1
                    continue
                
                # Found a galaxy with a fully confident decision tree
                total_confident_galaxies += 1
                
                # Check if we need more galaxies for this path
                if len(path_galaxies[path_string]) < MAX_PER_UNIQUE_PATH:
                    path_galaxies[path_string].append((global_idx, obj_id, path_nodes))
                    all_selected_indices.add(global_idx)
            
            # Print progress
            print_progress(path_galaxies, chunk_num, len(path_galaxies))
            print(f"Total confident galaxies found so far: {total_confident_galaxies:,}")
            print(f"Total selected galaxies: {len(all_selected_indices):,}")
            print(f"Skipped (excluded): {skipped_excluded:,}")
            print(f"Skipped (incomplete paths): {skipped_incomplete:,}")
        
        # Final summary
        print("\n=== Final Summary ===")
        print(f"Total galaxies processed: {total_rows:,}")
        print(f"Galaxies with fully confident paths: {total_confident_galaxies:,}")
        print(f"Unique decision tree paths found: {len(path_galaxies)}")
        print(f"Total galaxies selected: {len(all_selected_indices):,}")
        print(f"Galaxies skipped (excluded): {skipped_excluded:,}")
        print(f"Galaxies skipped (incomplete paths): {skipped_incomplete:,}")
        
        # Create output HDF5
        print(f"\nWriting output to {OUTPUT_FILE}")
        sorted_indices = sorted(all_selected_indices)
        
        # Create a mapping of index to path info
        index_to_path: Dict[int, Tuple[str, List[Tuple[str, str, int, float]]]] = {}
        for path_string, galaxies in path_galaxies.items():
            for idx, obj_id, path_nodes in galaxies:
                index_to_path[idx] = (path_string, path_nodes)
        
        with h5py.File(OUTPUT_FILE, 'w') as out_f:
            # Read all selected galaxy data
            print("Reading selected galaxy data...")
            galaxy_data_list = []
            
            for idx in sorted_indices:
                row = data_table[idx]
                galaxy_data_list.append(row)
            
            # Get dtype from first galaxy
            sample_row = galaxy_data_list[0]
            
            # Create new dtype with decision tree column
            dtype_list = []
            for name in sample_row.dtype.names:
                dtype_list.append((name, sample_row[name].dtype, sample_row[name].shape if len(sample_row[name].shape) > 0 else ()))
            
            # Add decision tree column
            dtype_list.append(('decision_tree', h5py.string_dtype(encoding='utf-8')))
            
            # Create output dataset
            out_dataset = out_f.create_dataset(
                '__astropy_table__', 
                (len(sorted_indices),), 
                dtype=np.dtype(dtype_list)
            )
            
            # Copy data
            print(f"Writing {len(sorted_indices)} galaxies...")
            for i, (galaxy_idx, galaxy_row) in enumerate(zip(sorted_indices, galaxy_data_list)):
                # Create new row
                new_row = out_dataset[i]
                
                # Copy each field
                for field_name in sample_row.dtype.names:
                    try:
                        new_row[field_name] = galaxy_row[field_name]
                    except Exception as e:
                        print(f"Error copying field {field_name} for galaxy {i}: {e}")
                
                # Add full decision tree
                path_string, path_nodes = index_to_path[galaxy_idx]
                decision_tree_str = create_decision_tree_json(path_nodes)
                new_row['decision_tree'] = decision_tree_str
                
                # Write row back
                out_dataset[i] = new_row
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(sorted_indices)} galaxies...")
            
            # Force flush
            out_f.flush()
            
            # Copy attributes
            for key, value in data_table.attrs.items():
                out_dataset.attrs[key] = value
        
        print("HDF5 file creation completed!")
    
    # Create summary JSON
    summary = {
        "total_galaxies_selected": len(all_selected_indices),
        "total_unique_paths": len(path_galaxies),
        "total_confident_galaxies_found": total_confident_galaxies,
        "galaxies_skipped_excluded": skipped_excluded,
        "galaxies_skipped_incomplete": skipped_incomplete,
        "paths": {}
    }
    
    # Add path details
    for path_string, galaxies in path_galaxies.items():
        path_key = path_string.replace(" -> ", "_").replace(":", "_")
        summary["paths"][path_key] = {
            "readable_path": path_string,
            "count": len(galaxies),
            "galaxy_ids": [obj_id for _, obj_id, _ in galaxies],
            "indices": [idx for idx, _, _ in galaxies]
        }
    
    # Sort paths by count for easier reading
    summary["paths"] = dict(sorted(summary["paths"].items(), 
                                 key=lambda x: x[1]["count"], 
                                 reverse=True))
    
    print(f"\nWriting summary to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print path distribution
    print("\n=== Decision Tree Path Distribution ===")
    path_counts = sorted([(len(galaxies), path) for path, galaxies in path_galaxies.items()], reverse=True)
    
    print("Top 20 most common paths:")
    for count, path in path_counts[:20]:
        # Create a shortened version for display
        nodes = path.split(" -> ")
        answers = [node.split(":")[1] for node in nodes]
        short_desc = " -> ".join(answers)
        print(f"  {count} galaxies: {short_desc}")
    
    print(f"\nDone! Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
