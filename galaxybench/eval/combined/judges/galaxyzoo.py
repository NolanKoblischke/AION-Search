"""Galaxy Zoo judge implementation."""

import os
import json
from typing import Dict, Any, Tuple, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .base import BaseJudge

load_dotenv()


class GalaxyDecisionTree(BaseModel):
    """Complete decision tree for galaxy classification, following the same structure as volunteer decisions."""
    
    # Root decision: What is the overall shape?
    overall_shape: Literal["smooth", "featured-or-disk", "artifact"] = Field(
        ..., description="The overall galaxy shape classification"
    )
    
    # For smooth galaxies: How rounded is it?
    roundness: Optional[Literal["round", "in-between", "cigar-shaped", "not-mentioned"]] = Field(
        None, description="How rounded the galaxy is (only for smooth galaxies)"
    )
    
    # For featured galaxies: Is the galaxy edge-on?
    edge_on: Optional[Literal["edge-on-yes", "edge-on-no", "not-mentioned"]] = Field(
        None, description="Whether the galaxy is viewed edge-on (only for featured galaxies)"
    )
    
    # For edge-on galaxies: What is the bulge shape?
    edge_on_bulge: Optional[Literal["boxy", "none", "rounded", "not-mentioned"]] = Field(
        None, description="Shape of the bulge for edge-on galaxies"
    )
    
    # For non-edge-on featured galaxies: Does it have spiral arms?
    has_spiral_arms: Optional[Literal["has-spiral-arms-yes", "has-spiral-arms-no", "not-mentioned"]] = Field(
        None, description="Whether the galaxy has visible spiral arms (only for non-edge-on featured galaxies)"
    )
    
    # For spiral galaxies: How tightly wound are the arms?
    spiral_winding: Optional[Literal["tight", "medium", "loose", "not-mentioned"]] = Field(
        None, description="How tightly wound the spiral arms are (only for galaxies with spiral arms)"
    )
    
    # For spiral galaxies: How many arms?
    spiral_arm_count: Optional[Literal["1", "2", "3", "4", "more-than-4", "cant-tell", "not-mentioned"]] = Field(
        None, description="Number of spiral arms (only for galaxies with spiral arms)"
    )
    
    # For non-edge-on featured galaxies: Is there a bar?
    bar: Optional[Literal["strong", "weak", "no", "not-mentioned"]] = Field(
        None, description="Strength of central bar feature (only for non-edge-on featured galaxies)"
    )
    
    # For non-edge-on featured galaxies: How prominent is the bulge?
    bulge_size: Optional[Literal["dominant", "large", "moderate", "small", "none", "not-mentioned"]] = Field(
        None, description="Size of the central bulge (only for non-edge-on featured galaxies)"
    )
    
    # For smooth and featured galaxies (NOT artifacts): Signs of disturbance or merging?
    merging: Optional[Literal["none", "minor-disturbance", "major-disturbance", "merger", "not-mentioned"]] = Field(
        None, description="Signs of disturbance, interaction, or merging (only for smooth and featured galaxies, NOT artifacts)"
    )


class GalaxyZooJudge(BaseJudge):
    """Judge for Galaxy Zoo morphology classification."""
    
    def get_score_field_name(self) -> str:
        return "decision_tree_score"
    
    def judge_description(self, description: str, galaxy_info: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Judge a galaxy description using the decision tree approach."""
        # Check if we have decision tree data
        if 'decision_tree' not in galaxy_info or not galaxy_info['decision_tree']:
            raise ValueError("No decision tree data available for galaxy")
        
        # Extract volunteer path
        volunteer_path = self._extract_volunteer_path(galaxy_info['decision_tree'])
        
        # Get judge's decision tree
        prompt = self._build_prompt(description)
        tree_data = self._get_structured_output(prompt, GalaxyDecisionTree)
        
        # Handle both dict and Pydantic model responses
        if isinstance(tree_data, dict):
            tree = GalaxyDecisionTree(**tree_data)
        else:
            tree = tree_data
        
        # Convert to path and calculate score
        judge_path = self._build_decision_tree_path(tree)
        score = self._calculate_score(judge_path, volunteer_path)
        
        # Prepare results
        results = {
            'judge_decision_tree': tree_data if isinstance(tree_data, dict) else tree_data.dict(),
            'judge_path': judge_path,
            'volunteer_path': volunteer_path,
            'judge_prompt': prompt,
        }
        
        return results, score
    
    def _extract_volunteer_path(self, decision_tree_data: List[Dict]) -> List[str]:
        """Extract a decision path from the structured decision tree data."""
        if not decision_tree_data:
            return []
        
        path = []
        for node_info in decision_tree_data:
            if isinstance(node_info, dict) and 'node' in node_info:
                path.append(node_info['node'])
            elif isinstance(node_info, dict) and 'question' in node_info and 'answer' in node_info:
                path.append(f"{node_info['question']}_{node_info['answer']}")
        
        return path
    
    def _build_decision_tree_path(self, tree: GalaxyDecisionTree) -> List[str]:
        """Convert a GalaxyDecisionTree into a path list."""
        path = [f"smooth-or-featured_{tree.overall_shape}"]
        
        if tree.overall_shape == "smooth":
            if tree.roundness:
                if tree.roundness == "not-mentioned":
                    path.append("how-rounded-not-mentioned")
                else:
                    path.append(f"how-rounded_{tree.roundness}")
        elif tree.overall_shape == "featured-or-disk":
            if tree.edge_on:
                if tree.edge_on == "not-mentioned":
                    path.append("disk-edge-on-not-mentioned")
                elif tree.edge_on == "edge-on-yes":
                    path.append("disk-edge-on_yes")
                    if tree.edge_on_bulge:
                        if tree.edge_on_bulge == "not-mentioned":
                            path.append("edge-on-bulge-not-mentioned")
                        else:
                            path.append(f"edge-on-bulge_{tree.edge_on_bulge}")
                elif tree.edge_on == "edge-on-no":
                    path.append("disk-edge-on_no")
                    if tree.has_spiral_arms:
                        if tree.has_spiral_arms == "not-mentioned":
                            path.append("has-spiral-arms-not-mentioned")
                        elif tree.has_spiral_arms == "has-spiral-arms-yes":
                            path.append("has-spiral-arms_yes")
                            if tree.spiral_winding:
                                if tree.spiral_winding == "not-mentioned":
                                    path.append("spiral-winding-not-mentioned")
                                else:
                                    path.append(f"spiral-winding_{tree.spiral_winding}")
                            if tree.spiral_arm_count:
                                if tree.spiral_arm_count == "not-mentioned":
                                    path.append("spiral-arm-count-not-mentioned")
                                else:
                                    path.append(f"spiral-arm-count_{tree.spiral_arm_count}")
                        elif tree.has_spiral_arms == "has-spiral-arms-no":
                            path.append("has-spiral-arms_no")
                    
                    # For all non-edge-on galaxies, check bar and bulge
                    if tree.bar:
                        if tree.bar == "not-mentioned":
                            path.append("bar-not-mentioned")
                        else:
                            path.append(f"bar_{tree.bar}")
                    if tree.bulge_size:
                        if tree.bulge_size == "not-mentioned":
                            path.append("bulge-size-not-mentioned")
                        else:
                            path.append(f"bulge-size_{tree.bulge_size}")
        
        # Only smooth and featured galaxies have merging status
        if tree.overall_shape != "artifact" and tree.merging:
            if tree.merging == "not-mentioned":
                path.append("merging-not-mentioned")
            else:
                path.append(f"merging_{tree.merging}")
        
        return path
    
    def _calculate_score(self, judge_path: List[str], volunteer_path: List[str]) -> float:
        """Calculate score by comparing decision tree paths."""
        if not volunteer_path:
            return 0.0
        
        judge_set = set(judge_path)
        volunteer_set = set(volunteer_path)
        matches = len(judge_set.intersection(volunteer_set))
        
        return matches / len(volunteer_path)
    
    def _build_prompt(self, description: str) -> str:
        """Build the prompt for galaxy decision tree classification."""
        prompt = """You are an expert astronomer who will classify a galaxy by following the same decision tree structure used by citizen scientists in Galaxy Zoo.

You will be given a galaxy description and must make decisions following this exact decision tree structure:

**STEP 1: Overall Shape (Required)**
- smooth: Galaxy appears smooth and featureless with no obvious structure
- featured-or-disk: Galaxy shows clear structure like spiral arms, bars, or disk features  
- artifact: Image appears to be a star, artifact, or defect rather than a galaxy

**STEP 2: Follow the appropriate branch based on Step 1**

**If SMOOTH:**
- Determine roundness: round, in-between, cigar-shaped, or not-mentioned

**If FEATURED-OR-DISK:**
- Determine if edge-on: edge-on-yes, edge-on-no, or not-mentioned

**If edge-on-yes:**
- Determine bulge shape: boxy, none, rounded, or not-mentioned

**If edge-on-no:**
- Determine if has spiral arms: has-spiral-arms-yes, has-spiral-arms-no, or not-mentioned
- If has spiral arms:
  - Determine spiral winding: tight, medium, loose, or not-mentioned
  - Determine arm count: 1, 2, 3, 4, more-than-4, cant-tell, or not-mentioned
- For all non-edge-on galaxies (regardless of spiral arms):
  - Determine bar strength: strong, weak, no, or not-mentioned
  - Determine bulge size: dominant, large, moderate, small, none, or not-mentioned

**STEP 3: Merging Status (ONLY for smooth and featured-or-disk galaxies, NOT artifacts)**
- none: No signs of disturbance or interaction
- minor-disturbance: Small distortions or asymmetries
- major-disturbance: Significant warping, tidal features, or distortions
- merger: Clear signs of ongoing merger or recent collision
- not-mentioned: The description does not mention anything about merging or disturbance

**IMPORTANT INSTRUCTIONS:**
- Base your decisions ONLY on what you can observe in the description
- **Use "not-mentioned" for any field where the description does not provide sufficient information to make a determination**
- Follow the decision tree logic - only fill out fields that are relevant for your path
- Be definitive in your choices when information is available
- **Merging status is ONLY asked for smooth and featured-or-disk galaxies, NOT for artifacts**
- If the description is unclear about a specific feature, choose "not-mentioned" rather than guessing

**GALAXY DESCRIPTION:**
{description}

Classify this galaxy by following the decision tree structure above."""
        
        return prompt.format(description=description)
    
    def format_judge_results_for_display(self, judge_results: Dict[str, Any], galaxy_info: Dict[str, Any]) -> str:
        """Format judge results as HTML for display."""
        judge_path = judge_results.get('judge_path', [])
        volunteer_path = judge_results.get('volunteer_path', [])
        score = galaxy_info.get(self.get_score_field_name(), 0)
        
        # Get score class based on score
        score_class = "poor"
        if score >= 0.8:
            score_class = "excellent"
        elif score >= 0.6:
            score_class = "good"
        elif score >= 0.4:
            score_class = "fair"
        
        max_length = max(len(judge_path), len(volunteer_path))
        
        judge_steps_html = ''
        volunteer_steps_html = ''
        
        for i in range(max_length):
            judge_step = judge_path[i] if i < len(judge_path) else None
            volunteer_step = volunteer_path[i] if i < len(volunteer_path) else None
            
            if judge_step:
                step_class = self._get_step_class(judge_step, volunteer_path)
                judge_steps_html += f'<div class="path-step {step_class}" data-step="{i + 1}">{self._format_path_step(judge_step)}</div>'
            
            if volunteer_step:
                step_class = self._get_step_class(volunteer_step, judge_path)
                volunteer_steps_html += f'<div class="path-step {step_class}" data-step="{i + 1}">{self._format_path_step(volunteer_step)}</div>'
        
        return f"""
            <div class="decision-tree-section">
                <div class="decision-tree-header">
                    <div class="decision-tree-title">
                        <i class="fas fa-sitemap"></i>
                        Decision Tree Comparison
                    </div>
                    <div class="decision-tree-score {score_class}">
                        <i class="fas fa-trophy"></i>
                        {(score * 100):.1f}%
                    </div>
                </div>
                
                <div class="decision-tree-comparison">
                    <div class="decision-path judge">
                        <div class="decision-path-title">
                            <i class="fas fa-gavel"></i>
                            AI Judge Path
                        </div>
                        <div class="path-steps">
                            {judge_steps_html or '<div class="path-step neutral">No path recorded</div>'}
                        </div>
                    </div>
                    
                    <div class="decision-path volunteer">
                        <div class="decision-path-title">
                            <i class="fas fa-users"></i>
                            Volunteer Consensus Path
                        </div>
                        <div class="path-steps">
                            {volunteer_steps_html or '<div class="path-step neutral">No path recorded</div>'}
                        </div>
                    </div>
                </div>
            </div>
        """
    
    def _format_path_step(self, step: str) -> str:
        """Format path step text."""
        if step.endswith('-not-mentioned'):
            field_name = step.replace('-not-mentioned', '').replace('-', ' ').title()
            return f"{field_name} Not Mentioned"
        return step.replace('-', ' ').replace('_', ': ').title()
    
    def _get_step_class(self, step: str, other_path: List[str]) -> str:
        """Get CSS class for step based on matching."""
        if step in other_path:
            return 'match'
        # Check for semantic equivalence
        for other_step in other_path:
            if step.lower() == other_step.lower():
                return 'match'
        return 'mismatch' 