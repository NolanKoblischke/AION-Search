// Function to format path step text for Galaxy Zoo
function formatPathStep(step) {
    if (step.endsWith('-not-mentioned')) {
        const fieldName = step.replace('-not-mentioned', '').replace(/-/g, ' ');
        return fieldName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') + ' Not Mentioned';
    }
    const formatted = step.replace(/-/g, ' ').replace(/_/g, ': ');
    return formatted.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// Function to get step class for styling Galaxy Zoo steps
function getStepClass(step, otherPath) {
    // Check for exact match first
    if (otherPath.includes(step)) {
        return 'match';
    }
    
    // Check for semantic equivalence
    for (const otherStep of otherPath) {
        if (areStepsEquivalent(step, otherStep)) {
            return 'match';
        }
    }
    
    return 'mismatch';
}

// Function to check if two steps are semantically equivalent
function areStepsEquivalent(step1, step2) {
    return step1.toLowerCase() === step2.toLowerCase();
}

// Function to create Galaxy Zoo decision tree comparison
function createDecisionTreeComparison(judgeResults, galaxyData, score, judging) {
    // In the unified system, paths are stored in judge_results
    const judgePath = (judging.judge_results && judging.judge_results.judge_path) || judgeResults.judge_path || judging.judge_path || [];
    const volunteerPath = (judging.judge_results && judging.judge_results.volunteer_path) || judgeResults.volunteer_path || judging.volunteer_path || [];
    
    // Debug: Log the actual paths being compared
    console.log('=== GALAXY ZOO PATH COMPARISON DEBUG ===');
    console.log('judgeResults:', judgeResults);
    console.log('judging object:', judging);
    console.log('judging.judge_results:', judging.judge_results);
    console.log('Judge path:', judgePath);
    console.log('Volunteer path:', volunteerPath);
    console.log('Score:', score);
    console.log('==============================');
    
    const maxLength = Math.max(judgePath.length, volunteerPath.length);
    const scoreClass = score >= 0.8 ? 'excellent' : score >= 0.6 ? 'good' : score >= 0.4 ? 'fair' : 'poor';
    
    let judgeStepsHtml = '';
    let volunteerStepsHtml = '';
    
    for (let i = 0; i < maxLength; i++) {
        const judgeStep = judgePath[i];
        const volunteerStep = volunteerPath[i];
        
        if (judgeStep) {
            const stepClass = getStepClass(judgeStep, volunteerPath);
            judgeStepsHtml += `<div class="path-step ${stepClass}" data-step="${i + 1}">${formatPathStep(judgeStep)}</div>`;
        }
        
        if (volunteerStep) {
            const stepClass = getStepClass(volunteerStep, judgePath);
            volunteerStepsHtml += `<div class="path-step ${stepClass}" data-step="${i + 1}">${formatPathStep(volunteerStep)}</div>`;
        }
    }
    
    return `
        <div class="decision-tree-section">
            <div class="decision-tree-header">
                <div class="decision-tree-title">
                    <i class="fas fa-sitemap"></i>
                    Decision Tree Comparison
                </div>
                <div class="decision-tree-score ${scoreClass}">
                    <i class="fas fa-trophy"></i>
                    ${(score * 100).toFixed(1)}%
                </div>
            </div>
            
            <div class="decision-tree-comparison">
                <div class="decision-path judge">
                    <div class="decision-path-title">
                        <i class="fas fa-gavel"></i>
                        AI Judge Path
                    </div>
                    <div class="path-steps">
                        ${judgeStepsHtml || '<div class="path-step neutral">No path recorded</div>'}
                    </div>
                </div>
                
                <div class="decision-path volunteer">
                    <div class="decision-path-title">
                        <i class="fas fa-users"></i>
                        Volunteer Consensus Path
                    </div>
                    <div class="path-steps">
                        ${volunteerStepsHtml || '<div class="path-step neutral">No path recorded</div>'}
                    </div>
                </div>
            </div>
        </div>
    `;
}