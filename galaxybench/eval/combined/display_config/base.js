// Function to create generic comparison displays
function createGenericComparison(judgeResults, galaxyData, score, judging) {
    const scoreClass = score >= 0.8 ? 'excellent' : score >= 0.6 ? 'good' : score >= 0.4 ? 'fair' : 'poor';
    
    return `
        <div class="decision-tree-section">
            <div class="decision-tree-header">
                <div class="decision-tree-title">
                    <i class="fas fa-chart-line"></i>
                    Analysis Results
                </div>
                <div class="decision-tree-score ${scoreClass}">
                    <i class="fas fa-trophy"></i>
                    ${(score * 100).toFixed(1)}%
                </div>
            </div>
            <div style="padding: 1rem; color: #666;">
                <p>Score: ${(score * 100).toFixed(1)}%</p>
                <p style="font-size: 0.9em; margin-top: 0.5rem;">
                    Judge Model: ${judging.judge_model || 'Unknown'}
                </p>
            </div>
        </div>
    `;
}

// Function to create comparison for any eval type
function createAllEvalComparison(judgeResults, galaxyData, score, judging) {
    // Detect eval type from the data
    if (galaxyData.decision_tree) {
        // Galaxy Zoo type - delegate to Galaxy Zoo function
        return createDecisionTreeComparison(judgeResults, galaxyData, score, judging);
    } else if (galaxyData.tidal_info) {
        // Tidal type - delegate to Tidal function  
        return createTidalFeatureComparison(judgeResults, galaxyData, score, judging);
    } else {
        // Unknown type - use generic comparison
        return createGenericComparison(judgeResults, galaxyData, score, judging);
    }
}