package GaitVision.com

// Video processing options (mirrors PC pipeline options)
var enableCLAHE: Boolean = false  // CLAHE disabled - testing without for parity comparison
var forceCpuInference: Boolean = false  // GPU delegate for ~2-3x speedup (falls back to CPU automatically if GPU fails)

// Debug/logging options
var enableVerboseLogging: Boolean = false  // Toggle heavy per-frame logging in FeatureExtractor

/** Reset all session state. Call when returning to Dashboard or starting a fresh analysis. */
fun resetAnalysisState() {
    AnalysisSession.clear()
}
