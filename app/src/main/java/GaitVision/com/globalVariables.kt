package GaitVision.com

import android.net.Uri
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.Signals
import GaitVision.com.gait.Stride
import GaitVision.com.mediapipe.PoseFrame

//Any URI or frames used in application
var galleryUri : Uri? = null
var editedUri : Uri? = null

// Pose frames for feature extraction (PC pipeline compatibility)
var poseFrames: MutableList<PoseFrame> = mutableListOf()

// Extracted gait features (16 features matching PC pipeline)
var extractedFeatures: GaitFeatures? = null
var extractionDiagnostics: GaitDiagnostics? = null
var scoringResult: ScoringResult? = null

// Signals for visualization (populated during feature extraction)
var extractedSignals: Signals? = null
var extractedStrides: List<Stride>? = null
var selectedStrideIndices: List<Int>? = null  // Indices of the 2 strides used for features
var stepSignalMode: String? = null

//User input for ID and height
var participantId: Int = 0
var participantHeight: Int = 0

//Database IDs for current session
var currentPatientId: Int? = null
var currentResultId: Long? = null


//Variable for keeping track of video length we processed on
var videoLength: Long = 0

// Video processing options (mirrors PC pipeline options)
var enableCLAHE: Boolean = false  // CLAHE disabled - testing without for parity comparison
var forceCpuInference: Boolean = false  // GPU delegate for ~2-3x speedup (falls back to CPU automatically if GPU fails)

// Debug/logging options
var enableVerboseLogging: Boolean = false  // Toggle heavy per-frame logging in FeatureExtractor

/** Reset all session state. Call when returning to Dashboard or starting a fresh analysis. WE NEED TO REFACTOR GLOBALVARS SYSTEM */
fun resetAnalysisState() {
    galleryUri = null
    editedUri = null
    poseFrames.clear()
    extractedFeatures = null
    extractionDiagnostics = null
    scoringResult = null
    extractedSignals = null
    extractedStrides = null
    selectedStrideIndices = null
    stepSignalMode = null
    participantId = 0
    participantHeight = 0
    currentPatientId = null
    currentResultId = null
    videoLength = 0
}