package GaitVision.com

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import GaitVision.com.data.PreferencesManager
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.PoseJitterComparison
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.Signals
import GaitVision.com.gait.Stride
import GaitVision.com.mediapipe.PoseFrame

/**
 * Holds all state for a single analysis session.
 * Call reset() when returning to the Dashboard or starting a fresh analysis.
 */
object AnalysisSession {

    // URIs and raw frame data
    var galleryUri: Uri? = null
    var editedUri: Uri? = null
    val frameList: MutableList<Bitmap> = mutableListOf()
    val rawPoseFrames: MutableList<PoseFrame> = mutableListOf()
    val poseFrames: MutableList<PoseFrame> = mutableListOf()

    var lastPoseFramesSnapshot: List<PoseFrame> = emptyList()

    // Extracted gait features (16 features matching PC pipeline)
    var extractedFeatures: GaitFeatures? = null
    var extractionDiagnostics: GaitDiagnostics? = null
    var scoringResult: ScoringResult? = null
    var jitterComparison: PoseJitterComparison? = null

    // Signals for visualization (populated during feature extraction)
    var extractedSignals: Signals? = null
    var extractedStrides: List<Stride>? = null
    var selectedStrideIndices: List<Int>? = null  // Indices of the 2 strides used for features
    var stepSignalMode: String? = null

    // Participant info set before analysis begins
    var participantId: Int = 0
    var participantHeight: Int = 0

    // Database IDs for the current session
    var currentPatientId: Int? = null
    var currentResultId: Long? = null

    // Duration of the processed video in microseconds
    var videoLength: Long = 0

    // User selected recording date
    var recordingDate: Long = System.currentTimeMillis()

    /** Clear all session state. */
    fun reset() {
        galleryUri = null
        editedUri = null
        frameList.clear()
        rawPoseFrames.clear()
        poseFrames.clear()
        extractedFeatures = null
        extractionDiagnostics = null
        scoringResult = null
        jitterComparison = null
        extractedSignals = null
        extractedStrides = null
        selectedStrideIndices = null
        stepSignalMode = null
        participantId = 0
        participantHeight = 0
        currentPatientId = null
        currentResultId = null
        videoLength = 0
        recordingDate = System.currentTimeMillis()
    }
}

// Video processing options — app-level config, not per-session state.
var enableCLAHE: Boolean = false          // CLAHE disabled - testing without for parity comparison
var enableROIRetry: Boolean = false       // EXPERIMENTAL/OFF - ROI retry non-functional in fast path
var enablePositionSmoothing: Boolean = false // One-Euro smoothing for high-confidence landmarks
var forceCpuInference: Boolean = false    // GPU delegate for ~2-3x speedup (falls back to CPU automatically)

// Debug/logging options
var enableVerboseLogging: Boolean = false  // Toggle heavy per-frame logging in FeatureExtractor

fun loadAnalysisOptions(context: Context) {
    val preferences = PreferencesManager(context.applicationContext)
    enablePositionSmoothing = preferences.positionSmoothingEnabled
}
