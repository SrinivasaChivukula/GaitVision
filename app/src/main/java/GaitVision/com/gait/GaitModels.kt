package GaitVision.com.gait

/**
 * Gait analysis data models - mirrors PC pipeline for feature parity.
 */

/**
 * Configuration for feature extraction (OPTIMAL_CONFIG from PC).
 */
object GaitConfig {
    // Pose backend params (used by MediaPipePoseBackend)
    const val MIN_DETECTION_CONFIDENCE = 0.40f
    const val MIN_TRACKING_CONFIDENCE = 0.61f
    const val MIN_PRESENCE_CONFIDENCE = 0.5f
    
    // Feature extractor params
    const val MIN_CONFIDENCE = 0.32f
    const val MAX_INTERP_GAP = 5
    const val EMA_ALPHA = 0.68f
    const val MIN_STEP_TIME_S = 0.32f
    const val MAX_STEP_TIME_S = 1.70f
    const val STEP_DISTANCE_FACTOR = 0.41f
    const val STEP_PROMINENCE_FACTOR = 0.33f
    const val VALID_FRAME_PCT = 0.66f
    const val STEP_TIME_TOLERANCE = 0.22f
    const val KNEE_ROM_MIN = 3.6f
    const val KNEE_ROM_MAX = 62.1f
    
    // Robust extrema
    const val USE_ROBUST_EXTREMA = true
    const val EXTREMA_PERCENTILE_LO = 5.0f
    const val EXTREMA_PERCENTILE_HI = 95.0f
    
    // ROI Tracking params (mirrors PC ROITracker)
    const val ROI_MARGIN = 1.4f                    // ROI is margin x body size
    const val ROI_EXPANDED_MARGIN = 1.8f           // Expanded margin when quality drops
    const val ROI_CENTER_EMA_ALPHA = 0.3f          // Smoothing for center movement
    const val ROI_SIZE_EXPAND_ALPHA = 0.5f         // Fast expansion
    const val ROI_SIZE_SHRINK_ALPHA = 0.1f         // Slow shrinking
    const val ROI_TARGET_SIZE = 512                // Resize ROI to this for pose
    const val ROI_ACQUIRE_STABLE_FRAMES = 10       // Frames needed to lock tracking
    const val ROI_QUALITY_WINDOW_SIZE = 15         // Rolling window for quality monitoring
    const val ROI_QUALITY_THRESHOLD = 0.3f         // Far-leg conf below this = degraded
    const val ROI_DEGRADED_RATIO_THRESHOLD = 0.5f  // 50% of window degraded = expand
    const val ROI_REACQUIRE_FRAMES = 15            // Burst full-frame frames during reacquire
    const val ROI_FAIL_RATIO_TRACK = 0.30f         // >= 30% failures in TRACK -> EXPAND
    const val ROI_FAIL_RATIO_EXPAND = 0.50f        // >= 50% failures in EXPAND -> REACQUIRE
    const val ROI_CONSECUTIVE_FAIL_REACQUIRE = 5   // Consecutive failures in TRACK -> REACQUIRE
    const val ROI_MIN_DWELL_FRAMES = 10            // Min frames in state before transition
    
    // Video processing options
    const val DEFAULT_FPS = 30f                    // Fallback if FPS detection fails
    const val CLAHE_CLIP_LIMIT = 3.0f              // CLAHE contrast enhancement clip limit
    const val CLAHE_TILE_SIZE = 8                  // CLAHE tile grid size
}

/**
 * The 16 gait features extracted from pose sequences.
 * Matches PC pipeline FEATURE_COLUMNS exactly.
 */
data class GaitFeatures(
    // Temporal (4)
    val cadence_spm: Float,
    val stride_time_s: Float,
    val stride_time_cv: Float,
    val step_time_asymmetry: Float,
    
    // Spatial/Kinematic (7)
    val stride_length_norm: Float,
    val stride_amp_norm: Float,
    val step_length_asymmetry: Float,
    val knee_left_rom: Float,
    val knee_right_rom: Float,
    val knee_left_max: Float,
    val knee_right_max: Float,
    
    // Smoothness/Jerk (3)
    val ldj_knee_left: Float,
    val ldj_knee_right: Float,
    val ldj_hip: Float,
    
    // Balance/Stability (2)
    val trunk_lean_std_deg: Float,
    val inter_ankle_cv: Float,
    
    val valid_stride_count: Int
) {
    /**
     * Get features as array in FEATURE_COLUMNS order for scoring models.
     */
    fun toFeatureArray(): FloatArray {
        return floatArrayOf(
            cadence_spm,
            stride_time_s,
            stride_time_cv,
            step_time_asymmetry,
            stride_length_norm,
            stride_amp_norm,
            step_length_asymmetry,
            knee_left_rom,
            knee_right_rom,
            knee_left_max,
            knee_right_max,
            ldj_knee_left,
            ldj_knee_right,
            ldj_hip,
            trunk_lean_std_deg,
            inter_ankle_cv
        )
    }
    
    companion object {
        val FEATURE_COLUMNS = listOf(
            "cadence_spm",
            "stride_time_s",
            "stride_time_cv",
            "step_time_asymmetry",
            "stride_length_norm",
            "stride_amp_norm",
            "step_length_asymmetry",
            "knee_left_rom",
            "knee_right_rom",
            "knee_left_max",
            "knee_right_max",
            "ldj_knee_left",
            "ldj_knee_right",
            "ldj_hip",
            "trunk_lean_std_deg",
            "inter_ankle_cv"
        )
        
        /**
         * Create empty/invalid features.
         */
        fun empty() = GaitFeatures(
            cadence_spm = Float.NaN,
            stride_time_s = Float.NaN,
            stride_time_cv = Float.NaN,
            step_time_asymmetry = Float.NaN,
            stride_length_norm = Float.NaN,
            stride_amp_norm = Float.NaN,
            step_length_asymmetry = Float.NaN,
            knee_left_rom = Float.NaN,
            knee_right_rom = Float.NaN,
            knee_left_max = Float.NaN,
            knee_right_max = Float.NaN,
            ldj_knee_left = Float.NaN,
            ldj_knee_right = Float.NaN,
            ldj_hip = Float.NaN,
            trunk_lean_std_deg = Float.NaN,
            inter_ankle_cv = Float.NaN,
            valid_stride_count = 0
        )
    }
}

/**
 * Quality flag for extraction result.
 */
enum class QualityFlag {
    OK,
    LOW_DETECTION,
    NO_CYCLES,
    UNPROCESSABLE
}

/**
 * Diagnostic information about extraction.
 */
data class GaitDiagnostics(
    val videoId: String,
    val fpsDetected: Float,
    val durationS: Float,
    val numFramesTotal: Int,
    val numFramesValid: Int,
    val validFrameRate: Float,
    val numStepsDetected: Int,
    val numStridesValid: Int,
    val estimatedCadenceSpm: Float,
    val walkingDirection: String,
    val wasFlipped: Boolean,
    val qualityFlag: QualityFlag,
    val rejectionReasons: List<String> = emptyList()
)

/**
 * A single detected step event.
 */
data class StepEvent(
    val frameIdx: Int,
    val timeS: Float
)

/**
 * A stride (2 consecutive steps).
 */
data class Stride(
    val startFrame: Int,
    val endFrame: Int,
    val startTimeS: Float,
    val endTimeS: Float,
    val step1Frame: Int,
    val step2Frame: Int,
    val step1TimeS: Float,
    val step2TimeS: Float,
    val isValid: Boolean = true,
    val invalidReason: String? = null,
    val kneeRomLeft: Float = 0f,
    val kneeRomRight: Float = 0f,
    val kneeMaxLeft: Float = 0f,
    val kneeMaxRight: Float = 0f,
    val validFramePct: Float = 0f,
    val qualityScore: Float = 0f
)

/**
 * Per-frame signals computed from pose.
 */
data class Signals(
    val timestamps: FloatArray,
    val frameIndices: IntArray,
    val isValid: BooleanArray,
    
    // Core signals
    val interAnkleDist: FloatArray,
    val kneeAngleLeft: FloatArray,
    val kneeAngleRight: FloatArray,
    val trunkAngle: FloatArray,
    
    // Ankle positions
    val ankleLeftX: FloatArray,
    val ankleRightX: FloatArray,
    val ankleLeftY: FloatArray,
    val ankleRightY: FloatArray,
    
    // Hip positions
    val hipLeftY: FloatArray,
    val hipRightY: FloatArray,
    
    // Velocities (computed after smoothing)
    var ankleLeftVy: FloatArray,
    var ankleRightVy: FloatArray,
    var hipAvgVy: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as Signals
        return timestamps.contentEquals(other.timestamps)
    }

    override fun hashCode(): Int = timestamps.contentHashCode()
}
