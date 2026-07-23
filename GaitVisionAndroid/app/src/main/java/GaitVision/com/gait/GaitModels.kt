package GaitVision.com.gait

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

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
    const val EMA_ALPHA = 0.4f
    const val MIN_STEP_TIME_S = 0.32f
    const val MAX_STEP_TIME_S = 1.70f
    const val STEP_DISTANCE_FACTOR = 0.41f
    const val STEP_PROMINENCE_FACTOR = 0.33f
    const val MIN_STEP_PEAK_WIDTH_S = 0.15f  // Reject peaks narrower than this (seconds)
    const val VALID_FRAME_PCT = 0.66f
    const val STEP_TIME_TOLERANCE = 0.22f
    const val KNEE_ROM_MIN = 3.6f
    const val KNEE_ROM_MAX = 62.1f
    
    // Robust extrema
    const val USE_ROBUST_EXTREMA = true
    const val EXTREMA_PERCENTILE_LO = 5.0f
    const val EXTREMA_PERCENTILE_HI = 95.0f

    // Position smoothing params (One-Euro on normalized 0..1 landmarks).
    // Higher minCutoff/beta = lighter smoothing and less lag.
    const val POSITION_SMOOTH_MIN_CUTOFF = 8.0f
    const val POSITION_SMOOTH_BETA = 40.0f
    const val POSITION_SMOOTH_MAX_MISS_STREAK = 3
    
    // ROI Tracking params (mirrors PC ROITracker)
    const val ROI_MARGIN = 1.6f                    // ROI is margin x body size
    const val ROI_BOUNDS_MIN_CONFIDENCE = 0.3f     // Keypoint confidence needed to contribute to ROI bounds

    // Step-detection confidence thresholds (looser than MIN_CONFIDENCE so step
    // coverage survives brief low-confidence frames). Used by computeStepSignal
    // and computeConfidenceCoverage in FeatureExtractor.
    const val STEP_DETECT_MIN_CONF = 0.15f         // Min confidence for a single-leg ankle/knee sample
    const val STEP_DETECT_MIN_CONF_BOTH = 0.30f    // Min confidence required on BOTH legs (inter-ankle mode)
    const val ROI_EXPANDED_MARGIN = 2.2f           // Expanded margin when quality drops
    const val ROI_CENTER_EMA_ALPHA = 0.3f          // Smoothing for center movement
    const val ROI_SIZE_EXPAND_ALPHA = 0.5f         // Fast expansion
    const val ROI_SIZE_SHRINK_ALPHA = 0.1f         // Slow shrinking
    const val ROI_TARGET_SIZE = 512                // Resize ROI to this for pose
    const val ROI_ACQUIRE_STABLE_FRAMES = 10       // Frames needed to lock tracking
    const val ROI_QUALITY_WINDOW_SIZE = 15         // Rolling window for quality monitoring
    const val ROI_QUALITY_THRESHOLD = 0.3f         // Far-leg conf below this = degraded
    const val ROI_ACQUIRE_MIN_QUALITY = 0.55f      // Need real lower-body confidence before locking ROI
    const val ROI_HIGH_CONFIDENCE_BYPASS = 0.70f   // Full-frame is good enough; skip ROI probe
    const val ROI_ACCEPT_QUALITY_MARGIN = 0.02f    // ROI must beat full-frame quality by this margin
    const val ROI_LOWER_BODY_EXTRA_PAD = 0.25f     // Extra downward margin to keep feet in crop
    const val ROI_UPPER_BODY_EXTRA_PAD = 0.08f     // Small head/shoulder safety margin
    const val ROI_FAST_REACQUIRE_BAD_FRAMES = 3    // Bad leg confidence frames before full-frame reacquire
    const val ROI_DEGRADED_RATIO_THRESHOLD = 0.5f  // 50% of window degraded = expand
    const val ROI_REACQUIRE_FRAMES = 8             // Burst full-frame frames during reacquire
    const val ROI_FAIL_RATIO_TRACK = 0.30f         // >= 30% failures in TRACK -> EXPAND
    const val ROI_FAIL_RATIO_EXPAND = 0.50f        // >= 50% failures in EXPAND -> REACQUIRE
    const val ROI_CONSECUTIVE_FAIL_REACQUIRE = 3   // Consecutive failures in TRACK -> REACQUIRE
    const val ROI_MIN_DWELL_FRAMES = 5             // Min frames in state before transition
    
    // Video processing options
    const val DEFAULT_FPS = 30f                    // Fallback if FPS detection fails
    const val CLAHE_CLIP_LIMIT = 3.0f              // CLAHE contrast enhancement clip limit
    const val CLAHE_TILE_SIZE = 8                  // CLAHE tile grid size
}

/**
 * The 16 gait features extracted from pose sequences.
 * Matches PC pipeline FEATURE_COLUMNS exactly.
 */
@Parcelize
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
) : Parcelable {
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
@Parcelize
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
) : Parcelable

/**
 * Position-level pose stability metrics.
 *
 * jitterSecondDiffNorm is the robust high-frequency landmark wobble estimate:
 * median(|p[t+1] - 2p[t] + p[t-1]| / bodyScale) over confident, consecutive
 * lower-body landmark samples. Lower is better.
 *
 * meanVelocityNorm is the average per-frame landmark displacement / bodyScale.
 * It is intentionally tracked next to jitter so a filter cannot look good by
 * simply flattening real walking motion.
 */
@Parcelize
data class PoseJitterMetrics(
    val numPoseFrames: Int,
    val detectionRate: Float,
    val confidenceCoverage: Float,
    val medianBodyScale: Float,
    val jitterSecondDiffNorm: Float,
    val meanVelocityNorm: Float,
    val snapRate: Float
) : Parcelable

@Parcelize
data class PoseJitterComparison(
    val raw: PoseJitterMetrics,
    val smoothed: PoseJitterMetrics
) : Parcelable {
    val jitterReductionPct: Float
        get() = percentDrop(raw.jitterSecondDiffNorm, smoothed.jitterSecondDiffNorm)

    val velocityRetentionPct: Float
        get() = percentRatio(smoothed.meanVelocityNorm, raw.meanVelocityNorm)

    val snapReductionPct: Float
        get() = percentDrop(raw.snapRate, smoothed.snapRate)

    private fun percentDrop(before: Float, after: Float): Float {
        if (!before.isFinite() || !after.isFinite() || before <= 0f) return Float.NaN
        return ((before - after) / before) * 100f
    }

    private fun percentRatio(numerator: Float, denominator: Float): Float {
        if (!numerator.isFinite() || !denominator.isFinite() || denominator <= 0f) return Float.NaN
        return (numerator / denominator) * 100f
    }
}

/**
 * A single detected step event.
 */
@Parcelize
data class StepEvent(
    val frameIdx: Int,
    val timeS: Float
) : Parcelable

/**
 * A stride (2 consecutive steps).
 */
@Parcelize
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
) : Parcelable

/**
 * Per-frame signals computed from pose.
 */
@Parcelize
data class Signals(
    val timestamps: FloatArray,
    val frameIndices: IntArray,
    val isValid: BooleanArray,
    
    // Core signals (used in feature extraction)
    val interAnkleDist: FloatArray,
    val kneeAngleLeft: FloatArray,
    val kneeAngleRight: FloatArray,
    val trunkAngle: FloatArray,
    
    // Visualization-only angles (for charts, not used in features)
    val ankleAngleLeft: FloatArray,
    val ankleAngleRight: FloatArray,
    val hipAngleLeft: FloatArray,
    val hipAngleRight: FloatArray,
    val strideAngle: FloatArray,

    // Frontal-plane knee position diagnostics (not used in the scoring model).
    // Offset is signed perpendicular knee distance from the hip-ankle line,
    // normalized by hip-ankle length.
    val frontalKneeOffsetLeft: FloatArray,
    val frontalKneeOffsetRight: FloatArray,
    
    // Ankle positions
    val ankleLeftX: FloatArray,
    val ankleRightX: FloatArray,
    val ankleLeftY: FloatArray,
    val ankleRightY: FloatArray,

    // Knee positions
    val kneeLeftX: FloatArray,
    val kneeRightX: FloatArray,
    val kneeLeftY: FloatArray,
    val kneeRightY: FloatArray,
    
    // Hip positions
    val hipLeftY: FloatArray,
    val hipRightY: FloatArray,
    
    // Velocities (computed after smoothing)
    var ankleLeftVy: FloatArray,
    var ankleRightVy: FloatArray,
    var hipAvgVy: FloatArray
) : Parcelable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as Signals
        return timestamps.contentEquals(other.timestamps) &&
               frameIndices.contentEquals(other.frameIndices) &&
               isValid.contentEquals(other.isValid) &&
               interAnkleDist.contentEquals(other.interAnkleDist) &&
               kneeAngleLeft.contentEquals(other.kneeAngleLeft) &&
               kneeAngleRight.contentEquals(other.kneeAngleRight) &&
               trunkAngle.contentEquals(other.trunkAngle) &&
               ankleAngleLeft.contentEquals(other.ankleAngleLeft) &&
               ankleAngleRight.contentEquals(other.ankleAngleRight) &&
               hipAngleLeft.contentEquals(other.hipAngleLeft) &&
               hipAngleRight.contentEquals(other.hipAngleRight) &&
               strideAngle.contentEquals(other.strideAngle) &&
               frontalKneeOffsetLeft.contentEquals(other.frontalKneeOffsetLeft) &&
               frontalKneeOffsetRight.contentEquals(other.frontalKneeOffsetRight) &&
               ankleLeftX.contentEquals(other.ankleLeftX) &&
               ankleRightX.contentEquals(other.ankleRightX) &&
               ankleLeftY.contentEquals(other.ankleLeftY) &&
               ankleRightY.contentEquals(other.ankleRightY) &&
               kneeLeftX.contentEquals(other.kneeLeftX) &&
               kneeRightX.contentEquals(other.kneeRightX) &&
               kneeLeftY.contentEquals(other.kneeLeftY) &&
               kneeRightY.contentEquals(other.kneeRightY) &&
               hipLeftY.contentEquals(other.hipLeftY) &&
               hipRightY.contentEquals(other.hipRightY) &&
               ankleLeftVy.contentEquals(other.ankleLeftVy) &&
               ankleRightVy.contentEquals(other.ankleRightVy) &&
               hipAvgVy.contentEquals(other.hipAvgVy)
    }

    override fun hashCode(): Int {
        var result = timestamps.contentHashCode()
        result = 31 * result + frameIndices.contentHashCode()
        result = 31 * result + isValid.contentHashCode()
        result = 31 * result + interAnkleDist.contentHashCode()
        result = 31 * result + kneeAngleLeft.contentHashCode()
        result = 31 * result + kneeAngleRight.contentHashCode()
        result = 31 * result + trunkAngle.contentHashCode()
        result = 31 * result + ankleAngleLeft.contentHashCode()
        result = 31 * result + ankleAngleRight.contentHashCode()
        result = 31 * result + hipAngleLeft.contentHashCode()
        result = 31 * result + hipAngleRight.contentHashCode()
        result = 31 * result + strideAngle.contentHashCode()
        result = 31 * result + frontalKneeOffsetLeft.contentHashCode()
        result = 31 * result + frontalKneeOffsetRight.contentHashCode()
        result = 31 * result + ankleLeftX.contentHashCode()
        result = 31 * result + ankleRightX.contentHashCode()
        result = 31 * result + ankleLeftY.contentHashCode()
        result = 31 * result + ankleRightY.contentHashCode()
        result = 31 * result + kneeLeftX.contentHashCode()
        result = 31 * result + kneeRightX.contentHashCode()
        result = 31 * result + kneeLeftY.contentHashCode()
        result = 31 * result + kneeRightY.contentHashCode()
        result = 31 * result + hipLeftY.contentHashCode()
        result = 31 * result + hipRightY.contentHashCode()
        result = 31 * result + ankleLeftVy.contentHashCode()
        result = 31 * result + ankleRightVy.contentHashCode()
        result = 31 * result + hipAvgVy.contentHashCode()
        return result
    }
}

@Parcelize
data class BackGaitFeatures(
    val cadence_spm: Float,
    val hip_drop_mean: Float,
    val hip_drop_std: Float,
    val hip_sway_std: Float,
    val stride_width_mean: Float,
    val stride_width_cv: Float,
    val step_width_asymmetry: Float,
    val knee_separation_mean: Float,
    val trunk_lean_mean: Float,
    val trunk_lean_std: Float,
    val valid_stride_count: Int
) : Parcelable {
    fun toFeatureArray(): FloatArray {
        return floatArrayOf(
            cadence_spm, hip_drop_mean, hip_drop_std, hip_sway_std,
            stride_width_mean, stride_width_cv, step_width_asymmetry,
            knee_separation_mean, trunk_lean_mean, trunk_lean_std
        )
    }
}