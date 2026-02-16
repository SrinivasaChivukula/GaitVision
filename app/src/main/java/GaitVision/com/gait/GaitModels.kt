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
    
    // Robust extrema
    const val USE_ROBUST_EXTREMA = true
    const val EXTREMA_PERCENTILE_LO = 5.0f
    const val EXTREMA_PERCENTILE_HI = 95.0f
    
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
    
    // V2: Camera-robust distance features (6)
    val stride_length_relL_norm: Float = 0f,
    val stride_length_relR_norm: Float = 0f,
    val stride_length_rel_asym: Float = 0f,
    val ankle_ap_range_rel_norm: Float = 0f,
    val midHip_ap_drift_norm: Float = 0f,
    
    // V2: Timing-of-extrema (6)
    val t_knee_left_peak_pct: Float = 0f,
    val t_knee_right_peak_pct: Float = 0f,
    val t_trunk_peak_abs_pct: Float = 0f,
    val t_toe_clearance_left_pct: Float = 0f,
    val t_toe_clearance_right_pct: Float = 0f,
    
    // V2: Foot clearance + pitch (8)
    val toe_clearance_left_max: Float = 0f,
    val toe_clearance_right_max: Float = 0f,
    val toe_clearance_left_range: Float = 0f,
    val toe_clearance_right_range: Float = 0f,
    val foot_pitch_left_mean: Float = 0f,
    val foot_pitch_right_mean: Float = 0f,
    val foot_pitch_left_range: Float = 0f,
    val foot_pitch_right_range: Float = 0f,
    
    // V2: Trunk enrichments (4)
    val trunk_abs_mean_deg: Float = 0f,
    val trunk_abs_p95_deg: Float = 0f,
    val trunk_ang_vel_mean_abs: Float = 0f,
    val trunk_ang_vel_p95_abs: Float = 0f,
    
    // V2: Dual aggregation — feat_diff = abs(cycle_A − cycle_B) for per-cycle features
    val cadence_diff: Float = 0f,
    val stride_time_diff: Float = 0f,
    val step_time_asymmetry_diff: Float = 0f,
    val stride_length_norm_diff: Float = 0f,
    val stride_amp_norm_diff: Float = 0f,
    val step_length_asymmetry_diff: Float = 0f,
    val knee_left_rom_diff: Float = 0f,
    val knee_right_rom_diff: Float = 0f,
    val knee_left_max_diff: Float = 0f,
    val knee_right_max_diff: Float = 0f,
    val ldj_knee_left_diff: Float = 0f,
    val ldj_knee_right_diff: Float = 0f,
    val ldj_hip_diff: Float = 0f,
    val stride_length_relL_norm_diff: Float = 0f,
    val stride_length_relR_norm_diff: Float = 0f,
    val stride_length_rel_asym_diff: Float = 0f,
    val ankle_ap_range_rel_norm_diff: Float = 0f,
    val midHip_ap_drift_norm_diff: Float = 0f,
    val t_knee_left_peak_pct_diff: Float = 0f,
    val t_knee_right_peak_pct_diff: Float = 0f,
    val t_trunk_peak_abs_pct_diff: Float = 0f,
    val t_toe_clearance_left_pct_diff: Float = 0f,
    val t_toe_clearance_right_pct_diff: Float = 0f,
    val toe_clearance_left_max_diff: Float = 0f,
    val toe_clearance_right_max_diff: Float = 0f,
    val toe_clearance_left_range_diff: Float = 0f,
    val toe_clearance_right_range_diff: Float = 0f,
    val foot_pitch_left_mean_diff: Float = 0f,
    val foot_pitch_right_mean_diff: Float = 0f,
    val foot_pitch_left_range_diff: Float = 0f,
    val foot_pitch_right_range_diff: Float = 0f,
    
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

    /** Get all 69 features in CSV_FEATURE_COLUMNS order for export. */
    fun toCsvFeatureArray(): FloatArray {
        return floatArrayOf(
            cadence_spm, stride_time_s, stride_time_cv, step_time_asymmetry,
            stride_length_norm, stride_amp_norm, step_length_asymmetry,
            knee_left_rom, knee_right_rom, knee_left_max, knee_right_max,
            ldj_knee_left, ldj_knee_right, ldj_hip, trunk_lean_std_deg, inter_ankle_cv,
            stride_length_relL_norm, stride_length_relR_norm, stride_length_rel_asym,
            ankle_ap_range_rel_norm, midHip_ap_drift_norm,
            t_knee_left_peak_pct, t_knee_right_peak_pct, t_trunk_peak_abs_pct,
            t_toe_clearance_left_pct, t_toe_clearance_right_pct,
            toe_clearance_left_max, toe_clearance_right_max,
            toe_clearance_left_range, toe_clearance_right_range,
            foot_pitch_left_mean, foot_pitch_right_mean, foot_pitch_left_range, foot_pitch_right_range,
            trunk_abs_mean_deg, trunk_abs_p95_deg, trunk_ang_vel_mean_abs, trunk_ang_vel_p95_abs,
            cadence_diff, stride_time_diff, step_time_asymmetry_diff,
            stride_length_norm_diff, stride_amp_norm_diff, step_length_asymmetry_diff,
            knee_left_rom_diff, knee_right_rom_diff, knee_left_max_diff, knee_right_max_diff,
            ldj_knee_left_diff, ldj_knee_right_diff, ldj_hip_diff,
            stride_length_relL_norm_diff, stride_length_relR_norm_diff, stride_length_rel_asym_diff,
            ankle_ap_range_rel_norm_diff, midHip_ap_drift_norm_diff,
            t_knee_left_peak_pct_diff, t_knee_right_peak_pct_diff, t_trunk_peak_abs_pct_diff,
            t_toe_clearance_left_pct_diff, t_toe_clearance_right_pct_diff,
            toe_clearance_left_max_diff, toe_clearance_right_max_diff,
            toe_clearance_left_range_diff, toe_clearance_right_range_diff,
            foot_pitch_left_mean_diff, foot_pitch_right_mean_diff,
            foot_pitch_left_range_diff, foot_pitch_right_range_diff
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

        /** All 69 feature columns for CSV export (V1 + V2). */
        val CSV_FEATURE_COLUMNS = listOf(
            *FEATURE_COLUMNS.toTypedArray(),
            "stride_length_relL_norm", "stride_length_relR_norm", "stride_length_rel_asym",
            "ankle_ap_range_rel_norm", "midHip_ap_drift_norm",
            "t_knee_left_peak_pct", "t_knee_right_peak_pct", "t_trunk_peak_abs_pct",
            "t_toe_clearance_left_pct", "t_toe_clearance_right_pct",
            "toe_clearance_left_max", "toe_clearance_right_max",
            "toe_clearance_left_range", "toe_clearance_right_range",
            "foot_pitch_left_mean", "foot_pitch_right_mean",
            "foot_pitch_left_range", "foot_pitch_right_range",
            "trunk_abs_mean_deg", "trunk_abs_p95_deg",
            "trunk_ang_vel_mean_abs", "trunk_ang_vel_p95_abs",
            "cadence_diff", "stride_time_diff", "step_time_asymmetry_diff",
            "stride_length_norm_diff", "stride_amp_norm_diff", "step_length_asymmetry_diff",
            "knee_left_rom_diff", "knee_right_rom_diff", "knee_left_max_diff", "knee_right_max_diff",
            "ldj_knee_left_diff", "ldj_knee_right_diff", "ldj_hip_diff",
            "stride_length_relL_norm_diff", "stride_length_relR_norm_diff", "stride_length_rel_asym_diff",
            "ankle_ap_range_rel_norm_diff", "midHip_ap_drift_norm_diff",
            "t_knee_left_peak_pct_diff", "t_knee_right_peak_pct_diff", "t_trunk_peak_abs_pct_diff",
            "t_toe_clearance_left_pct_diff", "t_toe_clearance_right_pct_diff",
            "toe_clearance_left_max_diff", "toe_clearance_right_max_diff",
            "toe_clearance_left_range_diff", "toe_clearance_right_range_diff",
            "foot_pitch_left_mean_diff", "foot_pitch_right_mean_diff",
            "foot_pitch_left_range_diff", "foot_pitch_right_range_diff"
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
            stride_length_relL_norm = Float.NaN,
            stride_length_relR_norm = Float.NaN,
            stride_length_rel_asym = Float.NaN,
            ankle_ap_range_rel_norm = Float.NaN,
            midHip_ap_drift_norm = Float.NaN,
            t_knee_left_peak_pct = Float.NaN,
            t_knee_right_peak_pct = Float.NaN,
            t_trunk_peak_abs_pct = Float.NaN,
            t_toe_clearance_left_pct = Float.NaN,
            t_toe_clearance_right_pct = Float.NaN,
            toe_clearance_left_max = Float.NaN,
            toe_clearance_right_max = Float.NaN,
            toe_clearance_left_range = Float.NaN,
            toe_clearance_right_range = Float.NaN,
            foot_pitch_left_mean = Float.NaN,
            foot_pitch_right_mean = Float.NaN,
            foot_pitch_left_range = Float.NaN,
            foot_pitch_right_range = Float.NaN,
            trunk_abs_mean_deg = Float.NaN,
            trunk_abs_p95_deg = Float.NaN,
            trunk_ang_vel_mean_abs = Float.NaN,
            trunk_ang_vel_p95_abs = Float.NaN,
            cadence_diff = Float.NaN,
            stride_time_diff = Float.NaN,
            step_time_asymmetry_diff = Float.NaN,
            stride_length_norm_diff = Float.NaN,
            stride_amp_norm_diff = Float.NaN,
            step_length_asymmetry_diff = Float.NaN,
            knee_left_rom_diff = Float.NaN,
            knee_right_rom_diff = Float.NaN,
            knee_left_max_diff = Float.NaN,
            knee_right_max_diff = Float.NaN,
            ldj_knee_left_diff = Float.NaN,
            ldj_knee_right_diff = Float.NaN,
            ldj_hip_diff = Float.NaN,
            stride_length_relL_norm_diff = Float.NaN,
            stride_length_relR_norm_diff = Float.NaN,
            stride_length_rel_asym_diff = Float.NaN,
            ankle_ap_range_rel_norm_diff = Float.NaN,
            midHip_ap_drift_norm_diff = Float.NaN,
            t_knee_left_peak_pct_diff = Float.NaN,
            t_knee_right_peak_pct_diff = Float.NaN,
            t_trunk_peak_abs_pct_diff = Float.NaN,
            t_toe_clearance_left_pct_diff = Float.NaN,
            t_toe_clearance_right_pct_diff = Float.NaN,
            toe_clearance_left_max_diff = Float.NaN,
            toe_clearance_right_max_diff = Float.NaN,
            toe_clearance_left_range_diff = Float.NaN,
            toe_clearance_right_range_diff = Float.NaN,
            foot_pitch_left_mean_diff = Float.NaN,
            foot_pitch_right_mean_diff = Float.NaN,
            foot_pitch_left_range_diff = Float.NaN,
            foot_pitch_right_range_diff = Float.NaN,
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
    
    // Ankle positions
    val ankleLeftX: FloatArray,
    val ankleRightX: FloatArray,
    val ankleLeftY: FloatArray,
    val ankleRightY: FloatArray,
    
    // Hip positions
    val hipLeftY: FloatArray,
    val hipRightY: FloatArray,
    
    // Foot positions 
    val heelLeftY: FloatArray,
    val heelRightY: FloatArray,
    val toeLeftY: FloatArray,
    val toeRightY: FloatArray,
    val midHipX: FloatArray,
    
    // Velocities (computed after smoothing)
    var ankleLeftVy: FloatArray,
    var ankleRightVy: FloatArray,
    var hipAvgVy: FloatArray
) {
}
