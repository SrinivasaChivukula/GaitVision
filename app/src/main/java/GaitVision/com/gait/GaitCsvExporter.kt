package GaitVision.com.gait

import android.util.Log
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * CSV export utility for gait analysis results.
 * Exports in PC pipeline compatible format.
 * Uses OutputStream so callers can write to any destination (SAF, file, etc.)
 */
object GaitCsvExporter {
    
    private const val TAG = "GaitLogging"

    /** Sanitize a value for safe CSV output (prevents formula injection in Excel/Sheets). */
    private fun sanitize(value: String): String {
        val needsQuoting = value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }
        val startsWithFormula = value.firstOrNull()?.let { it == '=' || it == '+' || it == '-' || it == '@' } ?: false
        return when {
            startsWithFormula -> "\"'${value.replace("\"", "\"\"")}\""
            needsQuoting -> "\"${value.replace("\"", "\"\"")}\""
            else -> value
        }
    }

    /** Generate a suggested filename for the CSV export. */
    fun generateFilename(participantId: String): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        return "${participantId}_gait_${timestamp}.csv"
    }

    /**
     * Write gait features and diagnostics as CSV to the given OutputStream.
     * Optionally includes stride boundaries, selected stride indices, and per-frame signals.
     * Returns true on success.
     *
     * @param strides All detected strides (for boundary export)
     * @param selectedStrideIndices Indices of the 2 strides used for feature computation
     * @param signals Per-frame signals (when provided, appends a GAIT_SIGNALS section)
     */
    fun writeToStream(
        outputStream: OutputStream,
        features: GaitFeatures?,
        diagnostics: GaitDiagnostics,
        score: ScoringResult?,
        participantId: String,
        videoName: String,
        strides: List<Stride>? = null,
        selectedStrideIndices: List<Int>? = null,
        signals: Signals? = null
    ): Boolean {
        return try {
            Log.d(TAG, "Exporting: features=${features != null}, strides=${strides?.size}, signals=${signals != null}, selectedIndices=$selectedStrideIndices")
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())

            OutputStreamWriter(outputStream).use { writer ->
                // Section 1: GAIT_RESULTS
                val headers = mutableListOf(
                    "participant_id",
                    "video_name",
                    "timestamp",
                    "quality_flag",
                    "walking_direction",
                    "was_flipped",
                    "fps_detected",
                    "duration_s",
                    "num_frames_total",
                    "num_frames_valid",
                    "valid_frame_rate",
                    "num_steps_detected",
                    "num_strides_valid"
                )

                // Stride selection info
                headers.add("selected_stride_indices")
                headers.add("stride_0_start_frame")
                headers.add("stride_0_end_frame")
                headers.add("stride_1_start_frame")
                headers.add("stride_1_end_frame")

                // Add feature columns (V1 + V2)
                headers.addAll(GaitFeatures.CSV_FEATURE_COLUMNS)

                // Add score columns (3 models)
                headers.addAll(listOf(
                    "ae_score",
                    "ridge_score",
                    "pca_score"
                ))

                writer.write(headers.joinToString(","))
                writer.write("\n")

                // Data row
                val values = mutableListOf<String>()

                // Metadata (sanitize user-controlled strings to prevent CSV injection)
                values.add(sanitize(participantId))
                values.add(sanitize(videoName))
                values.add(timestamp)
                values.add(diagnostics.qualityFlag.name)
                values.add(sanitize(diagnostics.walkingDirection))
                values.add(diagnostics.wasFlipped.toString())
                values.add(diagnostics.fpsDetected.toString())
                values.add(diagnostics.durationS.toString())
                values.add(diagnostics.numFramesTotal.toString())
                values.add(diagnostics.numFramesValid.toString())
                values.add(diagnostics.validFrameRate.toString())
                values.add(diagnostics.numStepsDetected.toString())
                values.add(diagnostics.numStridesValid.toString())

                // Stride selection
                values.add(selectedStrideIndices?.joinToString(";") ?: "")
                val sel0 = selectedStrideIndices?.getOrNull(0)
                val sel1 = selectedStrideIndices?.getOrNull(1)
                val stride0 = strides?.getOrNull(sel0 ?: -1)
                val stride1 = strides?.getOrNull(sel1 ?: -1)
                values.add(stride0?.startFrame?.toString() ?: "")
                values.add(stride0?.endFrame?.toString() ?: "")
                values.add(stride1?.startFrame?.toString() ?: "")
                values.add(stride1?.endFrame?.toString() ?: "")

                // Features (or NaN if not available)
                if (features != null) {
                    val featureArray = features.toCsvFeatureArray()
                    for (f in featureArray) {
                        values.add(if (f.isNaN()) "NaN" else f.toString())
                    }
                } else {
                    repeat(GaitFeatures.CSV_FEATURE_COLUMNS.size) { values.add("NaN") }
                }

                // Scores (3 models)
                if (score != null) {
                    values.add(if (score.aeScore.isNaN()) "NaN" else score.aeScore.toString())
                    values.add(if (score.ridgeScore.isNaN()) "NaN" else score.ridgeScore.toString())
                    values.add(if (score.pcaScore.isNaN()) "NaN" else score.pcaScore.toString())
                } else {
                    repeat(3) { values.add("NaN") }
                }

                writer.write(values.joinToString(","))
                writer.write("\n")

                // Section 2: GAIT_SIGNALS
                if (signals != null) {
                    writer.write("\n")
                    writer.write("# GAIT_SIGNALS\n")
                    val sigHeaders = listOf(
                        "frame", "timestamp", "is_valid",
                        "inter_ankle_dist", "knee_left", "knee_right", "trunk",
                        "ankle_left_y", "ankle_right_y", "hip_left_y", "hip_right_y",
                        "heel_left_y", "heel_right_y", "toe_left_y", "toe_right_y", "mid_hip_x",
                        "ankle_left_vy", "ankle_right_vy"
                    )
                    writer.write(sigHeaders.joinToString(","))
                    writer.write("\n")

                    fun fmt(v: Float) = if (v.isNaN()) "NaN" else v.toString()
                    val n = signals.timestamps.size
                    for (i in 0 until n) {
                        val row = listOf(
                            i.toString(),
                            fmt(signals.timestamps.getOrNull(i) ?: Float.NaN),
                            (signals.isValid.getOrNull(i) ?: true).toString(),
                            fmt(signals.interAnkleDist.getOrNull(i) ?: Float.NaN),
                            fmt(signals.kneeAngleLeft.getOrNull(i) ?: Float.NaN),
                            fmt(signals.kneeAngleRight.getOrNull(i) ?: Float.NaN),
                            fmt(signals.trunkAngle.getOrNull(i) ?: Float.NaN),
                            fmt(signals.ankleLeftY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.ankleRightY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.hipLeftY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.hipRightY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.heelLeftY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.heelRightY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.toeLeftY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.toeRightY.getOrNull(i) ?: Float.NaN),
                            fmt(signals.midHipX.getOrNull(i) ?: Float.NaN),
                            fmt(signals.ankleLeftVy.getOrNull(i) ?: Float.NaN),
                            fmt(signals.ankleRightVy.getOrNull(i) ?: Float.NaN)
                        )
                        writer.write(row.joinToString(","))
                        writer.write("\n")
                    }
                }
            }

            Log.d(TAG, "CSV written: features=${features != null}, signals=${signals?.timestamps?.size ?: 0} rows, strides=${strides?.size ?: 0}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error writing CSV", e)
            false
        }
    }
    
    /**
     * Generate a summary string for display.
     */
    fun generateSummary(
        features: GaitFeatures?,
        diagnostics: GaitDiagnostics,
        score: ScoringResult?
    ): String {
        val sb = StringBuilder()
        
        sb.appendLine("Gait Analysis Summary")
        sb.appendLine()
        
        // Quality status
        sb.appendLine("Quality: ${diagnostics.qualityFlag}")
        if (diagnostics.rejectionReasons.isNotEmpty()) {
            sb.appendLine("Issues: ${diagnostics.rejectionReasons.joinToString(", ")}")
        }
        sb.appendLine()
        
        // Video info
        sb.appendLine("Video Info:")
        sb.appendLine("  Duration: %.1f s".format(diagnostics.durationS))
        sb.appendLine("  Frames: ${diagnostics.numFramesValid}/${diagnostics.numFramesTotal} valid")
        sb.appendLine("  Detection rate: ${(diagnostics.validFrameRate * 100).toInt()}%")
        sb.appendLine("  Walking direction: ${diagnostics.walkingDirection}")
        sb.appendLine()
        
        if (features != null) {
            // Key metrics
            sb.appendLine("Gait Metrics:")
            sb.appendLine("  Cadence: %.1f steps/min".format(features.cadence_spm))
            sb.appendLine("  Stride time: %.2f s".format(features.stride_time_s))
            sb.appendLine("  Step asymmetry: %.1f%%".format(features.step_time_asymmetry * 100))
            sb.appendLine()
            
            sb.appendLine("Range of Motion:")
            sb.appendLine("  Knee (L/R): %.1f° / %.1f°".format(features.knee_left_rom, features.knee_right_rom))
            sb.appendLine("  Max knee flexion (L/R): %.1f° / %.1f°".format(features.knee_left_max, features.knee_right_max))
            sb.appendLine()
            
            sb.appendLine("Stability:")
            sb.appendLine("  Trunk lean std: %.1f°".format(features.trunk_lean_std_deg))
            sb.appendLine("  Inter-ankle CV: %.2f".format(features.inter_ankle_cv))
            sb.appendLine()
        }
        
        if (score != null) {
            sb.appendLine("Model Scores (100 = healthy):")
            if (!score.aeScore.isNaN()) sb.appendLine("  AE (DB): %.0f/100".format(score.aeScore))
            if (!score.ridgeScore.isNaN()) sb.appendLine("  Ridge: %.0f/100".format(score.ridgeScore))
            if (!score.pcaScore.isNaN()) sb.appendLine("  PCA: %.0f/100".format(score.pcaScore))
        }
        
        return sb.toString()
    }
}
