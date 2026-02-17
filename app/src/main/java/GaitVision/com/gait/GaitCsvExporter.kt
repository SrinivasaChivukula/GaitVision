package GaitVision.com.gait

import android.util.Log
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.io.Writer
import java.text.SimpleDateFormat
import GaitVision.com.batch.MetadataRow
import java.util.*

/** CSV export for gait results. PC pipeline format. */
object GaitCsvExporter {
    
    private const val TAG = "GaitLogging"

    /** Quote formulas to prevent Excel injection. */
    private fun sanitize(value: String): String {
        val needsQuoting = value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }
        val startsWithFormula = value.firstOrNull()?.let { it == '=' || it == '+' || it == '-' || it == '@' } ?: false
        return when {
            startsWithFormula -> "\"'${value.replace("\"", "\"\"")}\""
            needsQuoting -> "\"${value.replace("\"", "\"\"")}\""
            else -> value
        }
    }

    fun generateFilename(participantId: String): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        return "${participantId}_gait_${timestamp}.csv"
    }

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
                // GAIT_RESULTS
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

                // Stride selection
                headers.add("selected_stride_indices")
                headers.add("stride_0_start_frame")
                headers.add("stride_0_end_frame")
                headers.add("stride_1_start_frame")
                headers.add("stride_1_end_frame")

                // Features
                headers.addAll(GaitFeatures.CSV_FEATURE_COLUMNS)

                // Scores
                headers.addAll(listOf(
                    "ae_score",
                    "ridge_score",
                    "pca_score"
                ))

                writer.write(headers.joinToString(","))
                writer.write("\n")

                // Row
                val values = mutableListOf<String>()

                // Metadata
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

                // Features
                if (features != null) {
                    val featureArray = features.toCsvFeatureArray()
                    for (f in featureArray) {
                        values.add(if (f.isNaN()) "NaN" else f.toString())
                    }
                } else {
                    repeat(GaitFeatures.CSV_FEATURE_COLUMNS.size) { values.add("NaN") }
                }

                // Scores
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

    /** Batch header: training dataset format. */
    fun writeBatchHeader(writer: Writer) {
        val headers = listOf(
            "video_id", "success", "processing_time", "patient_id", "condition", "severity", "trial", "label",
            "quality_flag", "used_roi", "has_mid_turn", "turn_ratio",
            "num_frames_total", "num_frames_valid", "valid_frame_rate", "num_strides_valid", "num_strides_used",
            "stride_selection_reason", "fps", "duration_s"
        ) + GaitFeatures.CSV_FEATURE_COLUMNS + "error"
        writer.write(headers.joinToString(","))
        writer.write("\n")
    }

    /** Write one batch row. */
    fun writeBatchRow(
        writer: Writer,
        videoId: String,
        success: Boolean,
        processingTimeMs: Long,
        metadata: MetadataRow,
        features: GaitFeatures?,
        diagnostics: GaitDiagnostics,
        errorMsg: String = ""
    ) {
        val patientIdShort = shortenSubjectId(metadata.subjectId)
        val conditionPc = mapConditionToPc(metadata.condition)
        val severityPc = mapSeverityToPc(metadata.severity)
        val labelCap = when (metadata.label.lowercase()) {
            "clean" -> "Normal"
            "impaired" -> "Impaired"
            else -> metadata.label.replaceFirstChar { it.uppercase() }
        }
        val values = mutableListOf<String>()
        values.add(sanitize(videoId))
        values.add(success.toString())
        values.add(processingTimeMs.toString())
        values.add(sanitize(patientIdShort))
        values.add(sanitize(conditionPc))
        values.add(sanitize(severityPc))
        values.add(sanitize(metadata.trial))
        values.add(sanitize(labelCap))
        values.add(diagnostics.qualityFlag.name)
        values.add("False")
        values.add("False")
        values.add("0")
        values.add(diagnostics.numFramesTotal.toString())
        values.add(diagnostics.numFramesValid.toString())
        values.add(diagnostics.validFrameRate.toString())
        values.add(diagnostics.numStridesValid.toString())
        values.add("2")
        values.add("best_consecutive_pair")
        values.add(diagnostics.fpsDetected.toString())
        values.add(diagnostics.durationS.toString())
        if (features != null) {
            for (f in features.toCsvFeatureArray()) {
                values.add(if (f.isNaN()) "NaN" else f.toString())
            }
        } else {
            repeat(GaitFeatures.CSV_FEATURE_COLUMNS.size) { values.add("NaN") }
        }
        values.add(sanitize(errorMsg))
        writer.write(values.joinToString(","))
        writer.write("\n")
    }

    private fun shortenSubjectId(subjectId: String): String {
        return subjectId.removePrefix("CLIN_").removePrefix("KAG_")
    }

    private fun mapConditionToPc(condition: String): String = when (condition) {
        "KneeOA" -> "KOA"
        "Normal" -> "NM"
        "Parkinsons" -> "PD"
        else -> condition
    }

    private fun mapSeverityToPc(severity: String): String = when (severity.lowercase()) {
        "early" -> "EL"
        "moderate" -> "MD"
        "severe" -> "SV"
        "none" -> "NM"
        "mild" -> "ML"
        else -> severity
    }

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
