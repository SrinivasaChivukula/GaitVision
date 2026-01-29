package GaitVision.com.gait

import android.content.Context
import android.media.MediaScannerConnection
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * CSV export utility for gait analysis results.
 * Exports in PC pipeline compatible format.
 */
object GaitCsvExporter {
    
    private const val TAG = "GaitCsvExporter"
    
    /**
     * Export gait features and diagnostics to CSV.
     * Returns the file path of the exported CSV.
     */
    fun exportToCSV(
        context: Context,
        features: GaitFeatures?,
        diagnostics: GaitDiagnostics,
        score: ScoringResult?,
        participantId: String,
        videoName: String
    ): String? {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val filename = "${participantId}_gait_${timestamp}.csv"
            
            val file = File(context.getExternalFilesDir(null), filename)
            
            FileWriter(file).use { writer ->
                // Header row
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
                
                // Add feature columns
                headers.addAll(GaitFeatures.FEATURE_COLUMNS)
                
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
                
                // Metadata
                values.add(participantId)
                values.add(videoName)
                values.add(timestamp)
                values.add(diagnostics.qualityFlag.name)
                values.add(diagnostics.walkingDirection)
                values.add(diagnostics.wasFlipped.toString())
                values.add(diagnostics.fpsDetected.toString())
                values.add(diagnostics.durationS.toString())
                values.add(diagnostics.numFramesTotal.toString())
                values.add(diagnostics.numFramesValid.toString())
                values.add(diagnostics.validFrameRate.toString())
                values.add(diagnostics.numStepsDetected.toString())
                values.add(diagnostics.numStridesValid.toString())
                
                // Features (or NaN if not available)
                if (features != null) {
                    val featureArray = features.toFeatureArray()
                    for (f in featureArray) {
                        values.add(if (f.isNaN()) "NaN" else f.toString())
                    }
                } else {
                    repeat(16) { values.add("NaN") }
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
            }
            
            Log.d(TAG, "Exported CSV to: ${file.absolutePath}")
            return file.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting CSV", e)
            return null
        }
    }
    
    /**
     * Export per-frame signals to CSV (for debugging/visualization).
     * Includes all computed signals matching PC dashboard.
     */
    fun exportSignalsToCSV(
        context: Context,
        signals: Signals,
        participantId: String
    ): String? {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val filename = "${participantId}_signals_${timestamp}.csv"
            
            val fileDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val file = File(fileDirectory, filename)
            
            FileWriter(file).use { writer ->
                // Header - all signals
                writer.write("frame_idx,timestamp_s,is_valid,inter_ankle_dist,knee_angle_left,knee_angle_right,trunk_angle,ankle_left_y,ankle_right_y,hip_left_y,hip_right_y,ankle_left_vy,ankle_right_vy,hip_avg_vy\n")
                
                // Data rows
                for (i in signals.frameIndices.indices) {
                    val row = listOf(
                        signals.frameIndices[i].toString(),
                        formatFloat(signals.timestamps[i]),
                        signals.isValid[i].toString(),
                        formatFloat(signals.interAnkleDist[i]),
                        formatFloat(signals.kneeAngleLeft[i]),
                        formatFloat(signals.kneeAngleRight[i]),
                        formatFloat(signals.trunkAngle[i]),
                        formatFloat(signals.ankleLeftY[i]),
                        formatFloat(signals.ankleRightY[i]),
                        formatFloat(signals.hipLeftY[i]),
                        formatFloat(signals.hipRightY[i]),
                        formatFloat(signals.ankleLeftVy[i]),
                        formatFloat(signals.ankleRightVy[i]),
                        formatFloat(signals.hipAvgVy[i])
                    )
                    writer.write(row.joinToString(","))
                    writer.write("\n")
                }
            }
            
            // Scan so it shows up in file browsers
            MediaScannerConnection.scanFile(context, arrayOf(file.absolutePath), null, null)
            
            Log.d(TAG, "Exported signals CSV to: ${file.absolutePath}")
            return file.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting signals CSV", e)
            return null
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
        
        sb.appendLine("=== Gait Analysis Summary ===")
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
    
    private fun formatFloat(value: Float): String {
        return if (value.isNaN()) "NaN" else "%.4f".format(value)
    }
}
