package GaitVision.com.ui

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AnalysisResult
import GaitVision.com.data.AppDatabase
import GaitVision.com.participantId
import GaitVision.com.extractedFeatures
import GaitVision.com.extractionDiagnostics
import GaitVision.com.scoringResult
import GaitVision.com.gait.*
import GaitVision.com.galleryUri

class ResultsActivity : BaseActivity() {

    companion object {
        const val EXTRA_RESULT_ID = "result_id"
    }

    private lateinit var tvGaitScore: TextView
    private lateinit var tvScoreLabel: TextView
    private lateinit var tvAeScore: TextView
    private lateinit var tvRidgeScore: TextView
    private lateinit var tvPcaScore: TextView

    private var calculatedScore: Double = 0.0
    private var resultId: Long = -1L

    /** SAF file picker for CSV export, user chooses save location maybe more secure??? */
    private val csvExportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri -> uri?.let { writeCsvToUri(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results)

        setupCommonHeader("Analysis Results")
        initializeViews()
        setupButtons()

        resultId = intent.getLongExtra(EXTRA_RESULT_ID, -1L)

        if (resultId > 0) {
            // Load from DB into globals, then display
            loadFromDatabase(resultId)
        } else {
            // Globals already set from live analysis
            calculateGaitScore()
        }
    }

    private fun initializeViews() {
        tvGaitScore = findViewById(R.id.tvGaitScore)
        tvScoreLabel = findViewById(R.id.tvScoreLabel)
        tvAeScore = findViewById(R.id.tvAeScore)
        tvRidgeScore = findViewById(R.id.tvRidgeScore)
        tvPcaScore = findViewById(R.id.tvPcaScore)
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnMainMenu).setOnClickListener {
            val intent = Intent(this, DashboardActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()
        }

        findViewById<Button>(R.id.btnExportCsv).setOnClickListener {
            exportCsvFiles()
        }

        findViewById<Button>(R.id.btnViewFeatures).setOnClickListener {
            showFeaturesDialog()
        }

        findViewById<Button>(R.id.btnSignalsDashboard).setOnClickListener {
            val intent = Intent(this, SignalsDashboardActivity::class.java)
            if (resultId > 0) intent.putExtra(EXTRA_RESULT_ID, resultId)
            startActivity(intent)
        }
    }

    /**
     * Load an AnalysisResult from DB and populate the same globals that
     * the live analysis flow uses. Then call calculateGaitScore() as normal.
     *
     * WARNING HEY READ THIS REEEEEEEEAD: This overwrites shared globals (extractedFeatures, scoringResult, etc.).
     * Safe today because navigation is linear, but a ViewModel/StateFlow refactor
     * should replace this if we ever need concurrent or comparative analysis views.
     */
    private fun loadFromDatabase(id: Long) {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                AppDatabase.getDatabase(this@ResultsActivity).analysisResultDao().getResultById(id)
            }

            if (result == null) {
                Toast.makeText(this@ResultsActivity, "Analysis not found", Toast.LENGTH_SHORT).show()
                finish()
                return@launch
            }

            // Populate globals so all existing code paths work
            extractedFeatures = GaitFeatures(
                cadence_spm = result.cadenceSpm ?: Float.NaN,
                stride_time_s = result.strideTimeS ?: Float.NaN,
                stride_time_cv = result.strideTimeCv ?: Float.NaN,
                step_time_asymmetry = result.stepTimeAsymmetry ?: Float.NaN,
                stride_length_norm = result.strideLengthNorm ?: Float.NaN,
                stride_amp_norm = result.strideAmpNorm ?: Float.NaN,
                step_length_asymmetry = result.stepLengthAsymmetry ?: Float.NaN,
                knee_left_rom = result.kneeLeftRom ?: Float.NaN,
                knee_right_rom = result.kneeRightRom ?: Float.NaN,
                knee_left_max = result.kneeLeftMax ?: Float.NaN,
                knee_right_max = result.kneeRightMax ?: Float.NaN,
                ldj_knee_left = result.ldjKneeLeft ?: Float.NaN,
                ldj_knee_right = result.ldjKneeRight ?: Float.NaN,
                ldj_hip = result.ldjHip ?: Float.NaN,
                trunk_lean_std_deg = result.trunkLeanStdDeg ?: Float.NaN,
                inter_ankle_cv = result.interAnkleCv ?: Float.NaN,
                valid_stride_count = result.validStrideCount
            )

            scoringResult = ScoringResult(
                aeScore = result.aeScore ?: Float.NaN,
                ridgeScore = result.ridgeScore ?: Float.NaN,
                pcaScore = result.pcaScore ?: Float.NaN
            )

            extractionDiagnostics = GaitDiagnostics(
                videoId = result.videoFileName,
                fpsDetected = result.fpsDetected ?: 30f,
                durationS = (result.videoLengthMicroseconds ?: 0) / 1_000_000f,
                numFramesTotal = result.numFramesTotal,
                numFramesValid = result.numFramesValid,
                validFrameRate = result.validFrameRate ?: 0f,
                numStepsDetected = result.numStepsDetected,
                numStridesValid = result.validStrideCount,
                estimatedCadenceSpm = result.cadenceSpm ?: 0f,
                walkingDirection = result.walkingDirection ?: "unknown",
                wasFlipped = result.wasFlipped,
                qualityFlag = try { QualityFlag.valueOf(result.qualityFlag ?: "OK") } catch (_: Exception) { QualityFlag.OK }
            )

            participantId = result.patientId

            // Now just use the same display path
            calculateGaitScore()
        }
    }

    private fun calculateGaitScore() {
        val pcFeatures = extractedFeatures
        val pcScore = scoringResult
        val diagnostics = extractionDiagnostics

        if (pcFeatures != null && pcScore != null && pcFeatures.valid_stride_count > 0) {
            calculatedScore = pcScore.getScoreForDatabase()
            tvGaitScore.text = calculatedScore.toLong().toString()
            tvScoreLabel.text = "${getScoreLabel(calculatedScore)}\n(${pcFeatures.valid_stride_count} strides, ${String.format("%.1f", pcFeatures.cadence_spm)} spm)"

            tvAeScore.text = if (!pcScore.aeScore.isNaN()) pcScore.aeScore.toLong().toString() else "--"
            tvRidgeScore.text = if (!pcScore.ridgeScore.isNaN()) pcScore.ridgeScore.toLong().toString() else "--"
            tvPcaScore.text = if (!pcScore.pcaScore.isNaN()) pcScore.pcaScore.toLong().toString() else "--"

            tvAeScore.setTextColor(getScoreColor(pcScore.aeScore))
            tvRidgeScore.setTextColor(getScoreColor(pcScore.ridgeScore))
            tvPcaScore.setTextColor(getScoreColor(pcScore.pcaScore))

            val scoreColor = when {
                calculatedScore >= 80 -> "#4CAF50"
                calculatedScore >= 60 -> "#FF9800"
                else -> "#F44336"
            }
            tvGaitScore.setTextColor(android.graphics.Color.parseColor(scoreColor))
        } else {
            tvGaitScore.text = "--"
            tvAeScore.text = "--"
            tvRidgeScore.text = "--"
            tvPcaScore.text = "--"

            val errorMsg = when {
                pcFeatures == null && diagnostics != null ->
                    "Extraction failed: ${diagnostics.qualityFlag}\n${diagnostics.rejectionReasons.firstOrNull() ?: ""}"
                pcFeatures == null -> "Feature extraction not run"
                pcFeatures.valid_stride_count == 0 -> "No valid gait cycles detected"
                pcScore == null -> "Scoring failed to initialize"
                else -> "Unknown error"
            }
            tvScoreLabel.text = errorMsg
            tvGaitScore.setTextColor(android.graphics.Color.parseColor("#F44336"))
        }
    }

    private fun getScoreLabel(score: Double): String {
        return when {
            score >= 90 -> "Excellent Gait"
            score >= 80 -> "Good Gait"
            score >= 70 -> "Fair Gait"
            score >= 60 -> "Moderate Impairment"
            score >= 50 -> "Notable Impairment"
            else -> "Significant Impairment"
        }
    }

    private fun getScoreColor(score: Float): Int {
        if (score.isNaN()) return android.graphics.Color.GRAY
        return when {
            score >= 80 -> android.graphics.Color.parseColor("#4CAF50")
            score >= 60 -> android.graphics.Color.parseColor("#FF9800")
            else -> android.graphics.Color.parseColor("#F44336")
        }
    }

    private fun showFeaturesDialog() {
        val f = extractedFeatures
        if (f == null) {
            Toast.makeText(this, "No features available", Toast.LENGTH_SHORT).show()
            return
        }

        fun fmt(v: Float) = if (v.isNaN()) "--" else String.format("%.2f", v)

        val msg = buildString {
            appendLine("Cadence: ${fmt(f.cadence_spm)} spm")
            appendLine("Stride time: ${fmt(f.stride_time_s)} s")
            appendLine("Stride time CV: ${fmt(f.stride_time_cv)}")
            appendLine("Step time asymmetry: ${fmt(f.step_time_asymmetry)}")
            appendLine("Stride length: ${fmt(f.stride_length_norm)}")
            appendLine("Stride amplitude: ${fmt(f.stride_amp_norm)}")
            appendLine("Step length asymmetry: ${fmt(f.step_length_asymmetry)}")
            appendLine("Knee ROM L/R: ${fmt(f.knee_left_rom)} / ${fmt(f.knee_right_rom)}")
            appendLine("Knee max L/R: ${fmt(f.knee_left_max)} / ${fmt(f.knee_right_max)}")
            appendLine("LDJ knee L/R: ${fmt(f.ldj_knee_left)} / ${fmt(f.ldj_knee_right)}")
            appendLine("LDJ hip: ${fmt(f.ldj_hip)}")
            appendLine("Trunk lean std: ${fmt(f.trunk_lean_std_deg)}")
            append("Inter-ankle CV: ${fmt(f.inter_ankle_cv)}")
        }

        AlertDialog.Builder(this)
            .setTitle("Gait Features (${f.valid_stride_count} strides)")
            .setMessage(msg)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun exportCsvFiles() {
        if (extractionDiagnostics == null) {
            Toast.makeText(this, "Nothing to export", Toast.LENGTH_SHORT).show()
            return
        }

        val filePrefix = if (participantId == 0) {
            val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", java.util.Locale.US).format(java.util.Date())
            "${timestamp}_0"
        } else {
            participantId.toString()
        }

        val filename = GaitCsvExporter.generateFilename(filePrefix)
        csvExportLauncher.launch(filename)
    }

    private fun writeCsvToUri(uri: Uri) {
        try {
            val diagnostics = extractionDiagnostics ?: return
            val videoName = galleryUri?.lastPathSegment
                ?: diagnostics.videoId.takeIf { it.isNotBlank() }
                ?: "unknown"
            val filePrefix = if (participantId == 0) "0" else participantId.toString()

            contentResolver.openOutputStream(uri)?.use { stream ->
                val success = GaitCsvExporter.writeToStream(
                    outputStream = stream,
                    features = extractedFeatures,
                    diagnostics = diagnostics,
                    score = scoringResult,
                    participantId = filePrefix,
                    videoName = videoName
                )
                if (success) {
                    Toast.makeText(this, "CSV exported successfully", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Failed to write CSV", Toast.LENGTH_SHORT).show()
                }
            } ?: Toast.makeText(this, "Could not open file for writing", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("ResultsActivity", "Error exporting: ${e.message}", e)
            Toast.makeText(this, "Error exporting: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

}
