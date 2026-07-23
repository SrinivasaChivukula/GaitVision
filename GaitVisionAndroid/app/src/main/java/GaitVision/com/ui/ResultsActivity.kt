package GaitVision.com.ui

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.AnalysisSession
import GaitVision.com.gait.*

class ResultsActivity : BaseActivity() {

    companion object {
        private const val KEY_PROFESSIONALLY_REVIEWED = "professionally_reviewed"
        const val EXTRA_RESULT_ID = "result_id"
    }

    private lateinit var tvGaitScore: TextView
    private lateinit var tvScoreLabel: TextView
    private lateinit var tvAeScore: TextView
    private lateinit var tvRidgeScore: TextView
    private lateinit var tvPcaScore: TextView

    private var calculatedScore: Double = 0.0
    private var resultId: Long = -1L
    private var isProfessionallyReviewed = false
    
    // Local state to avoid race conditions with singleton AnalysisSession
    private var localFeatures: GaitFeatures? = null
    private var localDiagnostics: GaitDiagnostics? = null
    private var localScore: ScoringResult? = null
    private var localJitterComparison: PoseJitterComparison? = null
    private var localParticipantId: Int = 0
    private var localVideoUri: Uri? = null
    
    /** SAF file picker for CSV export -- user chooses save location */
    private val csvExportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri -> uri?.let { writeCsvToUri(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results)

        isProfessionallyReviewed = savedInstanceState?.getBoolean(KEY_PROFESSIONALLY_REVIEWED) ?: false

        setupCommonHeader("Analysis Results")
        initializeViews()
        setupButtons()

        resultId = intent.getLongExtra(EXTRA_RESULT_ID, -1L)

        if (resultId > 0) {
            loadFromDatabase(resultId)
        } else {
            // Copy from global session to local state (Live analysis path)
            // We copy immediately so if a new analysis starts in bg, we hold onto these results
            localFeatures = AnalysisSession.extractedFeatures
            localDiagnostics = AnalysisSession.extractionDiagnostics
            localScore = AnalysisSession.scoringResult
            localJitterComparison = AnalysisSession.jitterComparison
            localParticipantId = AnalysisSession.participantId
            localVideoUri = AnalysisSession.galleryUri

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
        findViewById<LinearLayout>(R.id.btnExportCsv).setOnClickListener {
            if (!isProfessionallyReviewed) {
                showReviewDialog()
                return@setOnClickListener
            }

            exportCsvFiles()
        }

        findViewById<LinearLayout>(R.id.btnViewFeatures).setOnClickListener {
            showFeaturesDialog()
        }

        findViewById<LinearLayout>(R.id.btnSignalsDashboard).setOnClickListener {
            val intent = Intent(this, SignalsDashboardActivity::class.java)
            if (resultId > 0) intent.putExtra(EXTRA_RESULT_ID, resultId)
            startActivity(intent)
        }
    }

    private fun showReviewDialog() {
        AlertDialog.Builder(this)
            .setTitle("Professional Review Required")
            .setMessage("These results are AI-assisted and should be reviewed by a qualified professional before export.")
            .setPositiveButton("Confirm Review") { _, _ ->
                isProfessionallyReviewed = true
                exportCsvFiles()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

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

            localFeatures = GaitFeatures(
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

            localScore = ScoringResult(
                aeScore = result.aeScore ?: Float.NaN,
                ridgeScore = result.ridgeScore ?: Float.NaN,
                pcaScore = result.pcaScore ?: Float.NaN
            )

            localJitterComparison = buildJitterComparisonFromResult(result)

            localDiagnostics = GaitDiagnostics(
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
                qualityFlag = try {
                    QualityFlag.valueOf(result.qualityFlag ?: "OK")
                } catch (_: Exception) {
                    QualityFlag.OK
                }
            )

            localParticipantId = result.patientId
            localVideoUri = null

            calculateGaitScore()
        }
    }

    private fun calculateGaitScore() {
        val pcFeatures = localFeatures
        val pcScore = localScore
        val diagnostics = localDiagnostics

        if (pcFeatures != null && pcScore != null && pcFeatures.valid_stride_count > 0) {
            calculatedScore = pcScore.getScoreForDatabase()
            tvGaitScore.text = calculatedScore.toLong().toString()
            tvScoreLabel.text =
                "${getScoreLabel(calculatedScore)}\n(${pcFeatures.valid_stride_count} strides, ${String.format("%.1f", pcFeatures.cadence_spm)} spm)"

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
            score >= 85 -> "Excellent Gait"
            score >= 70 -> "Good Gait"
            score >= 55 -> "Fair Gait"
            score >= 40 -> "Moderate Impairment"
            score >= 25 -> "Notable Impairment"
            else -> "Significant Impairment"
        }
    }

    private fun getScoreColor(score: Float): Int {
        if (score.isNaN()) return android.graphics.Color.GRAY
        return when {
            score >= 70 -> android.graphics.Color.parseColor("#4CAF50")
            score >= 40 -> android.graphics.Color.parseColor("#FF9800")
            else -> android.graphics.Color.parseColor("#F44336")
        }
    }

    private fun showFeaturesDialog() {
        val f = localFeatures
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
            appendLine("Inter-ankle CV: ${fmt(f.inter_ankle_cv)}")
            localJitterComparison?.let { jitter ->
                appendLine()
                appendLine("Pose jitter reduction: ${fmt(jitter.jitterReductionPct)}%")
                appendLine("Pose velocity retained: ${fmt(jitter.velocityRetentionPct)}%")
                append("Pose snap reduction: ${fmt(jitter.snapReductionPct)}%")
            }
        }

        AlertDialog.Builder(this)
            .setTitle("Gait Features (${f.valid_stride_count} strides)")
            .setMessage(msg)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun exportCsvFiles() {
        if (localDiagnostics == null) {
            Toast.makeText(this, "Nothing to export", Toast.LENGTH_SHORT).show()
            return
        }

        val filePrefix = if (localParticipantId == 0) {
            val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", java.util.Locale.US)
                .format(java.util.Date())
            "${timestamp}_0"
        } else {
            localParticipantId.toString()
        }

        val filename = GaitCsvExporter.generateFilename(filePrefix)
        csvExportLauncher.launch(filename)
    }

    private fun writeCsvToUri(uri: Uri) {
        try {
            val diagnostics = localDiagnostics ?: return
            val videoName = localVideoUri?.lastPathSegment
                ?: diagnostics.videoId.takeIf { it.isNotBlank() }
                ?: "unknown"

            val filePrefix = if (localParticipantId == 0) "0" else localParticipantId.toString()

            contentResolver.openOutputStream(uri)?.use { stream ->
                val success = GaitCsvExporter.writeToStream(
                    outputStream = stream,
                    features = localFeatures,
                    diagnostics = diagnostics,
                    score = localScore,
                    participantId = filePrefix,
                    videoName = videoName,
                    jitterComparison = localJitterComparison
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

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(KEY_PROFESSIONALLY_REVIEWED, isProfessionallyReviewed)
    }

    private fun buildJitterComparisonFromResult(
        result: GaitVision.com.data.AnalysisResult
    ): PoseJitterComparison? {
        val rawJitter = result.rawPoseJitter ?: return null
        val smoothedJitter = result.smoothedPoseJitter ?: return null
        val rawVelocity = result.rawPoseVelocity ?: Float.NaN
        val smoothedVelocity = result.smoothedPoseVelocity ?: Float.NaN
        val rawSnapRate = result.rawPoseSnapRate ?: Float.NaN
        val smoothedSnapRate = result.smoothedPoseSnapRate ?: Float.NaN
        val coverage = result.poseConfidenceCoverage ?: Float.NaN
        val bodyScale = result.poseMedianBodyScale ?: Float.NaN
        val poseFrames = result.numFramesValid
        val detectionRate = result.validFrameRate ?: Float.NaN

        return PoseJitterComparison(
            raw = PoseJitterMetrics(
                numPoseFrames = poseFrames,
                detectionRate = detectionRate,
                confidenceCoverage = coverage,
                medianBodyScale = bodyScale,
                jitterSecondDiffNorm = rawJitter,
                meanVelocityNorm = rawVelocity,
                snapRate = rawSnapRate
            ),
            smoothed = PoseJitterMetrics(
                numPoseFrames = poseFrames,
                detectionRate = detectionRate,
                confidenceCoverage = coverage,
                medianBodyScale = bodyScale,
                jitterSecondDiffNorm = smoothedJitter,
                meanVelocityNorm = smoothedVelocity,
                snapRate = smoothedSnapRate
            )
        )
    }
}
