package GaitVision.com.ui

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ScrollView
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
import GaitVision.com.extractedSignals
import GaitVision.com.extractedStrides
import GaitVision.com.extractionDiagnostics
import GaitVision.com.scoringResult
import GaitVision.com.selectedStrideIndices
import GaitVision.com.gait.*
import GaitVision.com.galleryUri

class ResultsActivity : BaseActivity() {

    companion object {
        const val EXTRA_RESULT_ID = "result_id"
        private const val TAG = "GaitUI"
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
        Log.d(TAG, "Loading result $id from DB")
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                AppDatabase.getDatabase(this@ResultsActivity).analysisResultDao().getResultById(id)
            }

            if (result == null) {
                Log.e(TAG, "Result $id not found in DB")
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
                stride_length_relL_norm = result.strideLengthRelLNorm ?: Float.NaN,
                stride_length_relR_norm = result.strideLengthRelRNorm ?: Float.NaN,
                stride_length_rel_asym = result.strideLengthRelAsym ?: Float.NaN,
                ankle_ap_range_rel_norm = result.ankleApRangeRelNorm ?: Float.NaN,
                midHip_ap_drift_norm = result.midHipApDriftNorm ?: Float.NaN,
                t_knee_left_peak_pct = result.tKneeLeftPeakPct ?: Float.NaN,
                t_knee_right_peak_pct = result.tKneeRightPeakPct ?: Float.NaN,
                t_trunk_peak_abs_pct = result.tTrunkPeakAbsPct ?: Float.NaN,
                t_toe_clearance_left_pct = result.tToeClearanceLeftPct ?: Float.NaN,
                t_toe_clearance_right_pct = result.tToeClearanceRightPct ?: Float.NaN,
                toe_clearance_left_max = result.toeClearanceLeftMax ?: Float.NaN,
                toe_clearance_right_max = result.toeClearanceRightMax ?: Float.NaN,
                toe_clearance_left_range = result.toeClearanceLeftRange ?: Float.NaN,
                toe_clearance_right_range = result.toeClearanceRightRange ?: Float.NaN,
                foot_pitch_left_mean = result.footPitchLeftMean ?: Float.NaN,
                foot_pitch_right_mean = result.footPitchRightMean ?: Float.NaN,
                foot_pitch_left_range = result.footPitchLeftRange ?: Float.NaN,
                foot_pitch_right_range = result.footPitchRightRange ?: Float.NaN,
                trunk_abs_mean_deg = result.trunkAbsMeanDeg ?: Float.NaN,
                trunk_abs_p95_deg = result.trunkAbsP95Deg ?: Float.NaN,
                trunk_ang_vel_mean_abs = result.trunkAngVelMeanAbs ?: Float.NaN,
                trunk_ang_vel_p95_abs = result.trunkAngVelP95Abs ?: Float.NaN,
                cadence_diff = result.cadenceDiff ?: Float.NaN,
                stride_time_diff = result.strideTimeDiff ?: Float.NaN,
                step_time_asymmetry_diff = result.stepTimeAsymmetryDiff ?: Float.NaN,
                stride_length_norm_diff = result.strideLengthNormDiff ?: Float.NaN,
                stride_amp_norm_diff = result.strideAmpNormDiff ?: Float.NaN,
                step_length_asymmetry_diff = result.stepLengthAsymmetryDiff ?: Float.NaN,
                knee_left_rom_diff = result.kneeLeftRomDiff ?: Float.NaN,
                knee_right_rom_diff = result.kneeRightRomDiff ?: Float.NaN,
                knee_left_max_diff = result.kneeLeftMaxDiff ?: Float.NaN,
                knee_right_max_diff = result.kneeRightMaxDiff ?: Float.NaN,
                ldj_knee_left_diff = result.ldjKneeLeftDiff ?: Float.NaN,
                ldj_knee_right_diff = result.ldjKneeRightDiff ?: Float.NaN,
                ldj_hip_diff = result.ldjHipDiff ?: Float.NaN,
                stride_length_relL_norm_diff = result.strideLengthRelLNormDiff ?: Float.NaN,
                stride_length_relR_norm_diff = result.strideLengthRelRNormDiff ?: Float.NaN,
                stride_length_rel_asym_diff = result.strideLengthRelAsymDiff ?: Float.NaN,
                ankle_ap_range_rel_norm_diff = result.ankleApRangeRelNormDiff ?: Float.NaN,
                midHip_ap_drift_norm_diff = result.midHipApDriftNormDiff ?: Float.NaN,
                t_knee_left_peak_pct_diff = result.tKneeLeftPeakPctDiff ?: Float.NaN,
                t_knee_right_peak_pct_diff = result.tKneeRightPeakPctDiff ?: Float.NaN,
                t_trunk_peak_abs_pct_diff = result.tTrunkPeakAbsPctDiff ?: Float.NaN,
                t_toe_clearance_left_pct_diff = result.tToeClearanceLeftPctDiff ?: Float.NaN,
                t_toe_clearance_right_pct_diff = result.tToeClearanceRightPctDiff ?: Float.NaN,
                toe_clearance_left_max_diff = result.toeClearanceLeftMaxDiff ?: Float.NaN,
                toe_clearance_right_max_diff = result.toeClearanceRightMaxDiff ?: Float.NaN,
                toe_clearance_left_range_diff = result.toeClearanceLeftRangeDiff ?: Float.NaN,
                toe_clearance_right_range_diff = result.toeClearanceRightRangeDiff ?: Float.NaN,
                foot_pitch_left_mean_diff = result.footPitchLeftMeanDiff ?: Float.NaN,
                foot_pitch_right_mean_diff = result.footPitchRightMeanDiff ?: Float.NaN,
                foot_pitch_left_range_diff = result.footPitchLeftRangeDiff ?: Float.NaN,
                foot_pitch_right_range_diff = result.footPitchRightRangeDiff ?: Float.NaN,
                valid_stride_count = result.validStrideCount
            )

            Log.d(TAG, "Loaded from DB: features=${extractedFeatures != null}, V2 sample: stride_relL=${extractedFeatures?.stride_length_relL_norm}, toe_max_L=${extractedFeatures?.toe_clearance_left_max}")

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
            appendLine("V1 Features")
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
            appendLine()
            appendLine("V2: Camera-robust")
            appendLine("Stride length rel L/R norm: ${fmt(f.stride_length_relL_norm)} / ${fmt(f.stride_length_relR_norm)}")
            appendLine("Stride length rel asym: ${fmt(f.stride_length_rel_asym)}")
            appendLine("Ankle AP range rel norm: ${fmt(f.ankle_ap_range_rel_norm)}")
            appendLine("MidHip AP drift norm: ${fmt(f.midHip_ap_drift_norm)}")
            appendLine()
            appendLine("V2: Timing-of-extrema")
            appendLine("t knee peak L/R pct: ${fmt(f.t_knee_left_peak_pct)} / ${fmt(f.t_knee_right_peak_pct)}")
            appendLine("t trunk peak abs pct: ${fmt(f.t_trunk_peak_abs_pct)}")
            appendLine("t toe clearance L/R pct: ${fmt(f.t_toe_clearance_left_pct)} / ${fmt(f.t_toe_clearance_right_pct)}")
            appendLine()
            appendLine("V2: Foot clearance + pitch")
            appendLine("Toe clearance max L/R: ${fmt(f.toe_clearance_left_max)} / ${fmt(f.toe_clearance_right_max)}")
            appendLine("Toe clearance range L/R: ${fmt(f.toe_clearance_left_range)} / ${fmt(f.toe_clearance_right_range)}")
            appendLine("Foot pitch mean L/R: ${fmt(f.foot_pitch_left_mean)} / ${fmt(f.foot_pitch_right_mean)}")
            appendLine("Foot pitch range L/R: ${fmt(f.foot_pitch_left_range)} / ${fmt(f.foot_pitch_right_range)}")
            appendLine()
            appendLine("V2: Trunk enrichments")
            appendLine("Trunk abs mean/p95 deg: ${fmt(f.trunk_abs_mean_deg)} / ${fmt(f.trunk_abs_p95_deg)}")
            appendLine("Trunk ang vel mean/p95 abs: ${fmt(f.trunk_ang_vel_mean_abs)} / ${fmt(f.trunk_ang_vel_p95_abs)}")
            appendLine()
            appendLine("V2: Diff (cycle A − B)")
            appendLine("Cadence diff: ${fmt(f.cadence_diff)} | Stride time diff: ${fmt(f.stride_time_diff)}")
            appendLine("Step time asym diff: ${fmt(f.step_time_asymmetry_diff)}")
            appendLine("Stride length/amp norm diff: ${fmt(f.stride_length_norm_diff)} / ${fmt(f.stride_amp_norm_diff)}")
            appendLine("Step length asym diff: ${fmt(f.step_length_asymmetry_diff)}")
            appendLine("Knee ROM diff L/R: ${fmt(f.knee_left_rom_diff)} / ${fmt(f.knee_right_rom_diff)}")
            appendLine("Knee max diff L/R: ${fmt(f.knee_left_max_diff)} / ${fmt(f.knee_right_max_diff)}")
            appendLine("LDJ diff knee L/R, hip: ${fmt(f.ldj_knee_left_diff)} / ${fmt(f.ldj_knee_right_diff)} / ${fmt(f.ldj_hip_diff)}")
            appendLine("Stride rel L/R/asym diff: ${fmt(f.stride_length_relL_norm_diff)} / ${fmt(f.stride_length_relR_norm_diff)} / ${fmt(f.stride_length_rel_asym_diff)}")
            appendLine("Ankle AP / MidHip drift diff: ${fmt(f.ankle_ap_range_rel_norm_diff)} / ${fmt(f.midHip_ap_drift_norm_diff)}")
            appendLine("t knee peak diff L/R: ${fmt(f.t_knee_left_peak_pct_diff)} / ${fmt(f.t_knee_right_peak_pct_diff)}")
            appendLine("t trunk/toe clearance diff: ${fmt(f.t_trunk_peak_abs_pct_diff)} / ${fmt(f.t_toe_clearance_left_pct_diff)} / ${fmt(f.t_toe_clearance_right_pct_diff)}")
            appendLine("Toe clearance max/range diff: ${fmt(f.toe_clearance_left_max_diff)} / ${fmt(f.toe_clearance_right_max_diff)} / ${fmt(f.toe_clearance_left_range_diff)} / ${fmt(f.toe_clearance_right_range_diff)}")
            append("Foot pitch mean/range diff: ${fmt(f.foot_pitch_left_mean_diff)} / ${fmt(f.foot_pitch_right_mean_diff)} / ${fmt(f.foot_pitch_left_range_diff)} / ${fmt(f.foot_pitch_right_range_diff)}")
        }

        val scrollView = ScrollView(this).apply {
            addView(TextView(this@ResultsActivity).apply {
                text = msg
                setPadding(48, 24, 48, 24)
                textSize = 12f
            })
        }
        AlertDialog.Builder(this)
            .setTitle("Gait Features (${f.valid_stride_count} strides)")
            .setView(scrollView)
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
                    videoName = videoName,
                    strides = extractedStrides,
                    selectedStrideIndices = selectedStrideIndices,
                    signals = extractedSignals
                )
                if (success) {
                    Toast.makeText(this, "CSV exported successfully", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Failed to write CSV", Toast.LENGTH_SHORT).show()
                }
            } ?: Toast.makeText(this, "Could not open file for writing", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting: ${e.message}", e)
            Toast.makeText(this, "Error exporting: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

}
