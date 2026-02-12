package GaitVision.com.ui

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.MediaController
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.LimitLine
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AnalysisResult
import GaitVision.com.data.SignalData
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.repository.AnalysisResultRepository
import GaitVision.com.data.repository.SignalDataRepository
import GaitVision.com.data.repository.PatientRepository
import GaitVision.com.ProcVidEmpty
import GaitVision.com.galleryUri
import GaitVision.com.editedUri
import GaitVision.com.extractedSignals
import GaitVision.com.participantId
import GaitVision.com.participantHeight
import GaitVision.com.currentPatientId
import GaitVision.com.currentResultId
import GaitVision.com.videoLength
import GaitVision.com.extractedFeatures
import GaitVision.com.extractedStrides
import GaitVision.com.extractionDiagnostics
import GaitVision.com.scoringResult
import GaitVision.com.selectedStrideIndices
import GaitVision.com.stepSignalMode
import android.provider.OpenableColumns
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class AnalysisActivity : BaseActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null
    private var isProcessing = false
    private var shouldSave = true
    private var lastCursorIndex = -1
    private var cursorLimitLine: LimitLine? = null

    // Cached view references (resolved once, used every frame)
    private lateinit var videoView: VideoView
    private lateinit var strideChart: LineChart
    private lateinit var tvChartLabel: TextView
    private lateinit var tvAnkleAngles: TextView
    private lateinit var tvKneeAngles: TextView
    private lateinit var tvHipAngles: TextView
    private lateinit var tvTorsoAngle: TextView

    companion object {
        const val EXTRA_SHOULD_SAVE = "should_save"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

        // Cache view references once
        videoView = findViewById(R.id.videoView)
        strideChart = findViewById(R.id.strideChart)
        tvChartLabel = findViewById(R.id.tvChartLabel)
        tvAnkleAngles = findViewById(R.id.tvAnkleAngles)
        tvKneeAngles = findViewById(R.id.tvKneeAngles)
        tvHipAngles = findViewById(R.id.tvHipAngles)
        tvTorsoAngle = findViewById(R.id.tvTorsoAngle)

        setupCommonHeader("Gait Analysis")
        setCustomBackAction {
            updateRunnable?.let { handler.removeCallbacks(it) }
            finish()
        }

        // Get intent extras
        shouldSave = intent.getBooleanExtra(EXTRA_SHOULD_SAVE, true)

        // Get the video URI from VideoPickerActivity
        intent.data?.let { uri ->
            galleryUri = uri
        }

        setupInitialUI()
        setupButtons()

        // Check if we have a video to process
        if (galleryUri == null) {
            Toast.makeText(this, "No video selected. Please go back.", Toast.LENGTH_SHORT).show()
            findViewById<Button>(R.id.btnRunAnalysis).isEnabled = false
        }

        // If already processed, show the video
        if (editedUri != null) {
            showProcessedVideo()
        }
    }

    private fun setupInitialUI() {
        // Hide processing UI initially
        findViewById<View>(R.id.progressSection).visibility = View.GONE
        findViewById<View>(R.id.videoSection).visibility = View.GONE
        findViewById<View>(R.id.anglesSection).visibility = View.GONE
        findViewById<Button>(R.id.btnViewResults).visibility = View.GONE

        // Show info section
        findViewById<View>(R.id.infoSection).visibility = View.VISIBLE

        // Update participant info
        val participantLabel = if (shouldSave) "Participant: $participantId" else "Quick Analysis"
        findViewById<TextView>(R.id.tvParticipantInfo).text = "$participantLabel\nHeight: ${participantHeight / 12}'${participantHeight % 12}\""

        galleryUri?.let {
            findViewById<TextView>(R.id.tvVideoStatus).text = "Video ready for analysis"
        } ?: run {
            findViewById<TextView>(R.id.tvVideoStatus).text = "No video selected"
        }
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnRunAnalysis).setOnClickListener {
            if (!isProcessing && galleryUri != null) {
                runAnalysis()
            }
        }

        findViewById<Button>(R.id.btnViewResults).setOnClickListener {
            updateRunnable?.let { handler.removeCallbacks(it) }
            startActivity(Intent(this, ResultsActivity::class.java))
        }
    }

    private fun runAnalysis() {
        isProcessing = true
        findViewById<Button>(R.id.btnRunAnalysis).isEnabled = false
        findViewById<Button>(R.id.btnRunAnalysis).text = "Processing..."

        // Show progress section
        findViewById<View>(R.id.infoSection).visibility = View.GONE
        findViewById<View>(R.id.progressSection).visibility = View.VISIBLE

        lifecycleScope.launch {
            try {
                // First, create/find patient in database if saving is enabled
                val database = AppDatabase.getDatabase(this@AnalysisActivity)
                
                if (shouldSave) {
                    val patientRepository = PatientRepository(database.patientDao())
                    val patient = withContext(Dispatchers.IO) {
                        patientRepository.findOrCreatePatientByParticipantId(
                            participantId = participantId,
                            height = participantHeight
                        )
                    }
                    currentPatientId = patient.participantId
                    Log.d("AnalysisActivity", "Patient ID: ${patient.participantId}")
                } else {
                    // For quick analysis, we need a dummy patient ID for processing if it uses it internally,
                    // but looking at saveToDatabase, currentPatientId is only used for saving.
                    // ProcVidEmpty doesn't seem to use currentPatientId.
                    Log.d("AnalysisActivity", "Skipping patient creation/lookup (One-off analysis)")
                }

                // Process the video (use app cache dir to avoid permission issues)
                val outputFilePath = "${cacheDir.absolutePath}/edited_video.mp4"
                val outputFile = File(outputFilePath)
                if (outputFile.exists()) {
                    outputFile.delete()
                }

                editedUri = withContext(Dispatchers.IO) {
                    ProcVidEmpty(this@AnalysisActivity, outputFilePath, this@AnalysisActivity)
                }

                // Save video and angle data to database
                saveToDatabase(database, outputFilePath)

                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    showProcessedVideo()
                }

            } catch (e: Exception) {
                Log.e("AnalysisActivity", "Error during analysis: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@AnalysisActivity,
                        "Error processing video: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    findViewById<View>(R.id.progressSection).visibility = View.GONE
                    findViewById<View>(R.id.infoSection).visibility = View.VISIBLE
                    val runButton = findViewById<Button>(R.id.btnRunAnalysis)
                    runButton.isEnabled = true
                    runButton.text = "Run Analysis"
                    isProcessing = false
                }
            }
        }
    }

    private suspend fun saveToDatabase(database: AppDatabase, outputPath: String) {
        if (!shouldSave || currentPatientId == null || editedUri == null) return

        try {
            val resultRepo = AnalysisResultRepository(database.analysisResultDao())
            val signalRepo = SignalDataRepository(database.signalDataDao())

            withContext(Dispatchers.IO) {
                val videoName = galleryUri?.let { uri ->
                    contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                        if (cursor.moveToFirst()) cursor.getString(0) else null
                    }
                } ?: ""
                val features = extractedFeatures
                val score = scoringResult
                val diagnostics = extractionDiagnostics

                // Save analysis result (replaces old Video + GaitScore saves)
                val result = AnalysisResult(
                    patientId = currentPatientId!!,
                    videoFileName = videoName,
                    videoLengthMicroseconds = videoLength,
                    recordedAt = System.currentTimeMillis(),

                    // Scores
                    overallScore = score?.getScoreForDatabase(),
                    aeScore = score?.aeScore,
                    ridgeScore = score?.ridgeScore,
                    pcaScore = score?.pcaScore,

                    // Pipeline metadata
                    stepSignalMode = stepSignalMode,
                    validStrideCount = features?.valid_stride_count ?: 0,
                    qualityFlag = diagnostics?.qualityFlag?.name,

                    // Diagnostics
                    fpsDetected = diagnostics?.fpsDetected,
                    numFramesTotal = diagnostics?.numFramesTotal ?: 0,
                    numFramesValid = diagnostics?.numFramesValid ?: 0,
                    validFrameRate = diagnostics?.validFrameRate,
                    numStepsDetected = diagnostics?.numStepsDetected ?: 0,
                    walkingDirection = diagnostics?.walkingDirection,
                    wasFlipped = diagnostics?.wasFlipped ?: false,

                    // 16 features
                    cadenceSpm = features?.cadence_spm,
                    strideTimeS = features?.stride_time_s,
                    strideTimeCv = features?.stride_time_cv,
                    stepTimeAsymmetry = features?.step_time_asymmetry,
                    strideLengthNorm = features?.stride_length_norm,
                    strideAmpNorm = features?.stride_amp_norm,
                    stepLengthAsymmetry = features?.step_length_asymmetry,
                    kneeLeftRom = features?.knee_left_rom,
                    kneeRightRom = features?.knee_right_rom,
                    kneeLeftMax = features?.knee_left_max,
                    kneeRightMax = features?.knee_right_max,
                    ldjKneeLeft = features?.ldj_knee_left,
                    ldjKneeRight = features?.ldj_knee_right,
                    ldjHip = features?.ldj_hip,
                    trunkLeanStdDeg = features?.trunk_lean_std_deg,
                    interAnkleCv = features?.inter_ankle_cv,

                    stridesJson = extractedStrides?.let { strides ->
                        JSONArray().apply {
                            strides.forEach { s ->
                                put(JSONObject().apply {
                                    put("sf", s.startFrame)
                                    put("ef", s.endFrame)
                                    put("st", s.startTimeS)
                                    put("et", s.endTimeS)
                                    put("s1f", s.step1Frame)
                                    put("s2f", s.step2Frame)
                                    put("s1t", s.step1TimeS)
                                    put("s2t", s.step2TimeS)
                                    put("v", s.isValid)
                                    put("r", s.invalidReason ?: JSONObject.NULL)
                                })
                            }
                        }.toString()
                    },
                    selectedStrideIndicesJson = selectedStrideIndices?.let {
                        JSONArray(it).toString()
                    }
                )
                val resultId = resultRepo.insertResult(result)
                currentResultId = resultId

                // Save per-frame signal data for graph reload
                val signals = extractedSignals
                if (signals != null) {
                    val signalDataList = mutableListOf<SignalData>()
                    val maxFrames = signals.kneeAngleLeft.size

                    for (frame in 0 until maxFrames) {
                        signalDataList.add(SignalData(
                            resultId = resultId,
                            frameNumber = frame,
                            interAnkleDist = signals.interAnkleDist.getOrNull(frame)?.takeIf { !it.isNaN() },
                            kneeAngleLeft = signals.kneeAngleLeft.getOrNull(frame)?.takeIf { !it.isNaN() },
                            kneeAngleRight = signals.kneeAngleRight.getOrNull(frame)?.takeIf { !it.isNaN() },
                            trunkAngle = signals.trunkAngle.getOrNull(frame)?.takeIf { !it.isNaN() },
                            ankleLeftY = signals.ankleLeftY.getOrNull(frame)?.takeIf { !it.isNaN() },
                            ankleRightY = signals.ankleRightY.getOrNull(frame)?.takeIf { !it.isNaN() },
                            hipLeftY = signals.hipLeftY.getOrNull(frame)?.takeIf { !it.isNaN() },
                            hipRightY = signals.hipRightY.getOrNull(frame)?.takeIf { !it.isNaN() },
                            ankleLeftVy = signals.ankleLeftVy.getOrNull(frame)?.takeIf { !it.isNaN() },
                            ankleRightVy = signals.ankleRightVy.getOrNull(frame)?.takeIf { !it.isNaN() },
                            isValid = signals.isValid.getOrNull(frame) ?: true,
                            timestamp = signals.timestamps.getOrNull(frame)?.takeIf { !it.isNaN() }
                        ))
                    }

                    if (signalDataList.isNotEmpty()) {
                        signalRepo.insertSignalDataList(signalDataList)
                    }

                    Log.d("AnalysisActivity", "Saved result ID: $resultId with ${signalDataList.size} signal records")
                } else {
                    Log.d("AnalysisActivity", "Saved result ID: $resultId (no signals)")
                }
            }
        } catch (e: Exception) {
            Log.e("AnalysisActivity", "Error saving to database: ${e.message}", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(this@AnalysisActivity, "Warning: failed to save to database", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showProcessedVideo() {
        isProcessing = false
        findViewById<View>(R.id.progressSection).visibility = View.GONE
        findViewById<View>(R.id.videoSection).visibility = View.VISIBLE
        findViewById<View>(R.id.anglesSection).visibility = View.VISIBLE
        
        findViewById<Button>(R.id.btnViewResults).visibility = View.VISIBLE
        
        findViewById<Button>(R.id.btnRunAnalysis).visibility = View.GONE

        editedUri?.let { uri ->
            videoView.setVideoURI(uri)
            val mediaController = MediaController(this)
            mediaController.setAnchorView(videoView)
            videoView.setMediaController(mediaController)

            videoView.setOnPreparedListener {
                videoView.start()
                setupStrideChart()
                startAngleUpdates()
            }
        }
    }

    private fun setupStrideChart() {
        val signals = extractedSignals ?: return

        val dataSets = mutableListOf<LineDataSet>()
        tvChartLabel.text = "Stride Signal: Inter-Ankle Distance"
        dataSets.add(makeDataSet(signals.interAnkleDist, "Inter-Ankle", Color.CYAN))

        strideChart.data = LineData(dataSets.toList())
        strideChart.description.isEnabled = false
        strideChart.legend.textColor = Color.WHITE
        strideChart.xAxis.textColor = Color.WHITE
        strideChart.axisLeft.textColor = Color.WHITE
        strideChart.axisRight.isEnabled = false
        
        // Draw stride boundaries on the chart
        addStrideBoundaries()
        
        strideChart.invalidate()
    }
    
    /**
     * Add stride boundary lines to the chart.
     * Gold = selected cycles, Green = valid (not selected), Gray = invalid.
     */
    private fun addStrideBoundaries() {
        val strides = extractedStrides ?: return
        val selectedIndices = selectedStrideIndices ?: emptyList()
        
        for ((idx, stride) in strides.withIndex()) {
            val isSelected = selectedIndices.contains(idx)
            val color: Int
            val width: Float
            val style: LimitLine.LimitLabelPosition
            
            when {
                isSelected -> {
                    color = Color.parseColor("#FFD700")  // Gold
                    width = 1.5f
                    style = LimitLine.LimitLabelPosition.RIGHT_TOP
                }
                stride.isValid -> {
                    color = Color.parseColor("#4CAF50")  // Green
                    width = 1f
                    style = LimitLine.LimitLabelPosition.RIGHT_TOP
                }
                else -> {
                    color = Color.parseColor("#888888")  // Gray
                    width = 0.5f
                    style = LimitLine.LimitLabelPosition.RIGHT_TOP
                }
            }
            
            // Start boundary line
            val startLine = LimitLine(stride.startFrame.toFloat()).apply {
                lineColor = color
                lineWidth = width
                enableDashedLine(if (isSelected) 0f else 6f, if (isSelected) 0f else 4f, 0f)
                labelPosition = style
                textColor = color
                textSize = 0f  // No label to keep chart clean
            }
            strideChart.xAxis.addLimitLine(startLine)
            
            // End boundary line
            val endLine = LimitLine(stride.endFrame.toFloat()).apply {
                lineColor = color
                lineWidth = width
                enableDashedLine(if (isSelected) 0f else 6f, if (isSelected) 0f else 4f, 0f)
                textSize = 0f
            }
            strideChart.xAxis.addLimitLine(endLine)
        }
    }

    private fun makeDataSet(data: FloatArray, label: String, color: Int): LineDataSet {
        val entries = mutableListOf<Entry>()
        for (i in data.indices) {
            val v = data[i]
            if (!v.isNaN()) entries.add(Entry(i.toFloat(), v))
        }
        return LineDataSet(entries, label).apply {
            this.color = color
            setDrawCircles(false)
            setDrawValues(false)
            lineWidth = 1.5f
        }
    }

    private fun updateChartCursor(index: Int) {
        if (index == lastCursorIndex) return  // Skip if frame hasn't changed
        lastCursorIndex = index
        if (strideChart.data == null) return
        
        // Remove only the previous cursor line (preserve stride boundaries)
        cursorLimitLine?.let { strideChart.xAxis.removeLimitLine(it) }
        
        val cursor = LimitLine(index.toFloat()).apply {
            lineColor = Color.WHITE
            lineWidth = 1f
        }
        cursorLimitLine = cursor
        strideChart.xAxis.addLimitLine(cursor)
        strideChart.invalidate()
    }

    private fun startAngleUpdates() {
        lastCursorIndex = -1
        val fps = extractionDiagnostics?.fpsDetected ?: 30f
        val msPerFrame = (1000f / fps).toInt().coerceAtLeast(1)
        updateRunnable = object : Runnable {
            override fun run() {
                if (videoView.isPlaying) {
                    val index = videoView.currentPosition / msPerFrame
                    updateAngleDisplay(index)
                    updateChartCursor(index)
                }
                handler.postDelayed(this, msPerFrame.toLong())
            }
        }
        handler.post(updateRunnable!!)
    }

    private fun updateAngleDisplay(index: Int) {
        val signals = extractedSignals
        tvAnkleAngles.text = buildAngleString("Ankle", signals?.ankleAngleLeft, signals?.ankleAngleRight, index)
        tvKneeAngles.text = buildAngleString("Knee", signals?.kneeAngleLeft, signals?.kneeAngleRight, index)
        tvHipAngles.text = buildAngleString("Hip", signals?.hipAngleLeft, signals?.hipAngleRight, index)
        tvTorsoAngle.text = formatSingleAngle("Torso", signals?.trunkAngle, index)
    }

    private fun buildAngleString(name: String, leftAngles: FloatArray?, rightAngles: FloatArray?, index: Int): String {
        val leftVal = leftAngles?.getOrNull(index)
        val rightVal = rightAngles?.getOrNull(index)
        val left = if (leftVal != null && !leftVal.isNaN()) String.format("%.1f", leftVal) else "--"
        val right = if (rightVal != null && !rightVal.isNaN()) String.format("%.1f", rightVal) else "--"
        return "L $name: $left°\nR $name: $right°"
    }
    
    private fun formatSingleAngle(name: String, angles: FloatArray?, index: Int): String {
        val value = angles?.getOrNull(index)
        return if (value != null && !value.isNaN()) {
            "$name: ${String.format("%.1f", value)}°"
        } else "$name: --"
    }

    override fun onPause() {
        super.onPause()
        updateRunnable?.let { handler.removeCallbacks(it) }
    }

    override fun onDestroy() {
        super.onDestroy()
        updateRunnable?.let { handler.removeCallbacks(it) }
    }
}
