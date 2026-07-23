package GaitVision.com.ui

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.MediaController
import android.widget.ProgressBar
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
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.repository.PatientRepository
import GaitVision.com.ProcVidEmpty
import GaitVision.com.AnalysisSession
import GaitVision.com.persistCurrentSession
import java.io.File
import kotlin.math.roundToInt

class AnalysisActivity : BaseActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null
    private var isProcessing = false
    private var shouldSave = true
    private var lastCursorIndex = -1
    private var playbackBasePositionMs = 0
    private var playbackClockStartedAtMs = 0L

    // Cached view references (resolved once, used every frame)
    private lateinit var videoView: VideoView
    private lateinit var cardVideoPreview: View
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
        cardVideoPreview = findViewById(R.id.cardVideoPreview)
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
            AnalysisSession.galleryUri = uri
        }

        setupInitialUI()
        setupButtons()

        // Check if we have a video to process
        if (AnalysisSession.galleryUri == null) {
            Toast.makeText(this, "No video selected. Please go back.", Toast.LENGTH_SHORT).show()
            // Card stays visible but click listener guards against null URI
        }

        // If already processed, show the video
        if (AnalysisSession.editedUri != null) {
            showProcessedVideo()
        }
    }

    private fun setupInitialUI() {
        // Hide processing UI initially
        findViewById<View>(R.id.progressSection).visibility = View.GONE
        findViewById<View>(R.id.videoSection).visibility = View.GONE
        findViewById<View>(R.id.anglesSection).visibility = View.GONE
        findViewById<View>(R.id.cardViewResults).visibility = View.GONE

        // Show info section
        findViewById<View>(R.id.infoSection).visibility = View.VISIBLE

        // Update participant info
        val participantLabel = if (shouldSave) "Participant: ${AnalysisSession.participantId}" else "Quick Analysis"
        findViewById<TextView>(R.id.tvParticipantInfo).text = "$participantLabel\nHeight: ${AnalysisSession.participantHeight / 12}'${AnalysisSession.participantHeight % 12}\""

        AnalysisSession.galleryUri?.let {
            findViewById<TextView>(R.id.tvVideoStatus).text = "Video ready for analysis"
        } ?: run {
            findViewById<TextView>(R.id.tvVideoStatus).text = "No video selected"
        }
    }

    private fun setupButtons() {
        findViewById<LinearLayout>(R.id.btnRunAnalysis).setOnClickListener {
            if (!isProcessing && AnalysisSession.galleryUri != null) {
                runAnalysis()
            }
        }

        findViewById<LinearLayout>(R.id.btnViewResults).setOnClickListener {
            updateRunnable?.let { handler.removeCallbacks(it) }
            startActivity(Intent(this, ResultsActivity::class.java))
        }
    }

    private fun runAnalysis() {
        isProcessing = true
        AnalysisSession.editedUri = null
        // Hide the run-analysis card; progress section takes over
        findViewById<View>(R.id.cardRunAnalysis).visibility = View.GONE

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
                            participantId = AnalysisSession.participantId,
                            height = AnalysisSession.participantHeight
                        )
                    }
                    AnalysisSession.currentPatientId = patient.participantId
                    Log.d("AnalysisActivity", "Patient ID: ${patient.participantId}")
                } else {
                    // For quick analysis, we need a dummy patient ID for processing if it uses it internally,
                    // but looking at saveToDatabase, currentPatientId is only used for saving.
                    // ProcVidEmpty doesn't seem to use currentPatientId.
                    Log.d("AnalysisActivity", "Skipping patient creation/lookup (One-off analysis)")
                }

                // Match the known-good playback path from the UNT app.
                // VideoView behaves more reliably here with a stable cache file
                // that is deleted before each new encode.
                val outputFilePath = "${cacheDir.absolutePath}/edited_video.mp4"
                val outputFile = File(outputFilePath)
                if (outputFile.exists()) {
                    outputFile.delete()
                }

                AnalysisSession.editedUri = withContext(Dispatchers.IO) {
                    ProcVidEmpty(this@AnalysisActivity, outputFilePath) { stage, percent ->
                        // pipeline runs on IO, view writes need Main
                        runOnUiThread { updateProcessingProgress(stage, percent) }
                    }
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
                    // Restore run button card on error
                    findViewById<View>(R.id.cardRunAnalysis).visibility = View.VISIBLE
                    isProcessing = false
                }
            }
        }
    }

    private var processingUiInitialized = false

    // Drives the splitting progress bar from pipeline callbacks. The second
    // bar (CreationText/VideoCreation) stayed hidden in the old flow too; it
    // was only ever shown during the disabled ROI retry path.
    private fun updateProcessingProgress(stage: String, percent: Int) {
        val splittingText = findViewById<TextView>(R.id.SplittingText)
        val splittingBar = findViewById<ProgressBar>(R.id.splittingBar)
        val splittingValue = findViewById<TextView>(R.id.splittingProgressValue)

        when (stage) {
            "processing" -> {
                if (!processingUiInitialized) {
                    splittingText.text = "Processing..."
                    splittingText.visibility = View.VISIBLE
                    splittingBar.visibility = View.VISIBLE
                    splittingValue.visibility = View.VISIBLE
                    findViewById<TextView>(R.id.CreationText).visibility = View.GONE
                    findViewById<ProgressBar>(R.id.VideoCreation).visibility = View.GONE
                    findViewById<TextView>(R.id.CreatingProgressValue).visibility = View.GONE
                    processingUiInitialized = true
                }
                splittingBar.progress = percent
                splittingValue.text = " $percent%"
            }
            "done" -> {
                splittingText.visibility = View.GONE
                splittingBar.visibility = View.GONE
                splittingValue.visibility = View.GONE
                processingUiInitialized = false
            }
        }
    }

    private suspend fun saveToDatabase(database: AppDatabase, outputPath: String) {
        if (!shouldSave || AnalysisSession.currentPatientId == null || AnalysisSession.editedUri == null) return
        try {
            persistCurrentSession(
                context = this@AnalysisActivity,
                db = database,
                patientId = AnalysisSession.currentPatientId!!
            )
        } catch (e: Exception) {
            Log.e("AnalysisActivity", "DB save failed: ${e.message}", e)
            withContext(Dispatchers.Main) {
                Toast.makeText(this@AnalysisActivity, "Warning: failed to save to database", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showProcessedVideo() {
        isProcessing = false
        updateRunnable?.let { handler.removeCallbacks(it) }
        updateRunnable = null
        lastCursorIndex = -1
        findViewById<View>(R.id.progressSection).visibility = View.GONE
        findViewById<View>(R.id.videoSection).visibility = View.VISIBLE
        findViewById<View>(R.id.anglesSection).visibility = View.VISIBLE
        
        findViewById<View>(R.id.cardViewResults).visibility = View.VISIBLE
        // Keep run analysis card hidden (processing is done)
        findViewById<View>(R.id.cardRunAnalysis).visibility = View.GONE

        val previewUri = AnalysisSession.editedUri ?: AnalysisSession.galleryUri
        previewUri?.let { uri ->
            videoView.stopPlayback()
            val mediaController = MediaController(this)
            mediaController.setAnchorView(videoView)
            videoView.setMediaController(mediaController)

            videoView.setOnPreparedListener { player ->
                player.isLooping = true
                adjustVideoPreviewSize(player.videoWidth, player.videoHeight)
                setupStrideChart()
                startAngleUpdates()
                videoView.requestFocus()
                videoView.post {
                    videoView.seekTo(0)
                    videoView.start()
                    resetPlaybackClock()
                    Log.d("AnalysisActivity", "Video playback started: playing=${videoView.isPlaying}, duration=${videoView.duration}ms")
                }
            }
            videoView.setOnCompletionListener {
                videoView.seekTo(0)
                videoView.start()
                resetPlaybackClock()
            }
            videoView.setOnErrorListener { _, what, extra ->
                Log.e("AnalysisActivity", "Video playback error: what=$what extra=$extra uri=$uri")
                Toast.makeText(this, "Could not play processed video", Toast.LENGTH_SHORT).show()
                true
            }
            videoView.setOnClickListener {
                if (videoView.isPlaying) {
                    playbackBasePositionMs = currentPlaybackPositionMs()
                    videoView.pause()
                } else {
                    playbackClockStartedAtMs = SystemClock.elapsedRealtime()
                    videoView.start()
                }
            }
            videoView.setVideoURI(uri)
        }
    }

    private fun adjustVideoPreviewSize(videoWidth: Int, videoHeight: Int) {
        if (videoWidth <= 0 || videoHeight <= 0) return
        val horizontalPaddingPx = (32f * resources.displayMetrics.density).toInt()
        val availableWidth = (resources.displayMetrics.widthPixels - horizontalPaddingPx).coerceAtLeast(1)
        val targetHeight = (availableWidth.toFloat() * videoHeight / videoWidth).toInt()
        val minHeight = (200f * resources.displayMetrics.density).toInt()
        val maxHeight = (360f * resources.displayMetrics.density).toInt()
        val clampedHeight = targetHeight.coerceIn(minHeight, maxHeight)
        val params: ViewGroup.LayoutParams = cardVideoPreview.layoutParams
        if (params.height != clampedHeight) {
            params.height = clampedHeight
            cardVideoPreview.layoutParams = params
        }
    }

    private fun setupStrideChart() {
        val signals = AnalysisSession.extractedSignals ?: return
        val mode = AnalysisSession.stepSignalMode ?: return

        val dataSets = mutableListOf<LineDataSet>()

        when (mode) {
            "inter_ankle" -> {
                tvChartLabel.text = "Stride Signal: Inter-Ankle Distance"
                dataSets.add(makeDataSet(signals.interAnkleDist, "Inter-Ankle", Color.CYAN))
            }
            "max_ankle_vy" -> {
                tvChartLabel.text = "Stride Signal: Ankle Velocity"
                dataSets.add(makeDataSet(signals.ankleLeftVy, "Ankle L Vy", Color.CYAN))
                dataSets.add(makeDataSet(signals.ankleRightVy, "Ankle R Vy", Color.MAGENTA))
            }
            "min_knee_angle" -> {
                tvChartLabel.text = "Stride Signal: Knee Angle"
                dataSets.add(makeDataSet(signals.kneeAngleLeft, "Knee L", Color.parseColor("#4CAF50")))
                dataSets.add(makeDataSet(signals.kneeAngleRight, "Knee R", Color.parseColor("#FF9800")))
            }
        }

        strideChart.data = LineData(dataSets.toList())
        strideChart.description.isEnabled = false
        strideChart.legend.textColor = Color.WHITE
        strideChart.xAxis.textColor = Color.WHITE
        strideChart.axisLeft.textColor = Color.WHITE
        strideChart.axisRight.isEnabled = false
        strideChart.invalidate()
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
        strideChart.xAxis.removeAllLimitLines()
        val cursor = LimitLine(index.toFloat()).apply {
            lineColor = Color.WHITE
            lineWidth = 1f
        }
        strideChart.xAxis.addLimitLine(cursor)
        strideChart.invalidate()
    }

    private fun startAngleUpdates() {
        lastCursorIndex = -1
        val fps = AnalysisSession.extractionDiagnostics?.fpsDetected
            ?.takeIf { it.isFinite() && it > 0f }
            ?: 30f
        val msPerFrame = (1000f / fps).roundToInt().coerceAtLeast(1)
        val updatePeriodMs = (msPerFrame / 2).coerceIn(16, 50)
        updateRunnable = object : Runnable {
            override fun run() {
                if (videoView.isPlaying) {
                    val index = currentPlaybackPositionMs() / msPerFrame
                    updateAngleDisplay(index)
                    updateChartCursor(index)
                }
                handler.postDelayed(this, updatePeriodMs.toLong())
            }
        }
        handler.post(updateRunnable!!)
    }

    private fun resetPlaybackClock() {
        playbackBasePositionMs = videoView.currentPosition
        playbackClockStartedAtMs = SystemClock.elapsedRealtime()
    }

    private fun currentPlaybackPositionMs(): Int {
        if (!videoView.isPlaying) return videoView.currentPosition
        val duration = videoView.duration.takeIf { it > 0 } ?: Int.MAX_VALUE
        val elapsedMs = (SystemClock.elapsedRealtime() - playbackClockStartedAtMs).toInt()
        return (playbackBasePositionMs + elapsedMs).coerceIn(0, duration)
    }

    private fun lastFrameIndex(): Int {
        val signals = AnalysisSession.extractedSignals ?: return 0
        return (signals.timestamps.size - 1).coerceAtLeast(0)
    }

    private fun updateAngleDisplay(index: Int) {
        val signals = AnalysisSession.extractedSignals
        tvAnkleAngles.text = buildAngleString("Ankle", signals?.ankleAngleLeft, signals?.ankleAngleRight, index)
        tvKneeAngles.text = buildKneeString(signals, index)
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

    private fun buildKneeString(signals: GaitVision.com.gait.Signals?, index: Int): String {
        val angleText = buildAngleString("Knee", signals?.kneeAngleLeft, signals?.kneeAngleRight, index)
        val leftOffset = formatFrontalKneeOffset(signals?.frontalKneeOffsetLeft, index)
        val rightOffset = formatFrontalKneeOffset(signals?.frontalKneeOffsetRight, index)
        return "$angleText\nFront offset L/R: $leftOffset / $rightOffset"
    }

    private fun formatFrontalKneeOffset(values: FloatArray?, index: Int): String {
        val value = values?.getOrNull(index)
        return if (value != null && !value.isNaN()) {
            "${String.format("%+.1f", value * 100f)}%"
        } else "--"
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
        updateRunnable = null
        if (::videoView.isInitialized && videoView.isPlaying) {
            playbackBasePositionMs = currentPlaybackPositionMs()
            videoView.pause()
        }
    }

    override fun onResume() {
        super.onResume()
        if (::videoView.isInitialized && AnalysisSession.editedUri != null) {
            videoView.post {
                if (!videoView.isPlaying) {
                    playbackClockStartedAtMs = SystemClock.elapsedRealtime()
                    videoView.start()
                }
                if (updateRunnable == null) {
                    startAngleUpdates()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        updateRunnable?.let { handler.removeCallbacks(it) }
        updateRunnable = null
    }
}
