package GaitVision.com.ui

import android.content.Intent
import android.graphics.Color
import android.media.MediaScannerConnection
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.MediaController
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.constraintlayout.widget.ConstraintSet
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AngleData
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.Video
import GaitVision.com.data.repository.AngleDataRepository
import GaitVision.com.data.repository.PatientRepository
import GaitVision.com.data.repository.VideoRepository
import GaitVision.com.ProcVidEmpty
import GaitVision.com.galleryUri
import GaitVision.com.editedUri
import GaitVision.com.frameList
import GaitVision.com.extractedSignals
import GaitVision.com.participantId
import GaitVision.com.participantHeight
import GaitVision.com.currentPatientId
import GaitVision.com.currentVideoId
import GaitVision.com.videoLength
import GaitVision.com.extractedFeatures
import java.io.File

class AnalysisActivity : AppCompatActivity() {

    private val handler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null
    private var isProcessing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_analysis)

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
        findViewById<TextView>(R.id.tvParticipantInfo).text = "Participant: $participantId\nHeight: ${participantHeight / 12}'${participantHeight % 12}\""

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

        findViewById<Button>(R.id.btnBack).setOnClickListener {
            updateRunnable?.let { handler.removeCallbacks(it) }
            finish()
        }

        // Angle selection popup
        findViewById<Button>(R.id.btnSelectAngle).setOnClickListener {
            showAnglePopup()
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
                // First, create/find patient in database
                val database = AppDatabase.getDatabase(this@AnalysisActivity)
                val patientRepository = PatientRepository(database.patientDao())

                val patient = withContext(Dispatchers.IO) {
                    patientRepository.findOrCreatePatientByParticipantId(
                        participantId = participantId,
                        height = participantHeight
                    )
                }
                currentPatientId = patient.participantId
                Log.d("AnalysisActivity", "Patient ID: ${patient.participantId}")

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
        if (currentPatientId == null || editedUri == null) return

        try {
            val videoRepository = VideoRepository(database.videoDao())
            val angleDataRepository = AngleDataRepository(database.angleDataDao())

            withContext(Dispatchers.IO) {
                val originalPath = galleryUri?.path ?: galleryUri?.toString() ?: ""
                val editedPath = editedUri?.path ?: editedUri?.toString() ?: ""
                // Use PC pipeline stride length if available
                val strideLengthAvg = extractedFeatures?.stride_length_norm?.toDouble() ?: 0.0

                val video = Video(
                    patientId = currentPatientId!!,
                    originalVideoPath = originalPath,
                    editedVideoPath = editedPath,
                    recordedAt = System.currentTimeMillis(),
                    strideLengthAvg = strideLengthAvg,
                    videoLengthMicroseconds = videoLength
                )
                val videoId = videoRepository.insertVideo(video)
                currentVideoId = videoId

                // Save angle data for each frame from Signals
                val signals = extractedSignals
                val angleDataList = mutableListOf<AngleData>()
                
                if (signals != null) {
                    val maxFrames = signals.kneeAngleLeft.size
                    for (frameNumber in 0 until maxFrames) {
                        val angleData = AngleData(
                            videoId = videoId,
                            frameNumber = frameNumber,
                            leftAnkleAngle = signals.ankleAngleLeft.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            rightAnkleAngle = signals.ankleAngleRight.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            leftKneeAngle = signals.kneeAngleLeft.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            rightKneeAngle = signals.kneeAngleRight.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            leftHipAngle = signals.hipAngleLeft.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            rightHipAngle = signals.hipAngleRight.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            torsoAngle = signals.trunkAngle.getOrNull(frameNumber)?.takeIf { !it.isNaN() },
                            strideAngle = signals.strideAngle.getOrNull(frameNumber)?.takeIf { !it.isNaN() }
                        )
                        angleDataList.add(angleData)
                    }
                }

                if (angleDataList.isNotEmpty()) {
                    angleDataRepository.insertAngleDataList(angleDataList)
                }

                Log.d("AnalysisActivity", "Saved video ID: $videoId with ${angleDataList.size} angle records")
            }
        } catch (e: Exception) {
            Log.e("AnalysisActivity", "Error saving to database: ${e.message}", e)
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
            MediaScannerConnection.scanFile(this, arrayOf(uri.path), null) { path, _ ->
                Log.d("AnalysisActivity", "Scanned: $path")
            }

            findViewById<VideoView>(R.id.videoView).setVideoURI(uri)
            val mediaController = MediaController(this)
            mediaController.setAnchorView(findViewById(R.id.videoView))
            findViewById<VideoView>(R.id.videoView).setMediaController(mediaController)

            findViewById<VideoView>(R.id.videoView).setOnPreparedListener {
                findViewById<VideoView>(R.id.videoView).start()
                startAngleUpdates("ALL ANGLES")
            }
        }
    }

    private fun showAnglePopup() {
        val popup = PopupMenu(this, findViewById<Button>(R.id.btnSelectAngle))
        popup.menu.add(0, 1, 0, "All Angles")
        popup.menu.add(0, 2, 1, "Hip Angles")
        popup.menu.add(0, 3, 2, "Knee Angles")
        popup.menu.add(0, 4, 3, "Ankle Angles")
        popup.menu.add(0, 5, 4, "Torso Angle")

        popup.setOnMenuItemClickListener { item ->
            updateRunnable?.let { handler.removeCallbacks(it) }
            when (item.itemId) {
                1 -> startAngleUpdates("ALL ANGLES")
                2 -> startAngleUpdates("HIP ANGLES")
                3 -> startAngleUpdates("KNEE ANGLES")
                4 -> startAngleUpdates("ANKLE ANGLES")
                5 -> startAngleUpdates("TORSO ANGLE")
            }
            findViewById<Button>(R.id.btnSelectAngle).text = item.title
            true
        }
        popup.show()
    }

    private fun startAngleUpdates(angleType: String) {
        updateRunnable = object : Runnable {
            override fun run() {
                if (findViewById<VideoView>(R.id.videoView).isPlaying) {
                    val currentPos = findViewById<VideoView>(R.id.videoView).currentPosition
                    val interval = 33
                    val index = currentPos / interval

                    updateAngleDisplay(angleType, index)
                }
                handler.postDelayed(this, 33)
            }
        }
        handler.post(updateRunnable!!)
    }

    private fun updateAngleDisplay(angleType: String, index: Int) {
        val signals = extractedSignals
        
        when (angleType) {
            "ALL ANGLES" -> {
                findViewById<TextView>(R.id.tvAnkleAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvKneeAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvHipAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvTorsoAngle).visibility = View.VISIBLE

                findViewById<TextView>(R.id.tvAnkleAngles).text = buildAngleString("Ankle", signals?.ankleAngleLeft, signals?.ankleAngleRight, index)
                findViewById<TextView>(R.id.tvKneeAngles).text = buildAngleString("Knee", signals?.kneeAngleLeft, signals?.kneeAngleRight, index)
                findViewById<TextView>(R.id.tvHipAngles).text = buildAngleString("Hip", signals?.hipAngleLeft, signals?.hipAngleRight, index)
                findViewById<TextView>(R.id.tvTorsoAngle).text = formatSingleAngle("Torso", signals?.trunkAngle, index)
            }
            "HIP ANGLES" -> {
                findViewById<TextView>(R.id.tvAnkleAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvKneeAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvHipAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvTorsoAngle).visibility = View.GONE
                findViewById<TextView>(R.id.tvHipAngles).text = buildAngleString("Hip", signals?.hipAngleLeft, signals?.hipAngleRight, index)
            }
            "KNEE ANGLES" -> {
                findViewById<TextView>(R.id.tvAnkleAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvKneeAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvHipAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvTorsoAngle).visibility = View.GONE
                findViewById<TextView>(R.id.tvKneeAngles).text = buildAngleString("Knee", signals?.kneeAngleLeft, signals?.kneeAngleRight, index)
            }
            "ANKLE ANGLES" -> {
                findViewById<TextView>(R.id.tvAnkleAngles).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvKneeAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvHipAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvTorsoAngle).visibility = View.GONE
                findViewById<TextView>(R.id.tvAnkleAngles).text = buildAngleString("Ankle", signals?.ankleAngleLeft, signals?.ankleAngleRight, index)
            }
            "TORSO ANGLE" -> {
                findViewById<TextView>(R.id.tvAnkleAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvKneeAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvHipAngles).visibility = View.GONE
                findViewById<TextView>(R.id.tvTorsoAngle).visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvTorsoAngle).text = formatSingleAngle("Torso", signals?.trunkAngle, index)
            }
        }
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
