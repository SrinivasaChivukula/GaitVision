package GaitVision.com.ui

import android.app.AlertDialog
import android.content.Intent
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.charts.LineChart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.GaitScore
import GaitVision.com.data.repository.GaitScoreRepository
import GaitVision.com.plotLineGraph
import GaitVision.com.editedUri
import GaitVision.com.leftAnkleAngles
import GaitVision.com.rightAnkleAngles
import GaitVision.com.leftKneeAngles
import GaitVision.com.rightKneeAngles
import GaitVision.com.leftHipAngles
import GaitVision.com.rightHipAngles
import GaitVision.com.torsoAngles
import GaitVision.com.participantId
import GaitVision.com.currentPatientId
import GaitVision.com.currentVideoId
import GaitVision.com.extractedFeatures
import GaitVision.com.extractionDiagnostics
import GaitVision.com.scoringResult
import GaitVision.com.gait.GaitCsvExporter
import GaitVision.com.galleryUri
import java.io.File
import java.io.FileOutputStream

class ResultsActivity : AppCompatActivity() {

    private lateinit var tvGaitScore: TextView
    private lateinit var tvScoreLabel: TextView
    private lateinit var tvAeScore: TextView
    private lateinit var tvRidgeScore: TextView
    private lateinit var tvPcaScore: TextView
    private lateinit var hipChart: LineChart
    private lateinit var kneeChart: LineChart
    private lateinit var ankleChart: LineChart
    private lateinit var torsoChart: LineChart
    private lateinit var btnSelectGraph: Button

    private var calculatedScore: Double = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_results)

        initializeViews()
        setupButtons()
        
        // Calculate and display gait score
        calculateGaitScore()
        
        // Setup charts
        setupCharts()
    }

    private fun initializeViews() {
        tvGaitScore = findViewById(R.id.tvGaitScore)
        tvScoreLabel = findViewById(R.id.tvScoreLabel)
        tvAeScore = findViewById(R.id.tvAeScore)
        tvRidgeScore = findViewById(R.id.tvRidgeScore)
        tvPcaScore = findViewById(R.id.tvPcaScore)
        hipChart = findViewById(R.id.lineChartHip)
        kneeChart = findViewById(R.id.lineChartKnee)
        ankleChart = findViewById(R.id.lineChartAnkle)
        torsoChart = findViewById(R.id.lineChartTorso)
        btnSelectGraph = findViewById(R.id.btnSelectGraph)
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnBack).setOnClickListener {
            finish()
        }

        findViewById<Button>(R.id.btnMainMenu).setOnClickListener {
            // Go back to dashboard and clear the back stack
            val intent = Intent(this, DashboardActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()
        }

        findViewById<Button>(R.id.btnExportCsv).setOnClickListener {
            exportCsvFiles()
        }

        findViewById<Button>(R.id.btnSignalsDashboard).setOnClickListener {
            val intent = Intent(this, SignalsDashboardActivity::class.java)
            startActivity(intent)
        }

        btnSelectGraph.setOnClickListener {
            showGraphPopup()
        }
    }

    private fun calculateGaitScore() {
        // PC Pipeline ONLY - no legacy fallback
        val pcFeatures = extractedFeatures
        val pcScore = scoringResult
        val diagnostics = extractionDiagnostics
        
        Log.d("ResultsActivity", "calculateGaitScore: features=${pcFeatures != null}, score=${pcScore != null}, diagnostics=${diagnostics != null}")
        
        // Check if PC pipeline succeeded
        if (pcFeatures != null && pcScore != null && pcFeatures.valid_stride_count > 0) {
            // SUCCESS: Use PC pipeline scores
            calculatedScore = pcScore.getScoreForDatabase()
            tvGaitScore.text = calculatedScore.toLong().toString()
            tvScoreLabel.text = "${getScoreLabel(calculatedScore)}\n(${pcFeatures.valid_stride_count} strides, ${String.format("%.1f", pcFeatures.cadence_spm)} spm)"
            
            // Display all 3 model scores
            tvAeScore.text = if (!pcScore.aeScore.isNaN()) pcScore.aeScore.toLong().toString() else "--"
            tvRidgeScore.text = if (!pcScore.ridgeScore.isNaN()) pcScore.ridgeScore.toLong().toString() else "--"
            tvPcaScore.text = if (!pcScore.pcaScore.isNaN()) pcScore.pcaScore.toLong().toString() else "--"
            
            // Color individual scores
            tvAeScore.setTextColor(getScoreColor(pcScore.aeScore))
            tvRidgeScore.setTextColor(getScoreColor(pcScore.ridgeScore))
            tvPcaScore.setTextColor(getScoreColor(pcScore.pcaScore))
            
            // Color primary score
            val scoreColor = when {
                calculatedScore >= 80 -> "#4CAF50"
                calculatedScore >= 60 -> "#FF9800"
                else -> "#F44336"
            }
            tvGaitScore.setTextColor(android.graphics.Color.parseColor(scoreColor))
            
            Log.d("ResultsActivity", "PC pipeline SUCCESS - AE: ${pcScore.aeScore}, Ridge: ${pcScore.ridgeScore}, PCA: ${pcScore.pcaScore}")
            
            saveGaitScoreToDatabase()
            
        } else {
            // FAILED: Show error with diagnostic info
            tvGaitScore.text = "--"
            tvAeScore.text = "--"
            tvRidgeScore.text = "--"
            tvPcaScore.text = "--"
            
            val errorMsg = when {
                pcFeatures == null && diagnostics != null -> 
                    "Extraction failed: ${diagnostics.qualityFlag}\n${diagnostics.rejectionReasons.firstOrNull() ?: ""}"
                pcFeatures == null -> 
                    "Feature extraction not run"
                pcFeatures.valid_stride_count == 0 -> 
                    "No valid gait cycles detected"
                pcScore == null -> 
                    "Scoring failed to initialize"
                else -> 
                    "Unknown error"
            }
            
            tvScoreLabel.text = errorMsg
            tvGaitScore.setTextColor(android.graphics.Color.parseColor("#F44336"))
            
            Log.e("ResultsActivity", "PC pipeline FAILED: $errorMsg")
            Log.e("ResultsActivity", "  pcFeatures=$pcFeatures, pcScore=$pcScore")
            if (diagnostics != null) {
                Log.e("ResultsActivity", "  diagnostics: flag=${diagnostics.qualityFlag}, reasons=${diagnostics.rejectionReasons}")
            }
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
            score >= 80 -> android.graphics.Color.parseColor("#4CAF50")  // Green
            score >= 60 -> android.graphics.Color.parseColor("#FF9800")  // Orange
            else -> android.graphics.Color.parseColor("#F44336")         // Red
        }
    }

    private fun saveGaitScoreToDatabase() {
        if (currentPatientId == null || currentVideoId == null) return

        val features = extractedFeatures ?: return
        
        lifecycleScope.launch(Dispatchers.IO) {
            val database = AppDatabase.getDatabase(this@ResultsActivity)
            val gaitScoreRepository = GaitScoreRepository(database.gaitScoreDao())

            // Use PC pipeline features for subscores
            val gaitScore = GaitScore(
                patientId = currentPatientId!!,
                videoId = currentVideoId!!,
                overallScore = calculatedScore,
                leftKneeScore = features.knee_left_rom.toDouble(),
                rightKneeScore = features.knee_right_rom.toDouble(),
                leftHipScore = features.ldj_hip.toDouble(),
                rightHipScore = null,
                torsoScore = features.trunk_lean_std_deg.toDouble(),
                recordedAt = System.currentTimeMillis()
            )

            gaitScoreRepository.insertGaitScore(gaitScore)
            Log.d("ResultsActivity", "Saved gait score: ${gaitScore.overallScore}")
        }
    }

    private fun setupCharts() {
        // Plot all charts
        if (leftKneeAngles.isNotEmpty() || rightKneeAngles.isNotEmpty()) {
            plotLineGraph(kneeChart, leftKneeAngles, rightKneeAngles, "Left Knee", "Right Knee")
        }
        if (leftAnkleAngles.isNotEmpty() || rightAnkleAngles.isNotEmpty()) {
            plotLineGraph(ankleChart, leftAnkleAngles, rightAnkleAngles, "Left Ankle", "Right Ankle")
        }
        if (leftHipAngles.isNotEmpty() || rightHipAngles.isNotEmpty()) {
            plotLineGraph(hipChart, leftHipAngles, rightHipAngles, "Left Hip", "Right Hip")
        }
        if (torsoAngles.isNotEmpty()) {
            plotLineGraph(torsoChart, torsoAngles, torsoAngles, "Torso", "Torso")
        }

        // Initially show knee chart
        showChart("KNEE")
    }

    private fun showGraphPopup() {
        val popup = PopupMenu(this, btnSelectGraph)
        popup.menu.add(0, 1, 0, "Knee Graph")
        popup.menu.add(0, 2, 1, "Hip Graph")
        popup.menu.add(0, 3, 2, "Ankle Graph")
        popup.menu.add(0, 4, 3, "Torso Graph")

        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                1 -> showChart("KNEE")
                2 -> showChart("HIP")
                3 -> showChart("ANKLE")
                4 -> showChart("TORSO")
            }
            btnSelectGraph.text = item.title
            true
        }
        popup.show()
    }

    private fun showChart(chartType: String) {
        hipChart.visibility = View.INVISIBLE
        kneeChart.visibility = View.INVISIBLE
        ankleChart.visibility = View.INVISIBLE
        torsoChart.visibility = View.INVISIBLE

        when (chartType) {
            "HIP" -> {
                hipChart.visibility = View.VISIBLE
                btnSelectGraph.text = "Hip Graph"
            }
            "KNEE" -> {
                kneeChart.visibility = View.VISIBLE
                btnSelectGraph.text = "Knee Graph"
            }
            "ANKLE" -> {
                ankleChart.visibility = View.VISIBLE
                btnSelectGraph.text = "Ankle Graph"
            }
            "TORSO" -> {
                torsoChart.visibility = View.VISIBLE
                btnSelectGraph.text = "Torso Graph"
            }
        }
    }

    private fun exportCsvFiles() {
        val fileData = listOf(
            leftHipAngles,
            rightHipAngles,
            leftKneeAngles,
            rightKneeAngles,
            leftAnkleAngles,
            rightAnkleAngles,
            torsoAngles
        )

        val angleNames = listOf(
            "LeftHip",
            "RightHip",
            "LeftKnee",
            "RightKnee",
            "LeftAnkle",
            "RightAnkle",
            "Torso"
        )

        try {
            // Export wireframe angle files (for visualization)
            for (i in fileData.indices) {
                val fileName = "${participantId}_${angleNames[i]}.csv"
                writeToFile(fileName, fileData[i])
            }
            
            // Export PC-style gait features CSV
            val diagnostics = extractionDiagnostics
            if (diagnostics != null) {
                val videoName = galleryUri?.lastPathSegment ?: "unknown"
                val csvPath = GaitCsvExporter.exportToCSV(
                    context = this,
                    features = extractedFeatures,
                    diagnostics = diagnostics,
                    score = scoringResult,
                    participantId = participantId.toString(),
                    videoName = videoName
                )
                if (csvPath != null) {
                    Log.d("ResultsActivity", "PC-style CSV exported to: $csvPath")
                }
            }

            // Rename edited video
            renameEditedVideo()

            // Build export message
            val message = StringBuilder()
            message.appendLine("CSV files saved to Documents as ${participantId}_GraphName.csv")
            if (extractedFeatures != null) {
                message.appendLine("\nGait features CSV exported with ${extractedFeatures!!.valid_stride_count} valid strides")
            }
            message.appendLine("\nVideo saved to Movies as ${participantId}_video.mp4")

            AlertDialog.Builder(this)
                .setTitle("Export Successful")
                .setMessage(message.toString())
                .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
                .show()

        } catch (e: Exception) {
            Log.e("ResultsActivity", "Error exporting: ${e.message}", e)
            Toast.makeText(this, "Error exporting files: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun writeToFile(fileName: String, fileData: List<Float>) {
        val fileDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val outputFile = File(fileDirectory, fileName)

        FileOutputStream(outputFile).use { output ->
            output.write("Frame #,Angle\n".toByteArray())
            for (i in fileData.indices) {
                output.write("$i,${fileData[i]}\n".toByteArray())
            }
        }

        MediaScannerConnection.scanFile(this, arrayOf(outputFile.absolutePath), null, null)
    }

    private fun renameEditedVideo() {
        val vidName = "${participantId}_video.mp4"
        val oldFilePath = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
            "edited_video.mp4"
        )
        val newFilePath = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
            vidName
        )

        if (oldFilePath.exists()) {
            oldFilePath.renameTo(newFilePath)
            editedUri = Uri.fromFile(newFilePath)
            MediaScannerConnection.scanFile(this, arrayOf(newFilePath.absolutePath), null, null)
        }
    }

}
