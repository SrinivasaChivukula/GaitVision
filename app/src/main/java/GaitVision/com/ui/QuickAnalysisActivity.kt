package GaitVision.com.ui

import android.content.Intent
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import GaitVision.com.R
import GaitVision.com.galleryUri
import GaitVision.com.editedUri
import GaitVision.com.frameList
import GaitVision.com.participantId
import GaitVision.com.participantHeight
import GaitVision.com.currentPatientId
import GaitVision.com.currentVideoId
import GaitVision.com.poseFrames
import GaitVision.com.extractedFeatures
import GaitVision.com.extractionDiagnostics
import GaitVision.com.scoringResult
import GaitVision.com.extractedSignals
import GaitVision.com.extractedStrides
import GaitVision.com.selectedStrideIndices
import GaitVision.com.stepSignalMode

class QuickAnalysisActivity : AppCompatActivity() {

    private lateinit var videoPreview: ImageView
    private lateinit var etFeet: EditText
    private lateinit var etInches: EditText
    private lateinit var participantIdInput: EditText

    private val videoPickerLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            updateVideoPreview()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_quick_analysis)

        resetGlobalState()
        initializeViews()
        setupBackButton()
        setupButtons()
    }

    private fun setupBackButton() {
        findViewById<ImageButton>(R.id.btnBack).setOnClickListener {
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        updateVideoPreview()
    }

    private fun resetGlobalState() {
        galleryUri = null
        editedUri = null
        frameList.clear()
        participantId = 0
        participantHeight = 0
        currentPatientId = null
        currentVideoId = null

        poseFrames.clear()
        extractedFeatures = null
        extractionDiagnostics = null
        scoringResult = null
        extractedSignals = null
        extractedStrides = null
        selectedStrideIndices = null
        stepSignalMode = null
    }

    private fun initializeViews() {
        videoPreview = findViewById(R.id.ivVideoPreview)
        etFeet = findViewById(R.id.etFeet)
        etInches = findViewById(R.id.etInches)
        participantIdInput = findViewById(R.id.etParticipantId)
        
        // Set default height values
        etFeet.setText("5")
        etInches.setText("9")
    }


    private fun setupButtons() {
        findViewById<Button>(R.id.btnRecord).setOnClickListener {
            if (validateAndSaveInputs()) {
                val intent = Intent(this, VideoPickerActivity::class.java)
                intent.putExtra("mode", "record")
                videoPickerLauncher.launch(intent)
            }
        }

        findViewById<Button>(R.id.btnSelect).setOnClickListener {
            if (validateAndSaveInputs()) {
                val intent = Intent(this, VideoPickerActivity::class.java)
                intent.putExtra("mode", "gallery")
                videoPickerLauncher.launch(intent)
            }
        }

        findViewById<Button>(R.id.btnAnalyze).setOnClickListener {
            if (validateForAnalysis()) {
                startActivity(Intent(this, AnalysisActivity::class.java))
            }
        }

        findViewById<Button>(R.id.btnViewResults).setOnClickListener {
            startActivity(Intent(this, ResultsActivity::class.java))
        }
    }

    private fun validateAndSaveInputs(): Boolean {
        val id = participantIdInput.text.toString().trim()
        if (id.isEmpty()) {
            participantIdInput.error = "Participant ID is required"
            participantIdInput.requestFocus()
            return false
        }

        // Validate feet and inches using same logic as PatientCreateActivity
        for ((editText, name, max) in listOf(
            Triple(etFeet, "Feet", 8),
            Triple(etInches, "Inches", 11)
        )) {
            val str = editText.text.toString().trim()
            if (str.isEmpty()) {
                editText.error = "$name is required"
                editText.requestFocus()
                return false
            }

            val value = str.toIntOrNull()
            if (value == null || value < 0 || value > max) {
                editText.error = "$name must be between 0 and $max"
                editText.requestFocus()
                return false
            }
        }

        val feet = etFeet.text.toString().trim().toInt()
        val inches = etInches.text.toString().trim().toInt()
        val heightInInches = (feet * 12) + inches

        participantId = id.toInt()
        participantHeight = heightInInches

        return true
    }

    private fun validateForAnalysis(): Boolean {
        if (!validateAndSaveInputs()) {
            return false
        }

        if (galleryUri == null) {
            Toast.makeText(this, "Please select or record a video first", Toast.LENGTH_SHORT).show()
            return false
        }

        return true
    }

    private fun updateVideoPreview() {
        galleryUri?.let { uri ->
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this, uri)
                val frame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)
                videoPreview.setImageBitmap(frame)
            } catch (e: IllegalArgumentException) {
                Log.e("QuickAnalysisActivity", "Invalid video URI or format", e)
            } catch (e: RuntimeException) {
                Log.e("QuickAnalysisActivity", "Error accessing video file", e)
            } finally {
                retriever.release()
            }
        }
    }
}
