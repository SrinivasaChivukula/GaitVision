package GaitVision.com.ui

import android.content.Intent
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Spinner
import android.view.View
import android.widget.TextView
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
    private lateinit var feetSpinner: Spinner
    private lateinit var inchesSpinner: Spinner
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
        setupSpinners()
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
        feetSpinner = findViewById(R.id.spinnerFeet)
        inchesSpinner = findViewById(R.id.spinnerInches)
        participantIdInput = findViewById(R.id.etParticipantId)
    }

    private fun setupSpinners() {
        val feetArray = arrayOf("4", "5", "6", "7")
        val feetAdapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, feetArray) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }
        }
        feetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        feetSpinner.adapter = feetAdapter
        feetSpinner.setSelection(1)

        val inchesArray = (0..11).map { it.toString() }.toTypedArray()
        val inchesAdapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, inchesArray) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }
        }
        inchesAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        inchesSpinner.adapter = inchesAdapter
        inchesSpinner.setSelection(9)
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
            Toast.makeText(this, "Please enter a Participant ID", Toast.LENGTH_SHORT).show()
            return false
        }

        val feet = feetSpinner.selectedItem.toString().toIntOrNull() ?: 5
        val inches = inchesSpinner.selectedItem.toString().toIntOrNull() ?: 9
        val heightInInches = (feet * 12) + inches

        if (heightInInches <= 0) {
            Toast.makeText(this, "Please select a valid height", Toast.LENGTH_SHORT).show()
            return false
        }

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
            try {
                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(this, uri)
                val frame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)
                videoPreview.setImageBitmap(frame)
                retriever.release()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
}
