package GaitVision.com.ui

import android.content.Intent
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.VideoView
import android.widget.Toast
import android.net.Uri
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import GaitVision.com.R

class VideoPickerActivity : BaseActivity() {

    private var selectedVideo: Uri? = null

    private lateinit var videoView: VideoView
    private lateinit var tvPlaceholder: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnContinue: Button
    

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_picker)

        // Initialize view properties
        videoView = findViewById(R.id.videoView)
        tvPlaceholder = findViewById(R.id.tvPlaceholder)
        tvStatus = findViewById(R.id.tvStatus)
        btnContinue = findViewById(R.id.btnContinue)

        setupCommonHeader("Select Video")
        setupButtons()
    }

    private fun setupButtons() {

        // photo picker api
        val pickVideoLauncher =
            registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
                Log.d("GaitMedia", "Selected URI: $uri")

                uri?.let { videoUri ->
                    selectedVideo = videoUri

                    // Show the video view and hide placeholder
                    videoView.visibility = View.VISIBLE
                    tvPlaceholder.visibility = View.GONE
                    btnContinue.visibility = View.VISIBLE

                    // Update status text
                    tvStatus.text = "Video loaded successfully"

                    // Set and start the video
                    videoView.setVideoURI(videoUri)
                    videoView.start()

                    Log.d("GaitMedia", "Video playback started")
                } ?: run {
                    Log.e("GaitMedia", "URI is null - no video selected or permission denied")
                    Toast.makeText(
                        this,
                        "Failed to load video. Please try again.",
                        Toast.LENGTH_SHORT
                    ).show()
                    tvStatus.text = "Failed to load video"
                }
            }

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            // Use PickVisualMedia to select only videos
            val request = PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.VideoOnly)
                .build()
            pickVideoLauncher.launch(request)
        }

        findViewById<Button>(R.id.btnRecord).setOnClickListener {
            val recordIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            startActivity(recordIntent)
        }

        btnContinue.setOnClickListener {
            selectedVideo?.let { uri ->
                val shouldSave = intent.getBooleanExtra(AnalysisActivity.EXTRA_SHOULD_SAVE, true)
                val analysisIntent = Intent(this, AnalysisActivity::class.java).apply {
                    data = uri
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    putExtra(AnalysisActivity.EXTRA_SHOULD_SAVE, shouldSave)
                }
                startActivity(analysisIntent)
            } ?: run {
                Toast.makeText(this, "Please select a video first", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
