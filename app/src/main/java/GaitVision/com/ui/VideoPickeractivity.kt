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
import androidx.appcompat.app.AppCompatActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import GaitVision.com.R

class VideoPickerActivity : AppCompatActivity() {

    private var selectedVideo: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_picker)

        val videoView = findViewById<VideoView>(R.id.videoView)
        val tvPlaceholder = findViewById<TextView>(R.id.tvPlaceholder)
        val tvStatus = findViewById<TextView>(R.id.tvStatus)
        val btnContinue = findViewById<Button>(R.id.btnContinue)

        // photo picker api
        val pickVideoLauncher =
            registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
                Log.d("VideoPicker", "Selected URI: $uri")
                
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

                    Log.d("VideoPicker", "Video playback started")
                } ?: run {
                    Log.e("VideoPicker", "URI is null - no video selected or permission denied")
                    Toast.makeText(this, "Failed to load video. Please try again.", Toast.LENGTH_SHORT).show()
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
                val intent = Intent(this, AnalysisActivity::class.java).apply {
                    data = uri
                }
                startActivity(intent)
            } ?: run {
                Toast.makeText(this, "Please select a video first", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
