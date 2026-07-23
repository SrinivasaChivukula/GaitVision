package GaitVision.com.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.PickVisualMediaRequest
import androidx.lifecycle.lifecycleScope
import GaitVision.com.R
import GaitVision.com.AnalysisSession
import GaitVision.com.ProcVidEmpty
import GaitVision.com.detectVideoFps
import GaitVision.com.gait.BackFeatureExtractor
import GaitVision.com.gait.FeatureExtractor
import GaitVision.com.gait.MultiviewGaitScorer
import GaitVision.com.mediapipe.PoseSequence
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MultiviewAnalysisActivity : BaseActivity() {

    private var sideVideoUri: Uri? = null
    private var backVideoUri: Uri? = null

    private lateinit var tvSideStatus: TextView
    private lateinit var tvBackStatus: TextView
    private lateinit var tvResult: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnRunAnalysis: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_multiview_analysis)

        setupCommonHeader("Multiview Analysis")

        tvSideStatus = findViewById(R.id.tvSideStatus)
        tvBackStatus = findViewById(R.id.tvBackStatus)
        tvResult = findViewById(R.id.tvResult)
        progressBar = findViewById(R.id.progressBar)
        btnRunAnalysis = findViewById(R.id.btnRunAnalysis)

        val pickSideLauncher = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
            uri?.let {
                sideVideoUri = it
                tvSideStatus.text = "Side video selected"
                Log.d("MultiviewUI", "Side video: $it")
            }
        }

        val pickBackLauncher = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
            uri?.let {
                backVideoUri = it
                tvBackStatus.text = "Back video selected"
                Log.d("MultiviewUI", "Back video: $it")
            }
        }

        findViewById<LinearLayout>(R.id.btnPickSide).setOnClickListener {
            pickSideLauncher.launch(
                PickVisualMediaRequest.Builder()
                    .setMediaType(ActivityResultContracts.PickVisualMedia.VideoOnly)
                    .build()
            )
        }

        findViewById<LinearLayout>(R.id.btnPickBack).setOnClickListener {
            pickBackLauncher.launch(
                PickVisualMediaRequest.Builder()
                    .setMediaType(ActivityResultContracts.PickVisualMedia.VideoOnly)
                    .build()
            )
        }

        btnRunAnalysis.setOnClickListener {
            runAnalysis()
        }
    }

    private fun runAnalysis() {
        val side = sideVideoUri
        val back = backVideoUri
        if (side == null || back == null) {
            Toast.makeText(this, "Please select both videos first", Toast.LENGTH_SHORT).show()
            return
        }

        progressBar.visibility = View.VISIBLE
        tvResult.text = ""
        btnRunAnalysis.isEnabled = false

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // --- Process side video ---
                AnalysisSession.galleryUri = side
                val sideOutputPath = "${cacheDir.absolutePath}/side_temp.mp4"
                ProcVidEmpty(this@MultiviewAnalysisActivity, sideOutputPath)
                val sideFps = detectVideoFps(applicationContext, side)
                val sideFramesCopy = AnalysisSession.lastPoseFramesSnapshot
                val sideNumFramesTotal = (sideFramesCopy.maxOfOrNull { it.frameIdx } ?: 0) + 1
                val sidePoseSeq = PoseSequence(
                    videoId = "side",
                    fps = sideFps,
                    frameWidth = 0,
                    frameHeight = 0,
                    numFramesTotal = sideNumFramesTotal,
                    frames = sideFramesCopy
                )
                val sideFeatures = FeatureExtractor().extract(sidePoseSeq).first

                // --- Process back video ---
                AnalysisSession.galleryUri = back
                val backOutputPath = "${cacheDir.absolutePath}/back_temp.mp4"
                ProcVidEmpty(this@MultiviewAnalysisActivity, backOutputPath)
                val backFps = detectVideoFps(applicationContext, back)
                val backFramesCopy = AnalysisSession.lastPoseFramesSnapshot
                val backNumFramesTotal = (backFramesCopy.maxOfOrNull { it.frameIdx } ?: 0) + 1
                val backPoseSeq = PoseSequence(
                    videoId = "back",
                    fps = backFps,
                    frameWidth = 0,
                    frameHeight = 0,
                    numFramesTotal = backNumFramesTotal,
                    frames = backFramesCopy
                )
                val backFeatures = BackFeatureExtractor().extract(backPoseSeq)

                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    btnRunAnalysis.isEnabled = true

                    if (sideFeatures == null || backFeatures == null) {
                        tvResult.text = "Could not extract features from one or both videos (low quality / insufficient steps)"
                        return@withContext
                    }

                    val scorer = MultiviewGaitScorer(applicationContext)
                    val score = scorer.score(sideFeatures, backFeatures)

                    if (score.isNaN()) {
                        tvResult.text = "Scoring failed"
                    } else {
                        val intent = Intent(
                            this@MultiviewAnalysisActivity,
                            MultiviewResultsActivity::class.java
                        ).apply {
                            putExtra(
                                MultiviewResultsActivity.EXTRA_MULTIVIEW_SCORE,
                                score
                            )
                        }

                        startActivity(intent)
                    }
                    scorer.release()
                }
            } catch (e: Exception) {
                Log.e("MultiviewUI", "Analysis failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    btnRunAnalysis.isEnabled = true
                    tvResult.text = "Error: ${e.message}"
                }
            }
        }
    }
}