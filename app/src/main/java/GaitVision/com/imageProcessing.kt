package GaitVision.com

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence
import GaitVision.com.gait.FeatureExtractor
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.GaitScorer
import GaitVision.com.gait.ScoringResult
import GaitVision.com.video.detectedFps
import GaitVision.com.video.initializeMediaPipeBackend
import GaitVision.com.video.processFrameWithMediaPipe
import GaitVision.com.video.releaseMediaPipeBackend
import GaitVision.com.video.stabilizeLandmarkIdentity
import GaitVision.com.video.isMediaPipeUsingGpu
import GaitVision.com.video.decodeFramesGetFrameAtTime
import GaitVision.com.video.decodeFramesMediaCodec
import GaitVision.com.video.detectVideoFps
import GaitVision.com.video.releaseDecoderCaches
import GaitVision.com.video.createEncoderState
import GaitVision.com.video.EncoderState

private const val TAG_MEDIA = "GaitMedia"

private suspend fun hideProgressUI(activity: AppCompatActivity) {
    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = GONE
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = GONE
    }
}

private fun processFrame(frame: Bitmap, frameIndex: Int): Bitmap {
    val poseFrame = processFrameWithMediaPipe(frame, frameIndex)
    val modifiedBitmap = drawOnBitmapMediaPipe(frame, poseFrame)
    if (poseFrame != null) {
        AnalysisSession.poseFrames.add(poseFrame)
    }
    return modifiedBitmap
}

/** UI path: decode, pose, overlay, encode, extract features. MediaCodec with getFrameAtTime fallback. */
suspend fun ProcVidEmpty(context: Context, outputPath: String, activity: AppCompatActivity): Uri? {
    val TAG = TAG_MEDIA
    
    // Clear all data
    AnalysisSession.poseFrames.clear()
    AnalysisSession.extractedFeatures = null
    AnalysisSession.extractionDiagnostics = null
    AnalysisSession.scoringResult = null
    AnalysisSession.extractedSignals = null
    
    if (AnalysisSession.galleryUri == null) {
        Log.e(TAG, "No video URI provided")
        return null
    }

    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).text = "Processing..."
        activity.findViewById<TextView>(R.id.SplittingText).visibility = VISIBLE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = VISIBLE
        activity.findViewById<ProgressBar>(R.id.splittingBar).progress = 0
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = VISIBLE
        activity.findViewById<TextView>(R.id.splittingProgressValue).text = " 0%"
        activity.findViewById<TextView>(R.id.CreationText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.VideoCreation).visibility = GONE
        activity.findViewById<TextView>(R.id.CreatingProgressValue).visibility = GONE
    }

    var encoderState: EncoderState? = null
    var frameIndex = 0
    val progressBar = activity.findViewById<ProgressBar>(R.id.splittingBar)
    val progressText = activity.findViewById<TextView>(R.id.splittingProgressValue)
    var lastProgress = -1

    initializeMediaPipeBackend(context)
    val metadata = withContext(Dispatchers.IO) {
        decodeFramesMediaCodec(
            context,
            AnalysisSession.galleryUri!!,
            onMetadataReady = { meta ->
                detectedFps = meta.fps
                AnalysisSession.videoLength = meta.durationUs
                encoderState = createEncoderState(outputPath, meta.width, meta.height, meta.fps)
                encoderState!!.mediaMuxer.setOrientationHint(0)
            },
            onFrame = { frame, idx ->
                val modified = processFrame(frame, idx)
                encoderState!!.encodeFrame(modified, idx)
                frameIndex = idx + 1
            },
            onProgress = { total, current ->
                val progress = ((current.toFloat() / total) * 100).toInt().coerceIn(0, 100)
                if (progress != lastProgress) {
                    lastProgress = progress
                    activity.runOnUiThread {
                        progressBar.progress = progress
                        progressText.text = " $progress%"
                    }
                }
            }
        )
    }

    if (metadata == null) {
        Log.e(TAG, "MediaCodec decode failed, falling back to getFrameAtTime")
        return procVidEmptyFallback(context, outputPath, activity)
    }

    val width = metadata.width
    val height = metadata.height
    val totalFrames = metadata.totalFrames
    Log.d(TAG, "FAST STREAMING complete. Processed $frameIndex frames, ${AnalysisSession.poseFrames.size} poses detected")
    Log.d(TAG, "GPU delegate: ${isMediaPipeUsingGpu()}, CLAHE: $enableCLAHE")

    encoderState!!.finishEncoding()
    encoderState!!.release()
    releaseMediaPipeBackend()
    hideProgressUI(activity)

    extractGaitFeatures(context, width, height, frameIndex, activity)

    val poseCount = AnalysisSession.poseFrames.size
    AnalysisSession.poseFrames.clear()
    releaseDecoderCaches()
    Log.d(TAG, "Cleared poseFrames ($poseCount poses freed)")
    
    Log.d(TAG, "Pipeline complete")

    val outputFile = File(outputPath)
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
    Log.d(TAG, "Generated URI: $uri")
    return uri
}

/** Fallback when MediaCodec fails. */
private suspend fun procVidEmptyFallback(context: Context, outputPath: String, activity: AppCompatActivity): Uri? {
    val TAG = TAG_MEDIA
    Log.w(TAG, "Using SLOW fallback method (getFrameAtTime)")
    detectedFps = withContext(Dispatchers.IO) { detectVideoFps(context, AnalysisSession.galleryUri) }
    initializeMediaPipeBackend(context)
    var encoderState: EncoderState? = null
    var frameIndex = 0
    var totalFrames = 1
    val progressBar = activity.findViewById<ProgressBar>(R.id.splittingBar)
    val progressText = activity.findViewById<TextView>(R.id.splittingProgressValue)
    var lastProgress = -1
    val metadata = withContext(Dispatchers.IO) {
        decodeFramesGetFrameAtTime(
            context,
            AnalysisSession.galleryUri!!,
            onMetadataReady = { meta ->
                totalFrames = meta.totalFrames
                AnalysisSession.videoLength = meta.durationUs
                encoderState = createEncoderState(outputPath, meta.width, meta.height, meta.fps)
                encoderState!!.mediaMuxer.setOrientationHint(0)
            },
            onFrame = { frame, idx ->
                val modified = processFrame(frame, idx)
                encoderState!!.encodeFrame(modified, idx)
                frameIndex = idx + 1
                val progress = ((idx + 1).toFloat() / totalFrames * 100).toInt().coerceIn(0, 100)
                if (progress != lastProgress) {
                    lastProgress = progress
                    activity.runOnUiThread {
                        progressBar.progress = progress
                        progressText.text = " $progress%"
                    }
                }
            }
        )
    }
    if (metadata == null) {
        releaseMediaPipeBackend()
        return AnalysisSession.galleryUri
    }
    encoderState!!.finishEncoding()
    encoderState!!.release()
    releaseMediaPipeBackend()
    hideProgressUI(activity)
    extractGaitFeatures(context, metadata.width, metadata.height, frameIndex, activity)
    val outputFile = File(outputPath)
    return FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
}

/** Extract features from pose frames. Shared by UI and batch paths. */
private fun extractGaitFeaturesFromPoses(
    context: Context,
    poseFrames: List<PoseFrame>,
    videoId: String,
    fps: Float,
    frameWidth: Int,
    frameHeight: Int,
    totalFrames: Int
): Pair<GaitFeatures?, GaitDiagnostics> {
    if (poseFrames.isEmpty()) {
        Log.w(TAG_MEDIA, "No pose frames, skipping feature extraction")
        return Pair(null, GaitDiagnostics.empty())
    }
    return try {
        var poseSequence = PoseSequence(
            videoId = videoId,
            fps = fps,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            numFramesTotal = totalFrames,
            frames = poseFrames
        )
        val featureExtractor = FeatureExtractor()
        poseSequence = stabilizeLandmarkIdentity(poseSequence)
        poseSequence = featureExtractor.normalizeDirection(poseSequence)
        Log.d(TAG_MEDIA, "Walking direction: ${poseSequence.walkingDirection}, flipped: ${poseSequence.wasFlipped}")
        featureExtractor.extract(poseSequence)
    } catch (e: Exception) {
        Log.e(TAG_MEDIA, "Error during feature extraction", e)
        Pair(null, GaitDiagnostics.empty())
    }
}

private suspend fun extractGaitFeatures(
    context: Context,
    frameWidth: Int,
    frameHeight: Int,
    totalFrames: Int,
    activity: AppCompatActivity
) {
    Log.d(TAG_MEDIA, "Starting feature extraction with ${AnalysisSession.poseFrames.size} pose frames")
    if (AnalysisSession.poseFrames.isEmpty()) {
        Log.w(TAG_MEDIA, "No pose frames collected, skipping feature extraction")
        return
    }
    val videoId = AnalysisSession.galleryUri?.lastPathSegment ?: "unknown"
    val (features, diagnostics) = extractGaitFeaturesFromPoses(
        context, AnalysisSession.poseFrames.toList(), videoId, detectedFps, frameWidth, frameHeight, totalFrames
    )
    AnalysisSession.extractedFeatures = features
    AnalysisSession.extractionDiagnostics = diagnostics
    if (features != null) {
        Log.d(TAG_MEDIA, "Feature extraction successful! Cadence: ${features.cadence_spm} spm")
        val scorer = GaitScorer(context)
        if (scorer.initialize()) {
            AnalysisSession.scoringResult = scorer.score(features)
            scorer.release()
        }
    } else {
        Log.w(TAG_MEDIA, "Feature extraction failed: ${diagnostics.qualityFlag}")
    }
}

/** Batch path: decode + pose only. progressCallback from IO thread. */
suspend fun runVideoPipelineHeadless(
    context: Context,
    uri: Uri,
    progressCallback: ((totalFrames: Int, currentFrame: Int) -> Unit)? = null
): Pair<GaitFeatures?, GaitDiagnostics>? {
    val TAG = TAG_MEDIA
    val videoId = uri.lastPathSegment ?: "unknown"
    val poseFramesList = mutableListOf<PoseFrame>()
    var frameIndex = 0
    var width = 0
    var height = 0
    var fps = 30f
    initializeMediaPipeBackend(context)
    val metadata = withContext(Dispatchers.IO) {
        decodeFramesMediaCodec(
            context,
            uri,
            onMetadataReady = { meta ->
                detectedFps = meta.fps
                width = meta.width
                height = meta.height
                fps = meta.fps
            },
            onFrame = { frame, idx ->
                processFrameWithMediaPipe(frame, idx, fps = fps)?.let { poseFramesList.add(it) }
                frameIndex = idx + 1
            },
            onProgress = progressCallback?.let { cb ->
                val handler = android.os.Handler(android.os.Looper.getMainLooper())
                var lastReported = -1
                { totalFrames: Int, currentFrame: Int ->
                    if (currentFrame != lastReported && (currentFrame % 10 == 0 || currentFrame == totalFrames)) {
                        lastReported = currentFrame
                        handler.post { cb(totalFrames, currentFrame) }
                    }
                }
            }
        )
    }
    if (metadata == null) {
        releaseMediaPipeBackend()
        return runVideoPipelineHeadlessFallback(context, uri)
    }
    releaseMediaPipeBackend()
    releaseDecoderCaches()
    val (features, diagnostics) = extractGaitFeaturesFromPoses(
        context, poseFramesList, videoId, fps, width, height, frameIndex
    )
    Log.d(TAG, "Headless: $frameIndex frames, ${poseFramesList.size} poses")
    return Pair(features, diagnostics)
}

private suspend fun runVideoPipelineHeadlessFallback(context: Context, uri: Uri): Pair<GaitFeatures?, GaitDiagnostics>? {
    val videoId = uri.lastPathSegment ?: "unknown"
    val poseFramesList = mutableListOf<PoseFrame>()
    var frameIndex = 0
    var width = 0
    var height = 0
    var fps = 30f
    var totalFrames = 1
    initializeMediaPipeBackend(context)
    val metadata = withContext(Dispatchers.IO) {
        decodeFramesGetFrameAtTime(
            context,
            uri,
            onMetadataReady = { meta ->
                width = meta.width
                height = meta.height
                fps = meta.fps
                totalFrames = meta.totalFrames
            },
            onFrame = { frame, idx ->
                processFrameWithMediaPipe(frame, idx, fps = fps)?.let { poseFramesList.add(it) }
                frameIndex = idx + 1
            }
        )
    }
    releaseMediaPipeBackend()
    if (metadata == null) return null
    return extractGaitFeaturesFromPoses(context, poseFramesList, videoId, fps, width, height, totalFrames.coerceAtLeast(frameIndex))
}

