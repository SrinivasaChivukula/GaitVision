package GaitVision.com

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.util.Log
import android.view.View
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

// MediaPipe Tasks imports
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence
import GaitVision.com.mediapipe.MediaPipeResultConverter

// Gait analysis imports
import GaitVision.com.gait.FeatureExtractor
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.GaitScorer
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.QualityFlag

/**
 * Convert YUV_420_888 Image (from MediaCodec) to ARGB Bitmap.
 * Uses JPEG as intermediate - this leverages Android's native (hardware-accelerated) 
 * YUV handling which is much faster than pure Kotlin pixel manipulation.
 */
private fun imageToBitmap(image: Image): Bitmap {
    val width = image.width
    val height = image.height
    
    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]
    
    val yBuffer = yPlane.buffer
    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer
    
    val yRowStride = yPlane.rowStride
    val uvRowStride = uPlane.rowStride
    val uvPixelStride = uPlane.pixelStride
    
    // Create NV21 byte array (Y plane + interleaved VU)
    val nv21 = ByteArray(width * height * 3 / 2)
    
    // Copy Y plane, handling row stride
    var pos = 0
    for (row in 0 until height) {
        yBuffer.position(row * yRowStride)
        yBuffer.get(nv21, pos, width)
        pos += width
    }
    
    // Copy UV planes interleaved as VU (for NV21)
    val uvHeight = height / 2
    val uvWidth = width / 2
    for (row in 0 until uvHeight) {
        for (col in 0 until uvWidth) {
            val uvIndex = row * uvRowStride + col * uvPixelStride
            
            vBuffer.position(uvIndex)
            uBuffer.position(uvIndex)
            
            nv21[pos++] = vBuffer.get()
            nv21[pos++] = uBuffer.get()
        }
    }
    
    // Use Android's native JPEG encoding/decoding (hardware accelerated)
    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
    val imageBytes = out.toByteArray()
    
    // Return mutable bitmap for wireframe drawing
    val options = BitmapFactory.Options().apply { inMutable = true }
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
}

/**
 * Plot a line graph with left and right data series.
 */
fun plotLineGraph(
    lineChart: LineChart,
    leftData: List<Float>,
    rightData: List<Float>,
    labelLeft: String,
    labelRight: String
) {
    val leftEntries = leftData.mapIndexed { index, angle ->
        val convertToSecond = index / 30f
        Entry(convertToSecond, angle)
    }
    val rightEntries = rightData.mapIndexed { index, angle ->
        val convertToSecond = index / 30f
        Entry(convertToSecond, angle)
    }

    val leftDataSet = LineDataSet(leftEntries, labelLeft)
    leftDataSet.color = Color.BLUE
    leftDataSet.valueTextSize = 12f
    leftDataSet.setDrawCircles(false)
    leftDataSet.setDrawValues(false)

    val rightDataSet = LineDataSet(rightEntries, labelRight)
    rightDataSet.color = Color.RED
    rightDataSet.valueTextSize = 12f
    rightDataSet.setDrawCircles(false)
    rightDataSet.setDrawValues(false)

    val lineData = LineData(leftDataSet, rightDataSet)

    lineChart.data = lineData
    lineChart.description.isEnabled = false
    lineChart.invalidate()
}

/**
 * Global MediaPipe backend instance (initialized once per video processing session).
 */
private var mediaPipeBackend: MediaPipePoseBackend? = null

/**
 * Detected FPS from video metadata (set during frame extraction).
 */
var detectedFps: Float = 30f
    private set

/**
 * Initialize MediaPipe backend for a processing session.
 * Uses OPTIMAL_CONFIG parameters from PC pipeline for feature parity.
 */
fun initializeMediaPipeBackend(context: Context) {
    if (mediaPipeBackend == null) {
        mediaPipeBackend = MediaPipePoseBackend(
            context = context,
            minDetectionConfidence = 0.40f,  // OPTIMAL_CONFIG
            minTrackingConfidence = 0.61f,   // OPTIMAL_CONFIG
            minPresenceConfidence = 0.5f
        )
        Log.d("ImageProcessing", "MediaPipe backend initialized with OPTIMAL_CONFIG")
    }
}

/**
 * Detect actual FPS from video metadata.
 * Falls back to 30 FPS if detection fails (mirrors PC behavior).
 */
fun detectVideoFps(context: Context, fileUri: Uri?): Float {
    if (fileUri == null) return 30f
    
    val retriever = MediaMetadataRetriever()
    return try {
        try {
            val pfd = context.contentResolver.openFileDescriptor(fileUri, "r")
            if (pfd != null) {
                retriever.setDataSource(pfd.fileDescriptor)
                pfd.close()
            } else {
                retriever.setDataSource(context, fileUri)
            }
        } catch (e: Exception) {
            retriever.setDataSource(context, fileUri)
        }
        
        // Try to get frame rate from metadata
        val frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
        val fps = frameRateStr?.toFloatOrNull()
        
        if (fps != null && fps > 0) {
            Log.d("ImageProcessing", "Detected FPS from metadata: $fps")
            fps
        } else {
            // Fallback: estimate from duration and frame count
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val videoFrameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            
            if (videoFrameCount != null && videoFrameCount > 0 && durationMs > 0) {
                val estimatedFps = (videoFrameCount * 1000f) / durationMs
                Log.d("ImageProcessing", "Estimated FPS from frame count: $estimatedFps")
                estimatedFps.coerceIn(15f, 120f)  // Sanity check
            } else {
                Log.d("ImageProcessing", "Could not detect FPS, using default 30")
                30f
            }
        }
    } catch (e: Exception) {
        Log.w("ImageProcessing", "Error detecting FPS: ${e.message}, using default 30")
        30f
    } finally {
        retriever.release()
    }
}

/**
 * Release MediaPipe backend resources.
 */
fun releaseMediaPipeBackend() {
    mediaPipeBackend?.release()
    mediaPipeBackend = null
    Log.d("ImageProcessing", "MediaPipe backend released")
}

// Timing accumulators for CLAHE vs pure MediaPipe (for diagnostics)
var totalClaheTimeMs = 0L
var totalPureMediaPipeTimeMs = 0L
var totalDownscaleTimeMs = 0L
var mediaPipeFrameCount = 0

// Processing resolution for CLAHE + MediaPipe (720p for speed, coords are normalized so wireframe works at any res)
const val PROCESSING_WIDTH = 1280
const val PROCESSING_HEIGHT = 720

/**
 * Process a single frame using MediaPipe Tasks PoseLandmarker.
 * 
 * Downscales to 720p for CLAHE + MediaPipe processing (2.25x faster).
 * Returns normalized coordinates (0-1) that work at any resolution.
 * 
 * @param bitmap Frame to process (any resolution)
 * @param frameIdx Frame index for timestamp calculation
 * @param fps Actual video FPS (from detectVideoFps)
 * @param applyClahe Whether to apply CLAHE contrast enhancement (mirrors PC enhance_contrast option)
 * @return PoseFrame with normalized coordinates, or null if detection failed
 */
fun processFrameWithMediaPipe(
    bitmap: Bitmap, 
    frameIdx: Int, 
    fps: Float = detectedFps,
    applyClahe: Boolean = enableCLAHE
): PoseFrame? {
    val backend = mediaPipeBackend ?: return null
    
    // Downscale to 720p for faster CLAHE + MediaPipe processing
    // Normalized coords (0-1) returned by MediaPipe work at any resolution
    var t0 = System.currentTimeMillis()
    val scaledBitmap = if (bitmap.width > PROCESSING_WIDTH || bitmap.height > PROCESSING_HEIGHT) {
        Bitmap.createScaledBitmap(bitmap, PROCESSING_WIDTH, PROCESSING_HEIGHT, true)
    } else {
        bitmap
    }
    totalDownscaleTimeMs += System.currentTimeMillis() - t0
    
    // Optionally apply CLAHE contrast enhancement (mirrors PC _apply_clahe)
    t0 = System.currentTimeMillis()
    val processedBitmap = if (applyClahe) {
        backend.applyCLAHE(scaledBitmap)
    } else {
        scaledBitmap
    }
    if (applyClahe) {
        totalClaheTimeMs += System.currentTimeMillis() - t0
    }
    
    // Calculate timestamp in milliseconds using actual FPS
    val timestampMs = (frameIdx * 1000L / fps).toLong()
    
    t0 = System.currentTimeMillis()
    val result = backend.processFrame(processedBitmap, timestampMs)
    totalPureMediaPipeTimeMs += System.currentTimeMillis() - t0
    mediaPipeFrameCount++
    
    return MediaPipeResultConverter.toPoseFrame(
        result = result,
        frameIdx = frameIdx,
        timestampS = timestampMs / 1000f
    )
}

/**
 * Draw skeleton overlay and calculate angles from MediaPipe pose frame.
 * 
 * MediaPipe returns normalized coordinates (0-1), which we convert to pixel
 * coordinates for drawing. Angle calculations use the same formulas as before.
 * 
 * @param bitmap Frame to draw on (modified in place)
 * @param poseFrame Pose detection result with normalized coordinates
 * @return The modified bitmap with skeleton overlay
 */
fun drawOnBitmapMediaPipe(bitmap: Bitmap, poseFrame: PoseFrame?): Bitmap {
    if (poseFrame == null) {
        return bitmap
    }
    
    val width = bitmap.width.toFloat()
    val height = bitmap.height.toFloat()
    val keypoints = poseFrame.keypoints
    val confidences = poseFrame.confidences
    
    // Helper to get pixel coordinates from normalized keypoint
    fun getPixelCoords(landmarkIdx: Int): Pair<Float, Float> {
        val x = keypoints[landmarkIdx][0] * width
        val y = keypoints[landmarkIdx][1] * height
        return Pair(x, y)
    }
    
    // Helper to check if landmark is visible (confidence threshold)
    fun isVisible(landmarkIdx: Int): Boolean {
        return confidences[landmarkIdx] > 0.3f
    }
    
    // Get coordinates for all landmarks we need
    val (leftShoulderX, leftShoulderY) = getPixelCoords(MediaPipePoseBackend.LEFT_SHOULDER)
    val (rightShoulderX, rightShoulderY) = getPixelCoords(MediaPipePoseBackend.RIGHT_SHOULDER)
    val (leftHipX, leftHipY) = getPixelCoords(MediaPipePoseBackend.LEFT_HIP)
    val (rightHipX, rightHipY) = getPixelCoords(MediaPipePoseBackend.RIGHT_HIP)
    val (leftKneeX, leftKneeY) = getPixelCoords(MediaPipePoseBackend.LEFT_KNEE)
    val (rightKneeX, rightKneeY) = getPixelCoords(MediaPipePoseBackend.RIGHT_KNEE)
    val (leftAnkleX, leftAnkleY) = getPixelCoords(MediaPipePoseBackend.LEFT_ANKLE)
    val (rightAnkleX, rightAnkleY) = getPixelCoords(MediaPipePoseBackend.RIGHT_ANKLE)
    val (leftHeelX, leftHeelY) = getPixelCoords(MediaPipePoseBackend.LEFT_HEEL)
    val (rightHeelX, rightHeelY) = getPixelCoords(MediaPipePoseBackend.RIGHT_HEEL)
    val (leftFootIndexX, leftFootIndexY) = getPixelCoords(MediaPipePoseBackend.LEFT_FOOT_INDEX)
    val (rightFootIndexX, rightFootIndexY) = getPixelCoords(MediaPipePoseBackend.RIGHT_FOOT_INDEX)
    
    // Angle Calculations (same formulas as before)
    // Ankle Angles
    val leftAnkleAngle = GetAnglesA(leftFootIndexX, leftFootIndexY, leftAnkleX, leftAnkleY, leftKneeX, leftKneeY)
    if (!leftAnkleAngle.isNaN() && leftAnkleAngle < 70 && leftAnkleAngle > -25) {
        leftAnkleAngles.add(leftAnkleAngle)
    } else {
        count++
        Log.d("ErrorCheck", "Left Ankle: $leftAnkleAngle")
    }
    
    val rightAnkleAngle = GetAnglesA(rightFootIndexX, rightFootIndexY, rightAnkleX, rightAnkleY, rightKneeX, rightKneeY)
    if (!rightAnkleAngle.isNaN() && rightAnkleAngle < 70 && rightAnkleAngle > -25) {
        rightAnkleAngles.add(rightAnkleAngle)
    } else {
        count++
        Log.d("ErrorCheck", "Right Ankle: $rightAnkleAngle")
    }

    // Knee Angles
    val leftKneeAngle = GetAngles(leftAnkleX, leftAnkleY, leftKneeX, leftKneeY, leftHipX, leftHipY)
    if (!leftKneeAngle.isNaN()) {
        leftKneeAngles.add(leftKneeAngle)
    } else {
        count++
    }
    
    val rightKneeAngle = GetAngles(rightAnkleX, rightAnkleY, rightKneeX, rightKneeY, rightHipX, rightHipY)
    if (!rightKneeAngle.isNaN()) {
        rightKneeAngles.add(rightKneeAngle)
    } else {
        count++
    }

    // Hip Angles
    val leftHipAngle = GetAngles(leftKneeX, leftKneeY, leftHipX, leftHipY, leftShoulderX, leftShoulderY)
    if (!leftHipAngle.isNaN()) {
        leftHipAngles.add(leftHipAngle)
    } else {
        count++
    }
    
    val rightHipAngle = GetAngles(rightKneeX, rightKneeY, rightHipX, rightHipY, rightShoulderX, rightShoulderY)
    if (!rightHipAngle.isNaN()) {
        rightHipAngles.add(rightHipAngle)
    } else {
        count++
    }

    // Torso Angle
    val torsoAngle = calcTorso(
        (leftHipX + rightHipX) / 2, (leftHipY + rightHipY) / 2,
        (rightShoulderX + leftShoulderX) / 2, (rightShoulderY + leftShoulderY) / 2
    )
    if (!torsoAngle.isNaN() && torsoAngle > -20 && torsoAngle < 20) {
        torsoAngles.add(torsoAngle)
    } else {
        count++
        Log.d("ErrorCheck", "TorsoAngle: $torsoAngle")
    }

    // Stride angle
    val strideAngle = calcStrideAngle(
        leftHeelX, leftHeelY,
        (leftHipX + rightHipX) / 2f, (leftHipY + rightHipY) / 2,
        rightHeelX, rightHeelY
    )
    strideAngles.add(strideAngle)

    // Draw skeleton overlay
    val canvas = Canvas(bitmap)

    val paintCircleRight = Paint().apply { setARGB(255, 255, 0, 0) }
    val paintCircleLeft = Paint().apply { setARGB(255, 0, 0, 255) }
    val paintLine = Paint().apply {
        setARGB(255, 255, 255, 255)
        strokeWidth = 4f
    }

    // Draw connections
    canvas.drawLine(rightHipX, rightHipY, rightKneeX, rightKneeY, paintLine)
    canvas.drawLine(leftHipX, leftHipY, leftKneeX, leftKneeY, paintLine)
    canvas.drawLine(rightKneeX, rightKneeY, rightAnkleX, rightAnkleY, paintLine)
    canvas.drawLine(leftKneeX, leftKneeY, leftAnkleX, leftAnkleY, paintLine)
    canvas.drawLine(rightAnkleX, rightAnkleY, rightFootIndexX, rightFootIndexY, paintLine)
    canvas.drawLine(leftAnkleX, leftAnkleY, leftFootIndexX, leftFootIndexY, paintLine)
    canvas.drawLine(rightAnkleX, rightAnkleY, rightHeelX, rightHeelY, paintLine)
    canvas.drawLine(leftAnkleX, leftAnkleY, leftHeelX, leftHeelY, paintLine)
    canvas.drawLine(rightHeelX, rightHeelY, rightFootIndexX, rightFootIndexY, paintLine)
    canvas.drawLine(leftHeelX, leftHeelY, leftFootIndexX, leftFootIndexY, paintLine)

    // Draw points
    canvas.drawCircle(rightHipX, rightHipY, 4f, paintCircleRight)
    canvas.drawCircle(leftHipX, leftHipY, 4f, paintCircleLeft)
    canvas.drawCircle(rightKneeX, rightKneeY, 4f, paintCircleRight)
    canvas.drawCircle(leftKneeX, leftKneeY, 4f, paintCircleLeft)
    canvas.drawCircle(rightAnkleX, rightAnkleY, 4f, paintCircleRight)
    canvas.drawCircle(leftAnkleX, leftAnkleY, 4f, paintCircleLeft)
    canvas.drawCircle(rightHeelX, rightHeelY, 4f, paintCircleRight)
    canvas.drawCircle(leftHeelX, leftHeelY, 4f, paintCircleLeft)
    canvas.drawCircle(rightFootIndexX, rightFootIndexY, 4f, paintCircleRight)
    canvas.drawCircle(leftFootIndexX, leftFootIndexY, 4f, paintCircleLeft)

    return bitmap
}

/**
 * Extract frames from video as bitmaps.
 */
suspend fun getFrameBitmaps(context: Context, fileUri: Uri?, activity: AppCompatActivity) {
    if (fileUri == null) return

    val retriever = MediaMetadataRetriever()
    frameList = mutableListOf()
    
    try {
        val pfd = context.contentResolver.openFileDescriptor(fileUri, "r")
        if (pfd != null) {
            retriever.setDataSource(pfd.fileDescriptor)
            pfd.close()
        } else {
            retriever.setDataSource(context, fileUri)
        }
    } catch (e: Exception) {
        Log.e("ImageProcessing", "Error opening video: ${e.message}")
        retriever.setDataSource(context, fileUri)
    }

    val mimeType = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE)
    
    if (mimeType == "video/mp4") {
        val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
        val frameInterval = (1000L * 1000L / detectedFps).toLong()  // Use detected FPS
        var currTime = 0L
        val videoLengthUs = videoLengthMs * 1000L
        videoLength = videoLengthUs

        withContext(Dispatchers.Main) {
            activity.findViewById<View>(R.id.splittingBar).visibility = VISIBLE
            activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = VISIBLE
            activity.findViewById<TextView>(R.id.splittingProgressValue).text = " 0%"
        }
        
        while (currTime <= videoLengthUs) {
            val frame = retriever.getFrameAtTime(currTime, MediaMetadataRetriever.OPTION_CLOSEST)
            if (frame != null) {
                frameList.add(frame)
            }

            val progress = ((currTime.toDouble() / videoLengthUs) * 100).toInt()
            withContext(Dispatchers.Main) {
                activity.findViewById<ProgressBar>(R.id.splittingBar).setProgress(progress)
                activity.findViewById<TextView>(R.id.splittingProgressValue).text = " $progress%"
            }
            currTime += frameInterval
        }
    } else if (mimeType == "image/jpeg" || mimeType == "image/png") {
        val stream = context.contentResolver.openInputStream(fileUri)
        val frame = BitmapFactory.decodeStream(stream)
        frameList.add(frame)
    }

    retriever.release()
}

/**
 * Main video processing function using MediaPipe Tasks + FAST MediaCodec extraction.
 * 
 * Uses MediaCodec decoder for 5-10x faster frame extraction vs getFrameAtTime().
 * 
 * Pipeline (mirrors PC cli.py "retry if bad" pattern):
 * 1. Set up MediaExtractor to read video data
 * 2. Set up MediaCodec decoder for fast frame extraction
 * 3. Initialize MediaPipe backend
 * 4. Process each frame with pose detection (streaming - no memory buildup)
 * 5. Draw skeleton overlay and calculate angles
 * 6. Encode processed frames back to video
 * 7. Extract 16 gait features (PC pipeline parity)
 * 8. If extraction fails (quality != OK), could retry with ROI
 * 9. Compute gait scores
 */
suspend fun ProcVidEmpty(context: Context, outputPath: String, activity: AppCompatActivity): Uri? {
    val TAG = "ImageProcessing"
    
    // Clear all data
    leftAnkleAngles.clear()
    rightAnkleAngles.clear()
    leftKneeAngles.clear()
    rightKneeAngles.clear()
    leftHipAngles.clear()
    rightHipAngles.clear()
    torsoAngles.clear()
    strideAngles.clear()
    poseFrames.clear()
    frameList.clear()
    extractedFeatures = null
    extractionDiagnostics = null
    scoringResult = null
    
    if (galleryUri == null) {
        Log.e(TAG, "No video URI provided")
        return null
    }

    // Setup UI - single progress bar for streaming
    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).text = "Processing..."
        activity.findViewById<TextView>(R.id.SplittingText).visibility = VISIBLE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = VISIBLE
        activity.findViewById<ProgressBar>(R.id.splittingBar).progress = 0
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = VISIBLE
        activity.findViewById<TextView>(R.id.splittingProgressValue).text = " 0%"
        // Hide the second progress bar - we use only one now
        activity.findViewById<TextView>(R.id.CreationText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.VideoCreation).visibility = GONE
        activity.findViewById<TextView>(R.id.CreatingProgressValue).visibility = GONE
    }

    // === Set up MediaExtractor for FAST video reading ===
    val extractor = MediaExtractor()
    val retriever = MediaMetadataRetriever()  // For FPS detection fallback
    
    try {
        val pfd = context.contentResolver.openFileDescriptor(galleryUri!!, "r")
        if (pfd != null) {
            extractor.setDataSource(pfd.fileDescriptor)
            retriever.setDataSource(pfd.fileDescriptor)
            pfd.close()
        } else {
            throw Exception("Could not open file descriptor")
        }
    } catch (e: Exception) {
        Log.e(TAG, "Error opening video: ${e.message}")
        try {
            extractor.setDataSource(context, galleryUri!!, null)
            retriever.setDataSource(context, galleryUri)
        } catch (e2: Exception) {
            Log.e(TAG, "Fallback also failed: ${e2.message}")
            return null
        }
    }
    
    // Find video track
    var videoTrackIndex = -1
    var videoFormat: MediaFormat? = null
    for (i in 0 until extractor.trackCount) {
        val format = extractor.getTrackFormat(i)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
        if (mime.startsWith("video/")) {
            videoTrackIndex = i
            videoFormat = format
            break
        }
    }
    
    if (videoTrackIndex < 0 || videoFormat == null) {
        Log.e(TAG, "No video track found")
        extractor.release()
        retriever.release()
        return galleryUri
    }
    
    extractor.selectTrack(videoTrackIndex)
    
    // Get video properties
    val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
    val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
    val videoMime = videoFormat.getString(MediaFormat.KEY_MIME) ?: "video/avc"
    val durationUs = videoFormat.getLong(MediaFormat.KEY_DURATION)
    val videoLengthMs = durationUs / 1000
    videoLength = durationUs
    
    // Detect FPS from format or metadata
    var fps = 30f
    if (videoFormat.containsKey(MediaFormat.KEY_FRAME_RATE)) {
        fps = videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE).toFloat()
    } else {
        // Fallback to metadata
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull()?.let { fps = it }
        }
        if (fps == 30f && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            val frameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            if (frameCount != null && videoLengthMs > 0) {
                val calculatedFps = (frameCount * 1000f) / videoLengthMs
                if (calculatedFps in 15f..120f) fps = calculatedFps
            }
        }
    }
    detectedFps = fps
    retriever.release()  // Done with retriever
    
    val totalFrames = ((durationUs * fps) / 1_000_000).toInt()
    Log.d(TAG, "Video: ${videoLengthMs}ms @ ${fps}fps, ${width}x${height}, ~$totalFrames frames")
    Log.d(TAG, "Using FAST MediaCodec extraction (5-10x faster than getFrameAtTime)")
    
    // === Set up MediaCodec decoder ===
    val decoder: MediaCodec
    try {
        decoder = MediaCodec.createDecoderByType(videoMime)
        // Don't modify the format - let decoder choose optimal color format
        decoder.configure(videoFormat, null, null, 0)
        decoder.start()
        Log.d(TAG, "MediaCodec decoder started for $videoMime, ${width}x${height}")
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize decoder: ${e.message}")
        Log.e(TAG, "Falling back to slow getFrameAtTime method")
        extractor.release()
        // Fallback to slow method
        return procVidEmptyFallback(context, outputPath, activity)
    }
    
    // Initialize MediaPipe
    initializeMediaPipeBackend(context)
    
    // === Set up video encoder ===
    val mediaMuxer: MediaMuxer
    val encoder: MediaCodec
    val inputSurface: android.view.Surface
    
    try {
        mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val format = MediaFormat.createVideoFormat("video/avc", width, height)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps.toInt())
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        format.setInteger(MediaFormat.KEY_ROTATION, 0)
        mediaMuxer.setOrientationHint(0)

        encoder = MediaCodec.createEncoderByType("video/avc")
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        inputSurface = encoder.createInputSurface()
        encoder.start()
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize encoder: ${e.message}", e)
        releaseMediaPipeBackend()
        decoder.stop()
        decoder.release()
        extractor.release()
        return null
    }

    val frameDurationUs = 1000000L / fps.toLong()
    var trackIndex = -1
    var muxerStarted = false
    val encoderBufferInfo = MediaCodec.BufferInfo()
    val decoderBufferInfo = MediaCodec.BufferInfo()
    var frameIndex = 0
    var inputDone = false
    var outputDone = false
    val startTime = System.currentTimeMillis()

    // Timing accumulators for performance analysis
    var totalYuvTime = 0L
    var totalClaheTime = 0L
    var totalMediaPipeTime = 0L
    var totalDrawTime = 0L
    var totalEncodeTime = 0L
    
    Log.d(TAG, "FAST STREAMING: Processing ~$totalFrames frames with MediaCodec")
    Log.d(TAG, "GPU delegate: ${mediaPipeBackend?.isUsingGpu() ?: false}, CLAHE: $enableCLAHE")

    // === FAST MediaCodec STREAMING LOOP ===
    while (!outputDone) {
        // Feed input to decoder
        if (!inputDone) {
            val inputBufferId = decoder.dequeueInputBuffer(10000)
            if (inputBufferId >= 0) {
                val inputBuffer = decoder.getInputBuffer(inputBufferId)
                if (inputBuffer != null) {
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        // End of stream
                        decoder.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        val presentationTimeUs = extractor.sampleTime
                        decoder.queueInputBuffer(inputBufferId, 0, sampleSize, presentationTimeUs, 0)
                        extractor.advance()
                    }
                }
            }
        }
        
        // Get output from decoder
        val outputBufferId = decoder.dequeueOutputBuffer(decoderBufferInfo, 10000)
        when {
            outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                // No output available yet, continue
            }
            outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                Log.d(TAG, "Decoder output format changed")
            }
            outputBufferId >= 0 -> {
                // Check for end of stream
                if ((decoderBufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    outputDone = true
                    decoder.releaseOutputBuffer(outputBufferId, false)
                } else {
                    // Get the decoded frame as Image
                    try {
                        val image = decoder.getOutputImage(outputBufferId)
                        if (image == null) {
                            Log.w(TAG, "getOutputImage returned null for frame $frameIndex")
                            decoder.releaseOutputBuffer(outputBufferId, false)
                            continue
                        }
                        
                        // TIMING: YUV -> RGB conversion
                        var t0 = System.currentTimeMillis()
                        val frame: Bitmap
                        try {
                            frame = imageToBitmap(image)
                        } finally {
                            image.close()
                        }
                        totalYuvTime += System.currentTimeMillis() - t0
                        
                        // TIMING: MediaPipe (includes CLAHE if enabled)
                        t0 = System.currentTimeMillis()
                        val poseFrame = processFrameWithMediaPipe(frame, frameIndex)
                        totalMediaPipeTime += System.currentTimeMillis() - t0
                        
                        // TIMING: Draw wireframe
                        t0 = System.currentTimeMillis()
                        val modifiedBitmap = drawOnBitmapMediaPipe(frame, poseFrame)
                        totalDrawTime += System.currentTimeMillis() - t0
                        
                        // Store pose data (small - just keypoints)
                        if (poseFrame != null) {
                            poseFrames.add(poseFrame)
                        }

                        // TIMING: Encode frame to video
                        t0 = System.currentTimeMillis()
                        val canvas = inputSurface.lockCanvas(null)
                        canvas.drawBitmap(modifiedBitmap, 0f, 0f, null)
                        inputSurface.unlockCanvasAndPost(canvas)

                        // Drain encoder
                        while (true) {
                            val encOutputId = encoder.dequeueOutputBuffer(encoderBufferInfo, 1000)
                            when {
                                encOutputId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                                    if (!muxerStarted) {
                                        trackIndex = mediaMuxer.addTrack(encoder.outputFormat)
                                        mediaMuxer.start()
                                        muxerStarted = true
                                    }
                                }
                                encOutputId >= 0 -> {
                                    val outputBuffer = encoder.getOutputBuffer(encOutputId) ?: break
                                    if (muxerStarted) {
                                        encoderBufferInfo.presentationTimeUs = frameIndex * frameDurationUs
                                        mediaMuxer.writeSampleData(trackIndex, outputBuffer, encoderBufferInfo)
                                    }
                                    encoder.releaseOutputBuffer(encOutputId, false)
                                }
                                else -> break
                            }
                        }
                        totalEncodeTime += System.currentTimeMillis() - t0
                        
                        // Update progress
                        frameIndex++
                        val progress = ((frameIndex.toFloat() / totalFrames) * 100).toInt().coerceIn(0, 100)
                        withContext(Dispatchers.Main) {
                            activity.findViewById<ProgressBar>(R.id.splittingBar).progress = progress
                            activity.findViewById<TextView>(R.id.splittingProgressValue).text = " $progress%"
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Error processing frame $frameIndex: ${e.message}")
                    }
                    
                    decoder.releaseOutputBuffer(outputBufferId, false)
                }
            }
        }
    }
    
    val elapsedSec = (System.currentTimeMillis() - startTime) / 1000.0
    Log.d(TAG, "Processed $frameIndex frames in ${elapsedSec}s (${String.format("%.1f", frameIndex/elapsedSec)} fps)")
    
    // === TIMING BREAKDOWN ===
    if (frameIndex > 0) {
        Log.d(TAG, "=== TIMING BREAKDOWN (avg per frame) ===")
        Log.d(TAG, "  YUV→RGB:      ${totalYuvTime / frameIndex}ms")
        if (mediaPipeFrameCount > 0) {
            Log.d(TAG, "  Downscale:    ${totalDownscaleTimeMs / mediaPipeFrameCount}ms (to 720p)")
            if (enableCLAHE) {
                Log.d(TAG, "  CLAHE:        ${totalClaheTimeMs / mediaPipeFrameCount}ms (at 720p)")
            }
            Log.d(TAG, "  MediaPipe:    ${totalPureMediaPipeTimeMs / mediaPipeFrameCount}ms (pure inference)")
        } else {
            Log.d(TAG, "  MediaPipe:    ${totalMediaPipeTime / frameIndex}ms")
        }
        Log.d(TAG, "  Draw:         ${totalDrawTime / frameIndex}ms")
        Log.d(TAG, "  Encode:       ${totalEncodeTime / frameIndex}ms")
        Log.d(TAG, "  Total:        ${(totalYuvTime + totalMediaPipeTime + totalDrawTime + totalEncodeTime) / frameIndex}ms")
        Log.d(TAG, "=== END TIMING ===")
    }
    
    // Reset timing counters
    totalClaheTimeMs = 0L
    totalPureMediaPipeTimeMs = 0L
    totalDownscaleTimeMs = 0L
    mediaPipeFrameCount = 0

    // Flush remaining encoder output
    encoder.signalEndOfInputStream()
    while (true) {
        val outputBufferId = encoder.dequeueOutputBuffer(encoderBufferInfo, 10000)
        if (outputBufferId >= 0) {
            val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: break
            if (muxerStarted) {
                encoderBufferInfo.presentationTimeUs = frameIndex.toLong() * frameDurationUs
                mediaMuxer.writeSampleData(trackIndex, outputBuffer, encoderBufferInfo)
            }
            encoder.releaseOutputBuffer(outputBufferId, false)
        } else {
            break
        }
    }

    // Release all resources
    decoder.stop()
    decoder.release()
    extractor.release()
    encoder.stop()
    encoder.release()
    mediaMuxer.stop()
    mediaMuxer.release()
    releaseMediaPipeBackend()

    // Hide progress UI
    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = GONE
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = GONE
    }

    // Smooth angle data for charts
    smoothDataUsingMovingAverage(rightKneeAngles, 5)
    smoothDataUsingMovingAverage(leftKneeAngles, 5)
    smoothDataUsingMovingAverage(rightHipAngles, 5)
    smoothDataUsingMovingAverage(leftHipAngles, 5)
    smoothDataUsingMovingAverage(rightAnkleAngles, 5)
    smoothDataUsingMovingAverage(leftAnkleAngles, 5)
    smoothDataUsingMovingAverage(torsoAngles, 5)
    smoothDataUsingMovingAverage(strideAngles, 5)

    Log.d(TAG, "FAST STREAMING complete. Processed $frameIndex frames, ${poseFrames.size} poses detected")
    
    // Feature extraction (uses poseFrames which is small)
    extractGaitFeatures(context, width, height, frameIndex, activity)
    
    Log.d(TAG, "Pipeline complete")

    val outputFile = File(outputPath)
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
    Log.d(TAG, "Generated URI: $uri")
    return uri
}

/**
 * Fallback video processing using slow getFrameAtTime() method.
 * Used if MediaCodec initialization fails.
 */
private suspend fun procVidEmptyFallback(context: Context, outputPath: String, activity: AppCompatActivity): Uri? {
    val TAG = "ImageProcessing"
    Log.w(TAG, "Using SLOW fallback method (getFrameAtTime)")
    
    // Detect video FPS
    detectedFps = withContext(Dispatchers.IO) {
        detectVideoFps(context, galleryUri)
    }

    val retriever = MediaMetadataRetriever()
    try {
        val pfd = context.contentResolver.openFileDescriptor(galleryUri!!, "r")
        if (pfd != null) {
            retriever.setDataSource(pfd.fileDescriptor)
            pfd.close()
        } else {
            retriever.setDataSource(context, galleryUri)
        }
    } catch (e: Exception) {
        retriever.setDataSource(context, galleryUri)
    }

    val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
    val videoLengthUs = videoLengthMs * 1000L
    videoLength = videoLengthUs
    val frameIntervalUs = (1000000L / detectedFps).toLong()
    val totalFrames = (videoLengthUs / frameIntervalUs).toInt()

    val firstFrame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)
    if (firstFrame == null) {
        retriever.release()
        return galleryUri
    }
    
    val width = firstFrame.width
    val height = firstFrame.height
    
    initializeMediaPipeBackend(context)

    val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val format = MediaFormat.createVideoFormat("video/avc", width, height)
    format.setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
    format.setInteger(MediaFormat.KEY_FRAME_RATE, detectedFps.toInt())
    format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
    format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

    val encoder = MediaCodec.createEncoderByType("video/avc")
    encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    val inputSurface = encoder.createInputSurface()
    encoder.start()

    var trackIndex = -1
    var muxerStarted = false
    val bufferInfo = MediaCodec.BufferInfo()
    var frameIndex = 0
    var currTimeUs = 0L

    while (currTimeUs <= videoLengthUs) {
        val frame = retriever.getFrameAtTime(currTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        
        if (frame != null) {
            val poseFrame = processFrameWithMediaPipe(frame, frameIndex)
            val modifiedBitmap = drawOnBitmapMediaPipe(frame, poseFrame)
            
            if (poseFrame != null) {
                poseFrames.add(poseFrame)
            }

            val canvas = inputSurface.lockCanvas(null)
            canvas.drawBitmap(modifiedBitmap, 0f, 0f, null)
            inputSurface.unlockCanvasAndPost(canvas)

            while (true) {
                val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
                when {
                    outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (!muxerStarted) {
                            trackIndex = mediaMuxer.addTrack(encoder.outputFormat)
                            mediaMuxer.start()
                            muxerStarted = true
                        }
                    }
                    outputBufferId >= 0 -> {
                        val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: continue
                        if (muxerStarted) {
                            bufferInfo.presentationTimeUs = frameIndex * frameIntervalUs
                            mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                        }
                        encoder.releaseOutputBuffer(outputBufferId, false)
                    }
                    outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                }
            }
            frameIndex++
        }
        
        val progress = ((currTimeUs.toDouble() / videoLengthUs) * 100).toInt().coerceIn(0, 100)
        withContext(Dispatchers.Main) {
            activity.findViewById<ProgressBar>(R.id.splittingBar).progress = progress
            activity.findViewById<TextView>(R.id.splittingProgressValue).text = " $progress%"
        }
        
        currTimeUs += frameIntervalUs
    }

    encoder.signalEndOfInputStream()
    while (true) {
        val outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
        if (outputBufferId >= 0) {
            val outputBuffer = encoder.getOutputBuffer(outputBufferId) ?: break
            if (muxerStarted) {
                mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
            }
            encoder.releaseOutputBuffer(outputBufferId, false)
        } else {
            break
        }
    }

    encoder.stop()
    encoder.release()
    mediaMuxer.stop()
    mediaMuxer.release()
    retriever.release()
    releaseMediaPipeBackend()

    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = GONE
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = GONE
    }

    smoothDataUsingMovingAverage(rightKneeAngles, 5)
    smoothDataUsingMovingAverage(leftKneeAngles, 5)
    smoothDataUsingMovingAverage(rightHipAngles, 5)
    smoothDataUsingMovingAverage(leftHipAngles, 5)
    smoothDataUsingMovingAverage(rightAnkleAngles, 5)
    smoothDataUsingMovingAverage(leftAnkleAngles, 5)
    smoothDataUsingMovingAverage(torsoAngles, 5)
    smoothDataUsingMovingAverage(strideAngles, 5)

    extractGaitFeatures(context, width, height, frameIndex, activity)

    val outputFile = File(outputPath)
    return FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
}

/**
 * Extract 16 gait features using PC pipeline logic.
 * This runs after all frames are processed and pose data is collected.
 * 
 * Implements PC "retry if bad" pattern:
 * 1. Extract features from first-pass pose data
 * 2. If quality != OK and enableROIRetry, reprocess frames with ROI tracking
 * 3. Use whichever result is better
 */
private suspend fun extractGaitFeatures(
    context: Context, 
    frameWidth: Int, 
    frameHeight: Int, 
    totalFrames: Int,
    activity: AppCompatActivity
) {
    Log.d("ImageProcessing", "Starting feature extraction with ${poseFrames.size} pose frames")
    
    if (poseFrames.isEmpty()) {
        Log.w("ImageProcessing", "No pose frames collected, skipping feature extraction")
        return
    }
    
    try {
        // Build PoseSequence from collected frames
        val videoId = galleryUri?.lastPathSegment ?: "unknown"
        val fps = detectedFps  // Use detected FPS instead of hardcoded 30
        
        var poseSequence = PoseSequence(
            videoId = videoId,
            fps = fps,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            numFramesTotal = totalFrames,
            frames = poseFrames.toList()
        )
        
        // Initialize feature extractor with OPTIMAL_CONFIG
        val featureExtractor = FeatureExtractor()
        
        // Normalize walking direction
        poseSequence = featureExtractor.normalizeDirection(poseSequence)
        Log.d("ImageProcessing", "Walking direction: ${poseSequence.walkingDirection}, flipped: ${poseSequence.wasFlipped}")
        
        // First pass: Extract features
        var (features, diagnostics) = featureExtractor.extract(poseSequence)
        var usedRoi = false
        
        // === PC "retry if bad" pattern ===
        // If first pass failed and ROI retry is enabled, reprocess with ROI tracking
        if (features == null && enableROIRetry && diagnostics.qualityFlag != QualityFlag.OK) {
            Log.d("ImageProcessing", "First pass: ${diagnostics.qualityFlag} - retrying with ROI tracking...")
            
            // Update UI to show ROI retry in progress
            withContext(Dispatchers.Main) {
                activity.findViewById<TextView>(R.id.CreationText).text = "Retrying with ROI tracking..."
                activity.findViewById<ProgressBar>(R.id.VideoCreation).progress = 0
                activity.findViewById<TextView>(R.id.CreatingProgressValue).text = " 0%"
            }
            
            val roiResult = reprocessWithRoiTracking(context, frameWidth, frameHeight, totalFrames, fps, activity)
            if (roiResult != null) {
                val (roiPoseSequence, roiFrames) = roiResult
                val normalizedRoiSeq = featureExtractor.normalizeDirection(roiPoseSequence)
                val (roiFeatures, roiDiagnostics) = featureExtractor.extract(normalizedRoiSeq)
                
                // Use ROI result if it's better
                if (roiFeatures != null || roiDiagnostics.numStridesValid > diagnostics.numStridesValid) {
                    features = roiFeatures
                    diagnostics = roiDiagnostics
                    poseSequence = normalizedRoiSeq
                    usedRoi = true
                    Log.d("ImageProcessing", "ROI retry: ${diagnostics.qualityFlag} - using ROI result")
                } else {
                    Log.d("ImageProcessing", "ROI retry: ${roiDiagnostics.qualityFlag} - keeping original")
                }
            }
            
            // Update UI to show completion
            withContext(Dispatchers.Main) {
                activity.findViewById<TextView>(R.id.CreationText).text = "Processing Complete"
                activity.findViewById<ProgressBar>(R.id.VideoCreation).progress = 100
                activity.findViewById<TextView>(R.id.CreatingProgressValue).text = " 100%"
            }
        }
        
        extractedFeatures = features
        extractionDiagnostics = diagnostics
        
        if (features != null) {
            Log.d("ImageProcessing", "Feature extraction successful!${if (usedRoi) " (with ROI)" else ""}")
            Log.d("ImageProcessing", "  Cadence: ${features.cadence_spm} spm")
            Log.d("ImageProcessing", "  Stride time: ${features.stride_time_s} s")
            Log.d("ImageProcessing", "  Knee ROM L/R: ${features.knee_left_rom}° / ${features.knee_right_rom}°")
            Log.d("ImageProcessing", "  Valid strides: ${features.valid_stride_count}")
            
            // Compute gait score
            val scorer = GaitScorer(context)
            if (scorer.initialize()) {
                scoringResult = scorer.score(features)
                Log.d("ImageProcessing", "Gait scores - AE: ${scoringResult?.aeScore}, Ridge: ${scoringResult?.ridgeScore}, PCA: ${scoringResult?.pcaScore}")
                scorer.release()
            } else {
                Log.w("ImageProcessing", "Failed to initialize gait scorer")
            }
        } else {
            Log.w("ImageProcessing", "Feature extraction failed: ${diagnostics.qualityFlag}")
            Log.w("ImageProcessing", "  Reasons: ${diagnostics.rejectionReasons}")
        }
        
    } catch (e: Exception) {
        Log.e("ImageProcessing", "Error during feature extraction", e)
    }
}

/**
 * Reprocess frames with ROI tracking enabled.
 * 
 * Mirrors PC pattern where video is re-extracted with use_roi_tracking=True.
 * Uses the ROITracker state machine: ACQUIRE -> TRACK -> EXPAND -> REACQUIRE
 * 
 * @return Pair of (PoseSequence, list of PoseFrames) or null if failed
 */
private suspend fun reprocessWithRoiTracking(
    context: Context,
    frameWidth: Int,
    frameHeight: Int,
    totalFrames: Int,
    fps: Float,
    activity: AppCompatActivity
): Pair<PoseSequence, List<PoseFrame>>? {
    if (frameList.isEmpty()) return null
    
    val backend = mediaPipeBackend ?: return null
    val roiTracker = GaitVision.com.mediapipe.ROITracker()
    
    val roiPoseFrames = mutableListOf<PoseFrame>()
    var useRoi = false
    var useExpanded = false
    val listSize = frameList.size
    
    Log.d("ImageProcessing", "Reprocessing ${frameList.size} frames with ROI tracking...")
    
    for ((frameIndex, frame) in frameList.withIndex()) {
        // Update progress UI
        val progress = ((frameIndex + 1) * 100 / listSize)
        withContext(Dispatchers.Main) {
            activity.findViewById<ProgressBar>(R.id.VideoCreation).progress = progress
            activity.findViewById<TextView>(R.id.CreatingProgressValue).text = " $progress%"
        }
        val timestampMs = (frameIndex * 1000L / fps).toLong()
        
        // Determine processing region based on ROI state machine
        val processedBitmap = if (useRoi) {
            val roiBounds = roiTracker.getRoiBounds(frameWidth, frameHeight, useExpanded)
            if (roiBounds.width() < frameWidth || roiBounds.height() < frameHeight) {
                // Crop to ROI
                roiTracker.cropToRoi(frame, roiBounds)
            } else {
                frame
            }
        } else {
            frame
        }
        
        // Apply CLAHE if enabled
        val enhancedBitmap = if (enableCLAHE) {
            backend.applyCLAHE(processedBitmap)
        } else {
            processedBitmap
        }
        
        // Process frame
        val result = backend.processFrame(enhancedBitmap, timestampMs)
        val detectionSuccess = result != null && result.landmarks().isNotEmpty()
        
        if (detectionSuccess) {
            val landmarks = result!!.landmarks()[0]
            
            var keypoints = Array(33) { i ->
                floatArrayOf(landmarks[i].x(), landmarks[i].y())
            }
            val confidences = FloatArray(33) { i ->
                landmarks[i].visibility().orElse(0f)
            }
            
            // Map keypoints back to full frame if using ROI
            if (useRoi) {
                val roiBounds = roiTracker.getRoiBounds(frameWidth, frameHeight, useExpanded)
                if (roiBounds.width() < frameWidth || roiBounds.height() < frameHeight) {
                    keypoints = roiTracker.mapKeypointsToFullFrame(
                        keypoints, roiBounds, frameWidth, frameHeight
                    )
                }
            }
            
            roiPoseFrames.add(PoseFrame(
                frameIdx = frameIndex,
                timestampS = timestampMs / 1000f,
                keypoints = keypoints,
                confidences = confidences
            ))
            
            // Update ROI state machine
            val (nextUseRoi, nextUseExpanded) = roiTracker.update(keypoints, confidences, true)
            useRoi = nextUseRoi
            useExpanded = nextUseExpanded
        } else {
            // Update ROI state machine with failure
            val (nextUseRoi, nextUseExpanded) = roiTracker.update(null, null, false)
            useRoi = nextUseRoi
            useExpanded = nextUseExpanded
        }
    }
    
    // Log ROI stats
    val stats = roiTracker.getStats()
    if (stats.isNotEmpty()) {
        Log.d("ImageProcessing", "ROI stats: acquire=${stats["acquire_pct"]}% " +
                "track=${stats["track_pct"]}% expand=${stats["expand_pct"]}% " +
                "reacquire=${stats["reacquire_pct"]}% (reacquires=${stats["reacquire_count"]})")
    }
    
    if (roiPoseFrames.isEmpty()) return null
    
    val videoId = galleryUri?.lastPathSegment ?: "unknown"
    val roiSequence = PoseSequence(
        videoId = "${videoId}_roi",
        fps = fps,
        frameWidth = frameWidth,
        frameHeight = frameHeight,
        numFramesTotal = totalFrames,
        frames = roiPoseFrames
    )
    
    Log.d("ImageProcessing", "ROI reprocessing complete: ${roiPoseFrames.size}/$totalFrames frames detected")
    
    return Pair(roiSequence, roiPoseFrames)
}
