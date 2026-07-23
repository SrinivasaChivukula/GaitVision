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
import android.os.Build
import android.net.Uri
import android.util.Log
import android.view.Surface
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.roundToInt
import kotlin.math.roundToLong
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

// MediaPipe Tasks imports
import GaitVision.com.mediapipe.LegIdentityTracker
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence
import GaitVision.com.mediapipe.MediaPipeResultConverter
import GaitVision.com.mediapipe.PoseOneEuroSmoother
import GaitVision.com.mediapipe.ROITracker

// Gait analysis imports
import GaitVision.com.gait.FeatureExtractor
import GaitVision.com.gait.GaitConfig
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.GaitScorer
import GaitVision.com.gait.PoseJitterAnalyzer
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.QualityFlag

// stage is "processing", "retry", or "done". percent is 0..100 within that stage.
// Lets the caller render progress however it wants (single-video bar, per-row
// status in batch) without dragging view IDs into the pipeline.
typealias ProgressReporter = (stage: String, percent: Int) -> Unit

/**
 * Convert YUV_420_888 Image (from MediaCodec) to ARGB Bitmap.
 * 
 * LOSSLESS direct YUV→RGB conversion to match PC's cv2.VideoCapture behavior.
 * 
 * Previously used JPEG as intermediate (quality=90) which introduced lossy compression
 * artifacts that caused ~0.1% landmark position drift vs PC. This direct conversion
 * eliminates that source of non-parity.
 * 
 * Uses standard BT.601 YUV→RGB formula (same as OpenCV's COLOR_YUV2RGB_NV21):
 *   R = Y + 1.370705 * (V - 128)
 *   G = Y - 0.698001 * (V - 128) - 0.337633 * (U - 128)
 *   B = Y + 1.732446 * (U - 128)
 */
// Reusable buffers for YUV→RGB conversion, allocated once per video
// Eliminates ~1500 large array allocations over a 300-frame video.
private var yBytesCache: ByteArray? = null
private var uBytesCache: ByteArray? = null
private var vBytesCache: ByteArray? = null
private var pixelsCache: IntArray? = null
private var bitmapCache: Bitmap? = null
private val smoothedKps = Array(33) { FloatArray(2) { Float.NaN } }
private const val KP_EMA_ALPHA = 0.2f

private fun imageToBitmap(image: Image): Bitmap {
    val width = image.width
    val height = image.height
    
    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]
    
    val yRowStride = yPlane.rowStride
    val uvRowStride = uPlane.rowStride
    val uvPixelStride = uPlane.pixelStride
    
    // Bulk-copy plane data into reusable byte arrays — one call per plane instead of
    // millions of ByteBuffer.get(index) calls with JVM bounds checking.
    val yBuffer = yPlane.buffer.duplicate().apply { position(0) }
    val uBuffer = uPlane.buffer.duplicate().apply { position(0) }
    val vBuffer = vPlane.buffer.duplicate().apply { position(0) }
    
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val pixelCount = width * height
    
    // Reuse arrays if same size, otherwise allocate new ones (happens once per video)
    val yBytes = yBytesCache?.takeIf { it.size == ySize } ?: ByteArray(ySize).also { yBytesCache = it }
    val uBytes = uBytesCache?.takeIf { it.size == uSize } ?: ByteArray(uSize).also { uBytesCache = it }
    val vBytes = vBytesCache?.takeIf { it.size == vSize } ?: ByteArray(vSize).also { vBytesCache = it }
    val pixels = pixelsCache?.takeIf { it.size == pixelCount } ?: IntArray(pixelCount).also { pixelsCache = it }
    
    yBuffer.get(yBytes, 0, ySize)
    uBuffer.get(uBytes, 0, uSize)
    vBuffer.get(vBytes, 0, vSize)
    
    // Direct YUV→RGB conversion (no JPEG lossy step)
    // Row-invariant values are hoisted out of the inner loop to avoid
    // millions of redundant multiplications and divisions.
    for (row in 0 until height) {
        val yRowOffset = row * yRowStride
        val uvRowOffset = (row shr 1) * uvRowStride  // row/2 via bit shift
        val pixelRowOffset = row * width
        
        for (col in 0 until width) {
            val y = (yBytes[yRowOffset + col].toInt() and 0xFF)
            
            val uvIndex = uvRowOffset + (col shr 1) * uvPixelStride
            val u = (uBytes[uvIndex].toInt() and 0xFF) - 128
            val v = (vBytes[uvIndex].toInt() and 0xFF) - 128
            
            // BT.601 YUV→RGB (matches OpenCV's COLOR_YUV2RGB behavior)
            var r = (y + 1.370705 * v).toInt()
            var g = (y - 0.698001 * v - 0.337633 * u).toInt()
            var b = (y + 1.732446 * u).toInt()
            
            r = r.coerceIn(0, 255)
            g = g.coerceIn(0, 255)
            b = b.coerceIn(0, 255)
            
            pixels[pixelRowOffset + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    
    // Reuse bitmap if same dimensions (avoids allocation + GC per frame)
    val bitmap = bitmapCache?.takeIf { it.width == width && it.height == height && !it.isRecycled }
        ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmapCache = it }
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
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
private var roiMediaPipeBackend: MediaPipePoseBackend? = null

/**
 * Tracks leg-label identity across frames. State resets per processing session.
 * See LegIdentityTracker for the rationale (handles MediaPipe L/R swaps).
 */
private val legIdentityTracker = LegIdentityTracker()
private val adaptiveRoiTracker = ROITracker()
private val poseSmoother = PoseOneEuroSmoother(
    minCutoff = GaitConfig.POSITION_SMOOTH_MIN_CUTOFF,
    beta = GaitConfig.POSITION_SMOOTH_BETA,
    maxMissStreak = GaitConfig.POSITION_SMOOTH_MAX_MISS_STREAK
)
private var previousPoseTimestampS: Float? = null

private class AdaptiveRoiStats {
    var fullFrameAccepted = 0
    var highConfidenceBypass = 0
    var roiAttempted = 0
    var roiAccepted = 0
    var roiRejected = 0
    var roiMissing = 0

    fun reset() {
        fullFrameAccepted = 0
        highConfidenceBypass = 0
        roiAttempted = 0
        roiAccepted = 0
        roiRejected = 0
        roiMissing = 0
    }
}

private val adaptiveRoiStats = AdaptiveRoiStats()

private fun resetPoseProcessingState() {
    synchronized(legIdentityTracker) {
        legIdentityTracker.reset()
        adaptiveRoiTracker.reset()
        adaptiveRoiStats.reset()
        poseSmoother.reset()
        previousPoseTimestampS = null
    }
}

private fun resetProcessingDiagnostics() {
    totalClaheTimeMs = 0L
    totalPureMediaPipeTimeMs = 0L
    totalRoiMediaPipeTimeMs = 0L
    totalDownscaleTimeMs = 0L
    mediaPipeFrameCount = 0
    roiMediaPipeFrameCount = 0
}

private fun correctAndMaybeSmoothPose(frame: PoseFrame, fps: Float = detectedFps): Pair<PoseFrame, PoseFrame> {
    return synchronized(legIdentityTracker) {
        val corrected = legIdentityTracker.correct(frame)
        val fallbackDt = 1f / sanitizeFps(fps)
        val dt = previousPoseTimestampS
            ?.let { frame.timestampS - it }
            ?.takeIf { it.isFinite() && it in (1f / 240f)..(1f / 1f) }
            ?: fallbackDt
        previousPoseTimestampS = frame.timestampS

        // A confirmed L/R mapping flip teleports each leg slot's trajectory to
        // the mirror side. Reset those filters so smoothing restarts at the
        // corrected positions instead of blending across the discontinuity.
        if (enablePositionSmoothing && legIdentityTracker.mappingFlippedOnLastFrame) {
            poseSmoother.resetLandmarks(legIdentityTracker.legLandmarkIndices)
        }

        val output = if (enablePositionSmoothing) {
            poseSmoother.smooth(
                frame = corrected,
                dt = dt,
                minConfidence = GaitConfig.MIN_CONFIDENCE
            )
        } else {
            corrected
        }
        corrected to output
    }
}

/**
 * Detected FPS from video metadata (set during frame extraction).
 */
var detectedFps: Float = 30f
    private set

private fun sanitizeFps(rawFps: Float?): Float {
    val fps = rawFps ?: 30f
    return if (fps.isFinite() && fps in 1f..240f) fps else 30f
}

private fun frameTimestampMs(frameIndex: Int, fps: Float): Long {
    val safeFps = sanitizeFps(fps)
    return (frameIndex * 1000.0 / safeFps).roundToLong()
}

/**
 * Initialize MediaPipe backend for a processing session.
 * Uses OPTIMAL_CONFIG parameters from PC pipeline for feature parity.
 */
/** Detect emulator — GPU delegate doesn't work on emulator OpenGL. */
private fun isEmulator(): Boolean {
    return (Build.FINGERPRINT.startsWith("generic")
            || Build.FINGERPRINT.startsWith("unknown")
            || Build.MODEL.contains("Emulator")
            || Build.MODEL.contains("Android SDK built for x86")
            || Build.HARDWARE.contains("goldfish")
            || Build.HARDWARE.contains("ranchu")
            || Build.PRODUCT.contains("sdk"))
}

fun initializeMediaPipeBackend(context: Context) {
    if (mediaPipeBackend == null) {
        // GPU delegate crashes at runtime on emulator (GL_INVALID_ENUM) — force CPU there
        val onEmulator = isEmulator()
        val useGpu = !forceCpuInference && !onEmulator
        if (onEmulator) {
            Log.d("ImageProcessing", "Emulator detected — forcing CPU delegate")
        }
        mediaPipeBackend = MediaPipePoseBackend(
            context = context,
            minDetectionConfidence = 0.40f,  // OPTIMAL_CONFIG
            minTrackingConfidence = 0.61f,   // OPTIMAL_CONFIG
            minPresenceConfidence = 0.5f,
            useGpu = useGpu
        )
        val delegateType = if (useGpu) "GPU" else "CPU${if (onEmulator) " (emulator)" else " (parity mode)"}"
        Log.d("ImageProcessing", "MediaPipe backend initialized with OPTIMAL_CONFIG, delegate: $delegateType")
    }
}

private fun getRoiMediaPipeBackend(context: Context): MediaPipePoseBackend? {
    if (roiMediaPipeBackend == null) {
        roiMediaPipeBackend = try {
            MediaPipePoseBackend(
                context = context,
                minDetectionConfidence = GaitConfig.MIN_DETECTION_CONFIDENCE,
                minTrackingConfidence = GaitConfig.MIN_TRACKING_CONFIDENCE,
                minPresenceConfidence = GaitConfig.MIN_PRESENCE_CONFIDENCE,
                useGpu = false
            ).also {
                Log.d("ImageProcessing", "Adaptive ROI backend initialized with CPU delegate")
            }
        } catch (e: Exception) {
            Log.w("ImageProcessing", "Adaptive ROI backend failed to initialize: ${e.message}")
            null
        }
    }
    return roiMediaPipeBackend
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
            val safeFps = sanitizeFps(fps)
            Log.d("ImageProcessing", "Detected FPS from metadata: $fps (using $safeFps)")
            safeFps
        } else {
            // Fallback: estimate from duration and frame count
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val videoFrameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            
            if (videoFrameCount != null && videoFrameCount > 0 && durationMs > 0) {
                val estimatedFps = (videoFrameCount * 1000f) / durationMs
                Log.d("ImageProcessing", "Estimated FPS from frame count: $estimatedFps")
                sanitizeFps(estimatedFps.coerceIn(15f, 120f))
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
    roiMediaPipeBackend?.release()
    roiMediaPipeBackend = null
    Log.d("ImageProcessing", "MediaPipe backends released")
}


/**
 * Holds mutable state for video encoding across frames.
 */
private class EncoderState(
    val encoder: MediaCodec,
    val mediaMuxer: MediaMuxer,
    val inputSurface: Surface,
    val bufferInfo: MediaCodec.BufferInfo = MediaCodec.BufferInfo(),
    val frameDurationUs: Long
) {
    var trackIndex: Int = -1
    var muxerStarted: Boolean = false
    private var nextPresentationTimeUs: Long = 0L
    
    /**
     * Write a processed bitmap to the encoder and drain output to muxer.
     */
    fun encodeFrame(bitmap: Bitmap, frameIndex: Int) {
        val canvas = inputSurface.lockCanvas(null)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
        inputSurface.unlockCanvasAndPost(canvas)
        drainEncoder(frameIndex)
    }
    
    /**
     * Drain pending encoder output to muxer.
     */
    fun drainEncoder(frameIndex: Int, timeout: Long = 1000) {
        while (true) {
            val outputId = encoder.dequeueOutputBuffer(bufferInfo, timeout)
            when {
                outputId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!muxerStarted) {
                        trackIndex = mediaMuxer.addTrack(encoder.outputFormat)
                        mediaMuxer.setOrientationHint(0)
                        mediaMuxer.start()
                        muxerStarted = true
                    }
                }
                outputId >= 0 -> {
                    val outputBuffer = encoder.getOutputBuffer(outputId) ?: break
                    if (muxerStarted) {
                        writeEncodedSample(outputBuffer)
                    }
                    encoder.releaseOutputBuffer(outputId, false)
                }
                else -> break
            }
        }
    }
    
    /**
     * Signal end of stream and flush remaining frames.
     */
    fun finishEncoding() {
        encoder.signalEndOfInputStream()
        while (true) {
            val outputId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputId >= 0) {
                val outputBuffer = encoder.getOutputBuffer(outputId) ?: break
                val isEndOfStream = (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                if (muxerStarted) {
                    writeEncodedSample(outputBuffer)
                }
                encoder.releaseOutputBuffer(outputId, false)
                if (isEndOfStream) break
            } else {
                break
            }
        }
    }

    private fun writeEncodedSample(outputBuffer: java.nio.ByteBuffer) {
        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
            bufferInfo.size = 0
        }
        if (bufferInfo.size <= 0) return

        outputBuffer.position(bufferInfo.offset)
        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
        bufferInfo.presentationTimeUs = nextPresentationTimeUs
        mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
        nextPresentationTimeUs += frameDurationUs
    }
    
    /**
     * Release all encoder resources.
     */
    fun release() {
        try {
            encoder.stop()
        } finally {
            inputSurface.release()
            encoder.release()
            if (muxerStarted) {
                mediaMuxer.stop()
            }
            mediaMuxer.release()
        }
    }
}

private fun lowerBodyQuality(frame: PoseFrame?): Float {
    return adaptiveRoiTracker.lowerBodyQuality(frame?.confidences)
}

private fun shouldAcceptRoiPose(fullFrame: PoseFrame?, roiFrame: PoseFrame?): Boolean {
    if (roiFrame == null) return false
    val fullQuality = lowerBodyQuality(fullFrame)
    val roiQuality = lowerBodyQuality(roiFrame)
    return roiQuality >= GaitConfig.MIN_CONFIDENCE &&
        (fullFrame == null || roiQuality >= fullQuality + GaitConfig.ROI_ACCEPT_QUALITY_MARGIN)
}

/**
 * Process a single frame: pose detection, wireframe drawing, store pose data.
 * Returns the modified bitmap ready for encoding.
 */
private fun processFrame(
    context: Context,
    frame: Bitmap,
    frameIndex: Int,
    presentationTimeUs: Long? = null
): Bitmap {
    val rawPoseFrame = processFrameWithAdaptiveRoi(
        context = context,
        frame = frame,
        frameIndex = frameIndex,
        presentationTimeUs = presentationTimeUs
    )
    val poseFrames = rawPoseFrame?.let { correctAndMaybeSmoothPose(it) }
    val poseFrame = poseFrames?.second
    val modifiedBitmap = drawOnBitmapMediaPipe(frame, poseFrame)
    if (poseFrames != null) {
        AnalysisSession.rawPoseFrames.add(poseFrames.first)
        AnalysisSession.poseFrames.add(poseFrames.second)
    }
    return modifiedBitmap
}

// Timing accumulators for CLAHE vs pure MediaPipe (for diagnostics)
var totalClaheTimeMs = 0L
var totalPureMediaPipeTimeMs = 0L
var totalRoiMediaPipeTimeMs = 0L
var totalDownscaleTimeMs = 0L
var mediaPipeFrameCount = 0
var roiMediaPipeFrameCount = 0

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
    presentationTimeUs: Long? = null,
    fps: Float = detectedFps,
    applyClahe: Boolean = enableCLAHE
): PoseFrame? {
    val backend = mediaPipeBackend ?: return null

    val timestampMs = presentationTimeUs
        ?.let { (it / 1000.0).roundToLong() }
        ?: frameTimestampMs(frameIdx, fps)

    return processBitmapWithMediaPipe(
        bitmap = bitmap,
        frameIdx = frameIdx,
        timestampMs = timestampMs,
        backend = backend,
        applyClahe = applyClahe,
        countAsRoi = false
    )
}

private fun processBitmapWithMediaPipe(
    bitmap: Bitmap,
    frameIdx: Int,
    timestampMs: Long,
    backend: MediaPipePoseBackend,
    applyClahe: Boolean,
    countAsRoi: Boolean
): PoseFrame? {
    // Always downscale to 720p for inference, MediaPipe internally resizes to ~256x256
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
    
    t0 = System.currentTimeMillis()
    val result = backend.processFrame(processedBitmap, timestampMs)
    val elapsed = System.currentTimeMillis() - t0
    if (countAsRoi) {
        totalRoiMediaPipeTimeMs += elapsed
        roiMediaPipeFrameCount++
    } else {
        totalPureMediaPipeTimeMs += elapsed
        mediaPipeFrameCount++
    }
    
    // Recycle intermediate bitmaps that are separate allocations from the input
    if (applyClahe && processedBitmap !== scaledBitmap) {
        processedBitmap.recycle()
    }
    if (scaledBitmap !== bitmap) {
        scaledBitmap.recycle()
    }
    
    return MediaPipeResultConverter.toPoseFrame(
        result = result,
        frameIdx = frameIdx,
        timestampS = timestampMs / 1000f
    )
}

private fun processFrameWithAdaptiveRoi(
    context: Context,
    frame: Bitmap,
    frameIndex: Int,
    presentationTimeUs: Long? = null,
    fps: Float = detectedFps,
    applyClahe: Boolean = enableCLAHE
): PoseFrame? {
    val timestampMs = presentationTimeUs
        ?.let { (it / 1000.0).roundToLong() }
        ?: frameTimestampMs(frameIndex, fps)

    val fullFrame = processFrameWithMediaPipe(
        bitmap = frame,
        frameIdx = frameIndex,
        presentationTimeUs = presentationTimeUs,
        fps = fps,
        applyClahe = applyClahe
    )
    val fullQuality = lowerBodyQuality(fullFrame)
    val shouldTryRoi = adaptiveRoiTracker.shouldUseRoi() &&
        fullQuality < GaitConfig.ROI_HIGH_CONFIDENCE_BYPASS

    var acceptedFrame = fullFrame
    var decision = "full"

    if (fullQuality >= GaitConfig.ROI_HIGH_CONFIDENCE_BYPASS) {
        adaptiveRoiStats.highConfidenceBypass++
    } else if (shouldTryRoi) {
        adaptiveRoiStats.roiAttempted++
        val roiFrame = processFrameWithRoi(
            context = context,
            frame = frame,
            frameIndex = frameIndex,
            timestampMs = timestampMs,
            applyClahe = applyClahe
        )

        if (shouldAcceptRoiPose(fullFrame, roiFrame)) {
            acceptedFrame = roiFrame
            adaptiveRoiStats.roiAccepted++
            decision = "roi"
        } else {
            if (roiFrame == null) {
                adaptiveRoiStats.roiMissing++
            } else {
                adaptiveRoiStats.roiRejected++
            }
        }
    }

    if (decision == "full") {
        adaptiveRoiStats.fullFrameAccepted++
    }

    adaptiveRoiTracker.update(
        keypoints = acceptedFrame?.keypoints,
        confidences = acceptedFrame?.confidences,
        detectionSuccess = acceptedFrame != null
    )

    if (enableVerboseLogging && frameIndex % 30 == 0) {
        Log.d(
            "ImageProcessing",
            "Adaptive ROI frame=$frameIndex state=${adaptiveRoiTracker.getCurrentState()} " +
                "decision=$decision fullQ=$fullQuality acceptedQ=${lowerBodyQuality(acceptedFrame)}"
        )
    }

    return acceptedFrame
}

private fun processFrameWithRoi(
    context: Context,
    frame: Bitmap,
    frameIndex: Int,
    timestampMs: Long,
    applyClahe: Boolean
): PoseFrame? {
    val backend = getRoiMediaPipeBackend(context) ?: return null
    val roiBounds = adaptiveRoiTracker.getRoiBounds(
        frameWidth = frame.width,
        frameHeight = frame.height,
        useExpanded = adaptiveRoiTracker.shouldUseExpandedRoi()
    )
    if (roiBounds.width() >= frame.width && roiBounds.height() >= frame.height) {
        return null
    }

    val roiCrop = adaptiveRoiTracker.cropToRoi(frame, roiBounds)
    return try {
        processBitmapWithMediaPipe(
            bitmap = roiCrop.bitmap,
            frameIdx = frameIndex,
            timestampMs = timestampMs,
            backend = backend,
            applyClahe = applyClahe,
            countAsRoi = true
        )?.let { roiPose ->
            adaptiveRoiTracker.mapPoseFrameToFullFrame(
                frame = roiPose,
                roiCrop = roiCrop,
                frameWidth = frame.width,
                frameHeight = frame.height
            )
        }
    } finally {
        if (!roiCrop.bitmap.isRecycled) {
            roiCrop.bitmap.recycle()
        }
    }
}

private fun logAdaptiveRoiStats(tag: String, totalFrames: Int) {
    val trackerStats = adaptiveRoiTracker.getStats()
    val roiAttempted = adaptiveRoiStats.roiAttempted
    val roiAccepted = adaptiveRoiStats.roiAccepted
    val roiAcceptPct = if (roiAttempted > 0) {
        roiAccepted * 100f / roiAttempted
    } else {
        0f
    }
    val roiProbeAvgMs = if (roiMediaPipeFrameCount > 0) {
        totalRoiMediaPipeTimeMs.toFloat() / roiMediaPipeFrameCount
    } else {
        0f
    }
    Log.d(
        tag,
        "Adaptive ROI summary: frames=$totalFrames " +
            "fullAccepted=${adaptiveRoiStats.fullFrameAccepted}, " +
            "highConfidenceBypass=${adaptiveRoiStats.highConfidenceBypass}, " +
            "roiAttempted=$roiAttempted, roiAccepted=$roiAccepted " +
            "(${String.format("%.1f", roiAcceptPct)}%), " +
            "roiRejected=${adaptiveRoiStats.roiRejected}, roiMissing=${adaptiveRoiStats.roiMissing}, " +
            "roiProbeFrames=$roiMediaPipeFrameCount, roiProbeAvgMs=${String.format("%.1f", roiProbeAvgMs)}, " +
            "tracker=$trackerStats"
    )
}

// drawOnBitmapMediaPipe moved to WireframeRenderer.kt

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
suspend fun ProcVidEmpty(
    context: Context,
    outputPath: String,
    onProgress: ProgressReporter? = null
): Uri? {
    val TAG = "ImageProcessing"
    loadAnalysisOptions(context)
    
    // Clear all data
    AnalysisSession.rawPoseFrames.clear()
    AnalysisSession.poseFrames.clear()
    AnalysisSession.frameList.clear()
    smoothedKps.forEach { it.fill(Float.NaN) }
    AnalysisSession.extractedFeatures = null
    AnalysisSession.extractionDiagnostics = null
    AnalysisSession.scoringResult = null
    AnalysisSession.jitterComparison = null
    AnalysisSession.extractedSignals = null
    resetPoseProcessingState()
    resetProcessingDiagnostics()

    if (AnalysisSession.galleryUri == null) {
        Log.e(TAG, "No video URI provided")
        return null
    }

    onProgress?.invoke("processing", 0)

    // === Set up MediaExtractor for FAST video reading ===
    val extractor = MediaExtractor()
    val retriever = MediaMetadataRetriever()  // For FPS detection fallback
    
    try {
        val pfd = context.contentResolver.openFileDescriptor(AnalysisSession.galleryUri!!, "r")
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
            extractor.setDataSource(context, AnalysisSession.galleryUri!!, null)
            retriever.setDataSource(context, AnalysisSession.galleryUri)
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
        return AnalysisSession.galleryUri
    }
    
    extractor.selectTrack(videoTrackIndex)
    
    // Get video properties
    val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
    val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
    val videoMime = videoFormat.getString(MediaFormat.KEY_MIME) ?: "video/avc"
    val durationUs = videoFormat.getLong(MediaFormat.KEY_DURATION)
    val videoLengthMs = durationUs / 1000
    AnalysisSession.videoLength = durationUs
    
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
    fps = sanitizeFps(fps)
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
        return procVidEmptyFallback(context, outputPath, onProgress)
    }
    
    // Initialize MediaPipe
    initializeMediaPipeBackend(context)
    
    // === Set up video encoder ===
    val mediaMuxer: MediaMuxer
    val encoder: MediaCodec
    val inputSurface: Surface
    
    try {
        mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val format = MediaFormat.createVideoFormat("video/avc", width, height)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, fps.roundToInt().coerceAtLeast(1))
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

    val frameDurationUs = (1_000_000.0 / fps).roundToLong()
    val encoderState = EncoderState(
        encoder = encoder,
        mediaMuxer = mediaMuxer,
        inputSurface = inputSurface,
        frameDurationUs = frameDurationUs,
    )
    val decoderBufferInfo = MediaCodec.BufferInfo()
    var frameIndex = 0
    var inputDone = false
    var outputDone = false
    val startTime = System.currentTimeMillis()
    
    Log.d(TAG, "FAST STREAMING: Processing ~$totalFrames frames with MediaCodec")
    Log.d(TAG, "GPU delegate: ${mediaPipeBackend?.isUsingGpu() ?: false}, CLAHE: $enableCLAHE, position smoothing: $enablePositionSmoothing")

    // only fire the callback when the integer percent actually changes
    var lastProgress = -1

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
                    try {
                        val image = decoder.getOutputImage(outputBufferId)
                        if (image != null) {
                            // Convert YUV to Bitmap, then release decoder buffer immediately.
                            // Pixel data is fully copied into our Bitmap, so the decoder
                            // can start decoding the next frame while we run MediaPipe.
                            val frame: Bitmap
                            try {
                                frame = imageToBitmap(image)
                            } finally {
                                image.close()
                                decoder.releaseOutputBuffer(outputBufferId, false)
                            }
                            
                            // Process frame (pose detection + wireframe) and encode
                            val modifiedBitmap = processFrame(
                                context = context,
                                frame = frame,
                                frameIndex = frameIndex,
                                presentationTimeUs = decoderBufferInfo.presentationTimeUs
                            )
                            encoderState.encodeFrame(modifiedBitmap, frameIndex)
                            // Note: frame/modifiedBitmap is reused via bitmapCache — don't recycle
                            
                            // Update progress only when percentage actually changes
                            frameIndex++
                            val progress = ((frameIndex.toFloat() / totalFrames) * 100).toInt().coerceIn(0, 100)
                            if (progress != lastProgress) {
                                lastProgress = progress
                                onProgress?.invoke("processing", progress)
                            }
                        } else {
                            decoder.releaseOutputBuffer(outputBufferId, false)
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Error processing frame $frameIndex: ${e.message}")
                    }
                }
            }
        }
    }
    
    val elapsedSec = (System.currentTimeMillis() - startTime) / 1000.0
    Log.d(TAG, "Processed $frameIndex frames in ${elapsedSec}s (${String.format("%.1f", frameIndex/elapsedSec)} fps)")

    // Finish encoding and release resources
    encoderState.finishEncoding()
    encoderState.release()
    decoder.stop()
    decoder.release()
    extractor.release()
    releaseMediaPipeBackend()

    onProgress?.invoke("processing", 100)

    Log.d(TAG, "FAST STREAMING complete. Processed $frameIndex frames, ${AnalysisSession.poseFrames.size} poses detected")
    logAdaptiveRoiStats(TAG, frameIndex)
    
    // Feature extraction (uses poseFrames which is small)
    extractGaitFeatures(context, width, height, frameIndex, onProgress)

    // Snapshot pose frames before they get cleared below (multiview flow reads this)
    AnalysisSession.lastPoseFramesSnapshot = AnalysisSession.poseFrames.toList()

    // Free heavy memory now that processing is done
    val frameCount = AnalysisSession.poseFrames.size
    AnalysisSession.rawPoseFrames.clear()
    AnalysisSession.frameList.clear()
    AnalysisSession.poseFrames.clear()
    smoothedKps.forEach { it.fill(Float.NaN) }

    // Release reusable YUV conversion buffers
    bitmapCache?.recycle()
    bitmapCache = null
    yBytesCache = null
    uBytesCache = null
    vBytesCache = null
    pixelsCache = null
    Log.d(TAG, "Cleared frameList and poseFrames ($frameCount poses freed)")
    
    Log.d(TAG, "Pipeline complete")
    onProgress?.invoke("done", 100)

    val outputFile = File(outputPath)
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
    Log.d(TAG, "Generated URI: $uri")
    return uri
}

/**
 * Fallback video processing using slow getFrameAtTime() method.
 * Used if MediaCodec initialization fails.
 */
private suspend fun procVidEmptyFallback(
    context: Context,
    outputPath: String,
    onProgress: ProgressReporter? = null
): Uri? {
    val TAG = "ImageProcessing"
    Log.w(TAG, "Using SLOW fallback method (getFrameAtTime)")

    // Detect video FPS
    detectedFps = withContext(Dispatchers.IO) {
        sanitizeFps(detectVideoFps(context, AnalysisSession.galleryUri))
    }

    // Mirror the reset block from ProcVidEmpty — the singleton
    // legIdentityTracker retains per-video state from any prior run, and
    // without this reset the first frame of the new video would be classified
    // against the previous video's prev PoseFrame and swapVotes.
    AnalysisSession.rawPoseFrames.clear()
    AnalysisSession.poseFrames.clear()
    AnalysisSession.frameList.clear()
    AnalysisSession.extractedFeatures = null
    AnalysisSession.extractionDiagnostics = null
    AnalysisSession.scoringResult = null
    AnalysisSession.jitterComparison = null
    AnalysisSession.extractedSignals = null
    resetPoseProcessingState()
    resetProcessingDiagnostics()

    val retriever = MediaMetadataRetriever()
    try {
        val pfd = context.contentResolver.openFileDescriptor(AnalysisSession.galleryUri!!, "r")
        if (pfd != null) {
            retriever.setDataSource(pfd.fileDescriptor)
            pfd.close()
        } else {
            retriever.setDataSource(context, AnalysisSession.galleryUri)
        }
    } catch (e: Exception) {
        retriever.setDataSource(context, AnalysisSession.galleryUri)
    }

    val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
    val videoLengthUs = videoLengthMs * 1000L
    AnalysisSession.videoLength = videoLengthUs
    val frameIntervalUs = (1_000_000.0 / detectedFps).roundToLong()
    val totalFrames = (videoLengthUs / frameIntervalUs).toInt()

    val firstFrame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)
    if (firstFrame == null) {
        retriever.release()
        return AnalysisSession.galleryUri
    }
    
    val width = firstFrame.width
    val height = firstFrame.height
    
    initializeMediaPipeBackend(context)

    // Set up encoder using shared helper
    val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
        setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
        setInteger(MediaFormat.KEY_FRAME_RATE, detectedFps.roundToInt().coerceAtLeast(1))
        setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
    }
    val encoder = MediaCodec.createEncoderByType("video/avc")
    encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    val inputSurface = encoder.createInputSurface()
    encoder.start()
    
    val encoderState = EncoderState(
        encoder = encoder,
        mediaMuxer = mediaMuxer,
        inputSurface = inputSurface,
        frameDurationUs = frameIntervalUs,
    )
    var frameIndex = 0
    var currTimeUs = 0L

    while (currTimeUs <= videoLengthUs) {
        val frame = retriever.getFrameAtTime(currTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        
        if (frame != null) {
            val modifiedBitmap = processFrame(
                context = context,
                frame = frame,
                frameIndex = frameIndex,
                presentationTimeUs = currTimeUs
            )
            encoderState.encodeFrame(modifiedBitmap, frameIndex)
            frameIndex++
        }
        
        val progress = ((currTimeUs.toDouble() / videoLengthUs) * 100).toInt().coerceIn(0, 100)
        onProgress?.invoke("processing", progress)
        
        currTimeUs += frameIntervalUs
    }

    encoderState.finishEncoding()
    encoderState.release()
    retriever.release()
    releaseMediaPipeBackend()

    logAdaptiveRoiStats(TAG, frameIndex)

    onProgress?.invoke("processing", 100)

    extractGaitFeatures(context, width, height, frameIndex, onProgress)

    AnalysisSession.rawPoseFrames.clear()
    AnalysisSession.frameList.clear()
    AnalysisSession.poseFrames.clear()

    val outputFile = File(outputPath)
    onProgress?.invoke("done", 100)
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
    onProgress: ProgressReporter? = null
) {
    Log.d("ImageProcessing", "Starting feature extraction with ${AnalysisSession.poseFrames.size} pose frames")

    if (AnalysisSession.poseFrames.isEmpty()) {
        Log.w("ImageProcessing", "No pose frames collected, skipping feature extraction")
        return
    }

    try {
        // Build PoseSequence from collected frames
        val videoId = AnalysisSession.galleryUri?.lastPathSegment ?: "unknown"
        val fps = detectedFps  // Use detected FPS instead of hardcoded 30

        var poseSequence = PoseSequence(
            videoId = videoId,
            fps = fps,
            frameWidth = frameWidth,
            frameHeight = frameHeight,
            numFramesTotal = totalFrames,
            frames = AnalysisSession.poseFrames.toList()
        )

        val rawPoseFrames = AnalysisSession.rawPoseFrames.toList()
        if (rawPoseFrames.isNotEmpty()) {
            val rawPoseSequence = PoseSequence(
                videoId = videoId,
                fps = fps,
                frameWidth = frameWidth,
                frameHeight = frameHeight,
                numFramesTotal = totalFrames,
                frames = rawPoseFrames
            )
            AnalysisSession.jitterComparison = PoseJitterAnalyzer.compare(
                raw = rawPoseSequence,
                smoothed = poseSequence
            )
            AnalysisSession.jitterComparison?.let { jitter ->
                Log.d(
                    "ImageProcessing",
                    "Pose jitter: raw=${jitter.raw.jitterSecondDiffNorm}, " +
                        "smoothed=${jitter.smoothed.jitterSecondDiffNorm}, " +
                        "reduction=${jitter.jitterReductionPct}%"
                )
            }
        }
        
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
            onProgress?.invoke("retry", 0)
            
            val roiResult = reprocessWithRoiTracking(context, frameWidth, frameHeight, totalFrames, fps, onProgress)
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
            
            onProgress?.invoke("retry", 100)
        }
        
        AnalysisSession.extractedFeatures = features
        AnalysisSession.extractionDiagnostics = diagnostics

        if (features != null) {
            Log.d("ImageProcessing", "Feature extraction successful!${if (usedRoi) " (with ROI)" else ""}")
            Log.d("ImageProcessing", "  Cadence: ${features.cadence_spm} spm")
            Log.d("ImageProcessing", "  Stride time: ${features.stride_time_s} s")
            Log.d("ImageProcessing", "  Knee ROM L/R: ${features.knee_left_rom}° / ${features.knee_right_rom}°")
            Log.d("ImageProcessing", "  Valid strides: ${features.valid_stride_count}")

            // Compute gait score
            val scorer = GaitScorer(context)
            if (scorer.initialize()) {
                AnalysisSession.scoringResult = scorer.score(features)
                Log.d("ImageProcessing", "Gait scores - AE: ${AnalysisSession.scoringResult?.aeScore}, Ridge: ${AnalysisSession.scoringResult?.ridgeScore}, PCA: ${AnalysisSession.scoringResult?.pcaScore}")
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
 * LEGACY/OFF - Reprocess frames with ROI tracking enabled after feature failure.
 * 
 * STATUS: Kept disabled. The active ROI path now runs conservatively during
 * first-pass processing; this retry path still depends on frameList and should
 * not be enabled unless it is rewritten to re-decode frames.
 * 
 * Mirrors PC pattern where video is re-extracted with use_roi_tracking=True.
 * Uses the ROITracker state machine: ACQUIRE -> TRACK -> EXPAND -> REACQUIRE
 * 
 * KNOWN ISSUE: This function requires frameList to be populated, but the fast
 * MediaCodec path streams frames without storing them. frameList is cleared
 * but never populated, so this function returns null in the normal fast path.
 * 
 * TO FIX (future work):
 *   Option A: Store frames during fast path (memory expensive ~500MB for 10s video)
 *   Option B: Re-decode video in this function (slower but correct)
 * 
 * @return Pair of (PoseSequence, list of PoseFrames) or null if failed
 */
private suspend fun reprocessWithRoiTracking(
    context: Context,
    frameWidth: Int,
    frameHeight: Int,
    totalFrames: Int,
    fps: Float,
    onProgress: ProgressReporter? = null
): Pair<PoseSequence, List<PoseFrame>>? {
    // frameList is empty in the fast path - see docstring above
    if (AnalysisSession.frameList.isEmpty()) return null

    val backend = mediaPipeBackend ?: return null
    val roiTracker = GaitVision.com.mediapipe.ROITracker()

    // Reset the leg-identity tracker so the ROI pass produces a consistent
    // L/R labelling within itself (and matches the first pass's convention).
    // The singleton tracker would otherwise still hold the last video's state
    // when this path is (re)enabled.
    resetPoseProcessingState()

    val roiPoseFrames = mutableListOf<PoseFrame>()
    var useRoi = false
    var useExpanded = false
    val listSize = AnalysisSession.frameList.size

    Log.d("ImageProcessing", "Reprocessing ${AnalysisSession.frameList.size} frames with ROI tracking...")

    for ((frameIndex, frame) in AnalysisSession.frameList.withIndex()) {
        val progress = ((frameIndex + 1) * 100 / listSize)
        onProgress?.invoke("retry", progress)
        val timestampMs = frameTimestampMs(frameIndex, fps)

        // Determine processing region based on ROI state machine
        var roiCrop: ROITracker.RoiCrop? = null
        val processedBitmap = if (useRoi) {
            val roiBounds = roiTracker.getRoiBounds(frameWidth, frameHeight, useExpanded)
            if (roiBounds.width() < frameWidth || roiBounds.height() < frameHeight) {
                // Crop to ROI
                roiCrop = roiTracker.cropToRoi(frame, roiBounds)
                roiCrop!!.bitmap
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

            val keypoints = Array(33) { i ->
                floatArrayOf(landmarks[i].x(), landmarks[i].y())
            }
            val confidences = FloatArray(33) { i ->
                landmarks[i].visibility().orElse(0f)
            }

            val roiPoseFrame = PoseFrame(
                frameIdx = frameIndex,
                timestampS = timestampMs / 1000f,
                keypoints = keypoints,
                confidences = confidences
            )
            val rawPoseFrame = roiCrop?.let {
                roiTracker.mapPoseFrameToFullFrame(
                    frame = roiPoseFrame,
                    roiCrop = it,
                    frameWidth = frameWidth,
                    frameHeight = frameHeight
                )
            } ?: roiPoseFrame

            // Run through LegIdentityTracker so the ROI sequence shares the
            // same L/R convention as the first pass — otherwise downstream
            // normalizeDirection / stride metrics would see inconsistent
            // labels within a single session.
            val poseFrame = correctAndMaybeSmoothPose(rawPoseFrame, fps).second
            roiPoseFrames.add(poseFrame)

            // Update ROI state machine
            val (nextUseRoi, nextUseExpanded) = roiTracker.update(
                rawPoseFrame.keypoints,
                rawPoseFrame.confidences,
                true
            )
            useRoi = nextUseRoi
            useExpanded = nextUseExpanded
        } else {
            // Update ROI state machine with failure
            val (nextUseRoi, nextUseExpanded) = roiTracker.update(null, null, false)
            useRoi = nextUseRoi
            useExpanded = nextUseExpanded
        }
        if (processedBitmap !== frame && !processedBitmap.isRecycled) {
            processedBitmap.recycle()
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

    val videoId = AnalysisSession.galleryUri?.lastPathSegment ?: "unknown"
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
