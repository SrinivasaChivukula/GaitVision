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
 * Global MediaPipe backend instance (initialized once per video processing session).
 */
private var mediaPipeBackend: MediaPipePoseBackend? = null

/**
 * Detected FPS from video metadata (set during frame extraction).
 */
var detectedFps: Float = 30f
    private set

/** Open a video URI via PFD, falling back to content resolver. */
private fun setDataSourceSafe(context: Context, uri: Uri, vararg targets: Any) {
    try {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
        if (pfd != null) {
            for (t in targets) {
                when (t) {
                    is MediaExtractor -> t.setDataSource(pfd.fileDescriptor)
                    is MediaMetadataRetriever -> t.setDataSource(pfd.fileDescriptor)
                }
            }
            pfd.close()
            return
        }
    } catch (_: Exception) {}
    // Fallback
    for (t in targets) {
        when (t) {
            is MediaExtractor -> t.setDataSource(context, uri, null)
            is MediaMetadataRetriever -> t.setDataSource(context, uri)
        }
    }
}

/** Create an encoder + muxer for output video. */
private fun createEncoderState(outputPath: String, width: Int, height: Int, fps: Float): EncoderState {
    val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
        setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
        setInteger(MediaFormat.KEY_FRAME_RATE, fps.toInt())
        setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
    }
    val encoder = MediaCodec.createEncoderByType("video/avc")
    encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    val inputSurface = encoder.createInputSurface()
    encoder.start()
    val frameDurationUs = (1000000.0 / fps).toLong()
    return EncoderState(encoder, mediaMuxer, inputSurface, frameDurationUs = frameDurationUs)
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

/**
 * Detect actual FPS from video metadata.
 * Falls back to 30 FPS if detection fails (mirrors PC behavior).
 */
fun detectVideoFps(context: Context, fileUri: Uri?): Float {
    if (fileUri == null) return 30f
    
    val retriever = MediaMetadataRetriever()
    return try {
        setDataSourceSafe(context, fileUri, retriever)
        
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


/** Map a MediaPipe landmark index to its left/right mirror. */
fun getLRSwapIndex(idx: Int): Int {
    return when (idx) {
        1 -> 4; 4 -> 1;  2 -> 5; 5 -> 2;  3 -> 6; 6 -> 3
        7 -> 8; 8 -> 7;  9 -> 10; 10 -> 9
        11 -> 12; 12 -> 11;  13 -> 14; 14 -> 13
        15 -> 16; 16 -> 15;  17 -> 18; 18 -> 17
        19 -> 20; 20 -> 19;  21 -> 22; 22 -> 21
        23 -> 24; 24 -> 23;  25 -> 26; 26 -> 25
        27 -> 28; 28 -> 27;  29 -> 30; 30 -> 29
        31 -> 32; 32 -> 31
        else -> idx
    }
}

/**
 * Fixes per-frame L/R landmark swaps caused by clothing/occlusion/angle.
 * Compares motion continuity cost (ankle+knee+hip) to detect when MediaPipe
 * assigned landmarks to the wrong side. Must run BEFORE normalizeDirection().
 *
 * Only swaps when cost_swap < 0.8 * cost_keep (symmetric hysteresis).
 * Low-confidence frames are skipped and carry forward the previous assignment.
 */
fun stabilizeLandmarkIdentity(poseSeq: PoseSequence): PoseSequence {
    val TAG = "IdentityStab"
    val MIN_CONF = GaitVision.com.gait.GaitConfig.MIN_CONFIDENCE
    val HYSTERESIS = 0.8f

    val trackedPairs = listOf(
        Pair(MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE),   // 27, 28
        Pair(MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.RIGHT_KNEE),     // 25, 26
        Pair(MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP)        // 23, 24
    )

    val frames = poseSeq.frames
    if (frames.size < 2) return poseSeq

    val stabilizedFrames = frames.toMutableList()
    var totalSwaps = 0
    var currentRunLength = 0
    var maxRunLength = 0
    var lastEvalIdx = 0  // index of last frame where we made a swap decision

    for (t in 1 until frames.size) {
        val prev = stabilizedFrames[lastEvalIdx]
        val curr = frames[t]

        val allConfident = trackedPairs.all { (l, r) ->
            curr.confidences[l] >= MIN_CONF && curr.confidences[r] >= MIN_CONF &&
            prev.confidences[l] >= MIN_CONF && prev.confidences[r] >= MIN_CONF
        }

        if (!allConfident) {
            stabilizedFrames[t] = curr  // low confidence, carry forward
            continue
        }

        var costKeep = 0f
        var costSwap = 0f
        for ((l, r) in trackedPairs) {
            val currL = curr.keypoints[l]
            val currR = curr.keypoints[r]
            val prevL = prev.keypoints[l]
            val prevR = prev.keypoints[r]

            costKeep += dist2d(currL, prevL) + dist2d(currR, prevR)
            costSwap += dist2d(currL, prevR) + dist2d(currR, prevL)
        }

        if (costSwap < HYSTERESIS * costKeep) {
            val swappedKeypoints = Array(33) { floatArrayOf(0f, 0f) }
            val swappedConfidences = FloatArray(33)
            for (i in 0 until 33) {
                val swapIdx = getLRSwapIndex(i)
                swappedKeypoints[swapIdx] = curr.keypoints[i].clone()
                swappedConfidences[swapIdx] = curr.confidences[i]
            }
            stabilizedFrames[t] = curr.copy(keypoints = swappedKeypoints, confidences = swappedConfidences)
            totalSwaps++
            currentRunLength++
            maxRunLength = maxOf(maxRunLength, currentRunLength)
        } else {
            stabilizedFrames[t] = curr
            currentRunLength = 0
        }

        lastEvalIdx = t
    }

    val swapRate = if (frames.size > 1) totalSwaps.toFloat() / (frames.size - 1) else 0f
    Log.d(TAG, "Identity stabilization: ${frames.size} frames, $totalSwaps swaps " +
        "(${String.format("%.1f", swapRate * 100)}%), longest run: $maxRunLength")

    return poseSeq.copy(frames = stabilizedFrames)
}

/** Euclidean distance between two 2D keypoints. */
private fun dist2d(a: FloatArray, b: FloatArray): Float {
    val dx = a[0] - b[0]
    val dy = a[1] - b[1]
    return kotlin.math.sqrt(dx * dx + dy * dy)
}

/**
 * Hide progress UI elements after processing.
 */
private suspend fun hideProgressUI(activity: AppCompatActivity) {
    withContext(Dispatchers.Main) {
        activity.findViewById<TextView>(R.id.SplittingText).visibility = GONE
        activity.findViewById<ProgressBar>(R.id.splittingBar).visibility = GONE
        activity.findViewById<TextView>(R.id.splittingProgressValue).visibility = GONE
    }
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
    
    /**
     * Write a processed bitmap to the encoder and drain output to muxer.
     */
    fun encodeFrame(bitmap: Bitmap, frameIndex: Int) {
        // Draw to encoder surface
        val canvas = inputSurface.lockCanvas(null)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
        inputSurface.unlockCanvasAndPost(canvas)
        
        // Drain encoder output
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
                        bufferInfo.presentationTimeUs = frameIndex * frameDurationUs
                        mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
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
        // Drain with longer timeout for final frames
        while (true) {
            val outputId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputId >= 0) {
                val outputBuffer = encoder.getOutputBuffer(outputId) ?: break
                if (muxerStarted) {
                    mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                }
                encoder.releaseOutputBuffer(outputId, false)
            } else {
                break
            }
        }
    }
    
    /**
     * Release all encoder resources.
     */
    fun release() {
        encoder.stop()
        encoder.release()
        mediaMuxer.stop()
        mediaMuxer.release()
    }
}

/**
 * Process a single frame: pose detection, wireframe drawing, store pose data.
 * Returns the modified bitmap ready for encoding.
 */
private fun processFrame(frame: Bitmap, frameIndex: Int): Bitmap {
    val poseFrame = processFrameWithMediaPipe(frame, frameIndex)
    val modifiedBitmap = drawOnBitmapMediaPipe(frame, poseFrame)
    if (poseFrame != null) {
        poseFrames.add(poseFrame)
    }
    return modifiedBitmap
}

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
    
    // Downscale to 720p for inference
    val scaledBitmap = if (bitmap.width > PROCESSING_WIDTH || bitmap.height > PROCESSING_HEIGHT) {
        Bitmap.createScaledBitmap(bitmap, PROCESSING_WIDTH, PROCESSING_HEIGHT, true)
    } else {
        bitmap
    }
    
    // Optionally apply CLAHE contrast enhancement
    val processedBitmap = if (applyClahe) {
        backend.applyCLAHE(scaledBitmap)
    } else {
        scaledBitmap
    }
    
    val timestampMs = (frameIdx * 1000L / fps).toLong()
    val result = backend.processFrame(processedBitmap, timestampMs)
    
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
    poseFrames.clear()
    extractedFeatures = null
    extractionDiagnostics = null
    scoringResult = null
    extractedSignals = null
    
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
    val retriever = MediaMetadataRetriever()
    try {
        setDataSourceSafe(context, galleryUri!!, extractor, retriever)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to open video: ${e.message}")
        return null
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
    val encoderState: EncoderState
    try {
        encoderState = createEncoderState(outputPath, width, height, fps)
        encoderState.mediaMuxer.setOrientationHint(0)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize encoder: ${e.message}", e)
        releaseMediaPipeBackend()
        decoder.stop()
        decoder.release()
        extractor.release()
        return null
    }
    val decoderBufferInfo = MediaCodec.BufferInfo()
    var frameIndex = 0
    var inputDone = false
    var outputDone = false
    val startTime = System.currentTimeMillis()
    
    Log.d(TAG, "FAST STREAMING: Processing ~$totalFrames frames with MediaCodec")
    Log.d(TAG, "GPU delegate: ${mediaPipeBackend?.isUsingGpu() ?: false}, CLAHE: $enableCLAHE")

    // Cache view references and progress state to avoid repeated findViewById + unnecessary context switches
    val progressBar = activity.findViewById<ProgressBar>(R.id.splittingBar)
    val progressText = activity.findViewById<TextView>(R.id.splittingProgressValue)
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
                            val modifiedBitmap = processFrame(frame, frameIndex)
                            encoderState.encodeFrame(modifiedBitmap, frameIndex)
                            // Note: frame/modifiedBitmap is reused via bitmapCache — don't recycle
                            
                            // Update progress only when percentage actually changes
                            frameIndex++
                            val progress = ((frameIndex.toFloat() / totalFrames) * 100).toInt().coerceIn(0, 100)
                            if (progress != lastProgress) {
                                lastProgress = progress
                                withContext(Dispatchers.Main) {
                                    progressBar.progress = progress
                                    progressText.text = " $progress%"
                                }
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

    // Hide progress UI
    hideProgressUI(activity)

    Log.d(TAG, "FAST STREAMING complete. Processed $frameIndex frames, ${poseFrames.size} poses detected")
    
    // Feature extraction (uses poseFrames which is small)
    extractGaitFeatures(context, width, height, frameIndex, activity)
    
    // Free heavy memory now that processing is done
    val frameCount = poseFrames.size
    poseFrames.clear()
    
    // Release reusable YUV conversion buffers
    bitmapCache?.recycle()
    bitmapCache = null
    yBytesCache = null
    uBytesCache = null
    vBytesCache = null
    pixelsCache = null
    Log.d(TAG, "Cleared poseFrames ($frameCount poses freed)")
    
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
    setDataSourceSafe(context, galleryUri!!, retriever)

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
    val encoderState = createEncoderState(outputPath, width, height, detectedFps)
    var frameIndex = 0
    var currTimeUs = 0L

    while (currTimeUs <= videoLengthUs) {
        val frame = retriever.getFrameAtTime(currTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        
        if (frame != null) {
            val modifiedBitmap = processFrame(frame, frameIndex)
            encoderState.encodeFrame(modifiedBitmap, frameIndex)
            frameIndex++
        }
        
        val progress = ((currTimeUs.toDouble() / videoLengthUs) * 100).toInt().coerceIn(0, 100)
        withContext(Dispatchers.Main) {
            activity.findViewById<ProgressBar>(R.id.splittingBar).progress = progress
            activity.findViewById<TextView>(R.id.splittingProgressValue).text = " $progress%"
        }
        
        currTimeUs += frameIntervalUs
    }

    encoderState.finishEncoding()
    encoderState.release()
    retriever.release()
    releaseMediaPipeBackend()

    hideProgressUI(activity)

    extractGaitFeatures(context, width, height, frameIndex, activity)

    val outputFile = File(outputPath)
    return FileProvider.getUriForFile(context, "${context.packageName}.provider", outputFile)
}

/** Extract gait features after all frames are processed and pose data is collected. */
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
        
        // Stabilize L/R identity, then normalize walking direction
        poseSequence = stabilizeLandmarkIdentity(poseSequence)
        poseSequence = featureExtractor.normalizeDirection(poseSequence)
        Log.d("ImageProcessing", "Walking direction: ${poseSequence.walkingDirection}, flipped: ${poseSequence.wasFlipped}")
        
        // Extract features
        val (features, diagnostics) = featureExtractor.extract(poseSequence)
        
        extractedFeatures = features
        extractionDiagnostics = diagnostics
        
        if (features != null) {
            Log.d("ImageProcessing", "Feature extraction successful!")
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

