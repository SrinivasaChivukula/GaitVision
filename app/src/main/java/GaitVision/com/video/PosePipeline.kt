package GaitVision.com.video

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import GaitVision.com.enableCLAHE
import GaitVision.com.forceCpuInference
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.MediaPipeResultConverter
import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence

private const val TAG = "GaitMedia"

const val PROCESSING_WIDTH = 1280
const val PROCESSING_HEIGHT = 720

private var mediaPipeBackend: MediaPipePoseBackend? = null

var detectedFps: Float = 30f
    internal set

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
        val onEmulator = isEmulator()
        val useGpu = !forceCpuInference && !onEmulator
        if (onEmulator) {
            Log.d(TAG, "Emulator detected, forcing CPU delegate")
        }
        mediaPipeBackend = MediaPipePoseBackend(
            context = context,
            minDetectionConfidence = 0.40f,
            minTrackingConfidence = 0.61f,
            minPresenceConfidence = 0.5f,
            useGpu = useGpu
        )
        val delegateType = if (useGpu) "GPU" else "CPU${if (onEmulator) " (emulator)" else " (parity mode)"}"
        Log.d(TAG, "MediaPipe backend initialized, delegate: $delegateType")
    }
}

fun releaseMediaPipeBackend() {
    mediaPipeBackend?.release()
    mediaPipeBackend = null
    Log.d(TAG, "MediaPipe backend released")
}

fun isMediaPipeUsingGpu(): Boolean = mediaPipeBackend?.isUsingGpu() ?: false

/** L/R mirror index for landmark swap. */
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

private fun dist2d(a: FloatArray, b: FloatArray): Float {
    val dx = a[0] - b[0]
    val dy = a[1] - b[1]
    return kotlin.math.sqrt(dx * dx + dy * dy)
}

/** Fix per-frame L/R swaps. Run before normalizeDirection(). */
fun stabilizeLandmarkIdentity(poseSeq: PoseSequence): PoseSequence {
    val MIN_CONF = GaitVision.com.gait.GaitConfig.MIN_CONFIDENCE
    val HYSTERESIS = 0.8f
    val trackedPairs = listOf(
        Pair(MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE),
        Pair(MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.RIGHT_KNEE),
        Pair(MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP)
    )
    val frames = poseSeq.frames
    if (frames.size < 2) return poseSeq
    val stabilizedFrames = frames.toMutableList()
    var totalSwaps = 0
    var currentRunLength = 0
    var maxRunLength = 0
    var lastEvalIdx = 0
    for (t in 1 until frames.size) {
        val prev = stabilizedFrames[lastEvalIdx]
        val curr = frames[t]
        val allConfident = trackedPairs.all { (l, r) ->
            curr.confidences[l] >= MIN_CONF && curr.confidences[r] >= MIN_CONF &&
            prev.confidences[l] >= MIN_CONF && prev.confidences[r] >= MIN_CONF
        }
        if (!allConfident) {
            stabilizedFrames[t] = curr
            continue
        }
        var costKeep = 0f
        var costSwap = 0f
        for ((l, r) in trackedPairs) {
            costKeep += dist2d(curr.keypoints[l], prev.keypoints[l]) + dist2d(curr.keypoints[r], prev.keypoints[r])
            costSwap += dist2d(curr.keypoints[l], prev.keypoints[r]) + dist2d(curr.keypoints[r], prev.keypoints[l])
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
    Log.d(TAG, "Identity stabilization: ${frames.size} frames, $totalSwaps swaps (${String.format("%.1f", swapRate * 100)}%)")
    return poseSeq.copy(frames = stabilizedFrames)
}

fun processFrameWithMediaPipe(
    bitmap: Bitmap,
    frameIdx: Int,
    fps: Float = detectedFps,
    applyClahe: Boolean = enableCLAHE
): PoseFrame? {
    val backend = mediaPipeBackend ?: return null
    val scaledBitmap = if (bitmap.width > PROCESSING_WIDTH || bitmap.height > PROCESSING_HEIGHT) {
        Bitmap.createScaledBitmap(bitmap, PROCESSING_WIDTH, PROCESSING_HEIGHT, true)
    } else {
        bitmap
    }
    val processedBitmap = if (applyClahe) backend.applyCLAHE(scaledBitmap) else scaledBitmap
    val timestampMs = (frameIdx * 1000L / fps).toLong()
    val result = backend.processFrame(processedBitmap, timestampMs)
    if (applyClahe && processedBitmap !== scaledBitmap) processedBitmap.recycle()
    if (scaledBitmap !== bitmap) scaledBitmap.recycle()
    return MediaPipeResultConverter.toPoseFrame(
        result = result,
        frameIdx = frameIdx,
        timestampS = timestampMs / 1000f
    )
}
