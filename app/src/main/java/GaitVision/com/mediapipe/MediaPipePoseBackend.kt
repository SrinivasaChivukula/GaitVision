package GaitVision.com.mediapipe

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
// CLAHE will be implemented later if needed (can use pure Java implementation)

/**
 * MediaPipe Tasks Pose Backend for Android
 * 
 * Mirrors PC implementation (pose_backend_tasks.py) for feature parity.
 * Uses MediaPipe Tasks API with pose_landmarker_heavy.task model.
 * 
 * Key features:
 * - OPTIMAL_CONFIG parameters from PC pipeline
 * - CLAHE contrast enhancement support
 * - Normalized coordinates (0-1) matching PC
 * - Walking direction normalization support
 */
class MediaPipePoseBackend(
    context: Context,
    minDetectionConfidence: Float = 0.40f,  // OPTIMAL_CONFIG value
    minTrackingConfidence: Float = 0.61f,   // OPTIMAL_CONFIG value
    minPresenceConfidence: Float = 0.5f,
    useGpu: Boolean = true  // GPU delegate for ~2-3x speedup
) {
    
    private var landmarker: PoseLandmarker? = null
    private val context: Context = context.applicationContext
    private var usingGpu: Boolean = false
    
    init {
        // Try GPU first, fall back to CPU if GPU fails
        var initialized = false
        
        if (useGpu) {
            try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath("pose_landmarker_heavy.task")
                    .setDelegate(Delegate.GPU)
                    .build()
                
                val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.VIDEO)
                    .setMinPoseDetectionConfidence(minDetectionConfidence)
                    .setMinPosePresenceConfidence(minPresenceConfidence)
                    .setMinTrackingConfidence(minTrackingConfidence)
                    .setOutputSegmentationMasks(false)
                    .build()
                
                landmarker = PoseLandmarker.createFromOptions(context, options)
                usingGpu = true
                initialized = true
                
                Log.d(TAG, "Initialized MediaPipe with GPU delegate: " +
                        "det_conf=$minDetectionConfidence, track_conf=$minTrackingConfidence")
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate failed, falling back to CPU: ${e.message}")
            }
        }
        
        // Fallback to CPU if GPU failed or wasn't requested
        if (!initialized) {
            try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath("pose_landmarker_heavy.task")
                    .setDelegate(Delegate.CPU)
                    .build()
                
                val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.VIDEO)
                    .setMinPoseDetectionConfidence(minDetectionConfidence)
                    .setMinPosePresenceConfidence(minPresenceConfidence)
                    .setMinTrackingConfidence(minTrackingConfidence)
                    .setOutputSegmentationMasks(false)
                    .build()
                
                landmarker = PoseLandmarker.createFromOptions(context, options)
                usingGpu = false
                
                Log.d(TAG, "Initialized MediaPipe with CPU delegate: " +
                        "det_conf=$minDetectionConfidence, track_conf=$minTrackingConfidence")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize MediaPipe PoseLandmarker", e)
                throw e
            }
        }
    }
    
    /**
     * Check if GPU delegate is being used.
     */
    fun isUsingGpu(): Boolean = usingGpu
    
    /**
     * Process a single frame and return pose landmarks.
     * 
     * @param bitmap Input frame bitmap
     * @param timestampMs Frame timestamp in milliseconds
     * @return PoseLandmarkerResult or null if detection failed
     */
    fun processFrame(bitmap: Bitmap, timestampMs: Long): PoseLandmarkerResult? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            landmarker?.detectForVideo(mpImage, timestampMs)
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame at timestamp $timestampMs", e)
            null
        }
    }
    
    /**
     * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to bitmap.
     * Mirrors PC implementation _apply_clahe().
     * 
     * PC uses: cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
     * This is a pure Kotlin implementation that works on the luminance channel.
     * 
     * @param bitmap Input bitmap (RGB)
     * @param clipLimit Contrast limit (default 3.0 to match PC)
     * @param tileSize Tile grid size (default 8 to match PC)
     * @return Enhanced bitmap with improved contrast
     */
    fun applyCLAHE(
        bitmap: Bitmap, 
        clipLimit: Float = 3.0f, 
        tileSize: Int = 8
    ): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        // Get pixels
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        // Extract luminance channel (using weighted RGB to Y conversion)
        val luminance = IntArray(pixels.size)
        val chromaR = IntArray(pixels.size)
        val chromaG = IntArray(pixels.size)
        val chromaB = IntArray(pixels.size)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            
            // Convert to YCbCr-style: Y = 0.299R + 0.587G + 0.114B
            luminance[i] = ((0.299 * r + 0.587 * g + 0.114 * b).toInt()).coerceIn(0, 255)
            chromaR[i] = r
            chromaG[i] = g
            chromaB[i] = b
        }
        
        // Apply CLAHE to luminance
        val numTilesX = maxOf(1, width / tileSize)
        val numTilesY = maxOf(1, height / tileSize)
        val tileWidth = width / numTilesX
        val tileHeight = height / numTilesY
        
        val enhancedLuminance = IntArray(pixels.size)
        
        for (ty in 0 until numTilesY) {
            for (tx in 0 until numTilesX) {
                val x0 = tx * tileWidth
                val y0 = ty * tileHeight
                val x1 = if (tx == numTilesX - 1) width else x0 + tileWidth
                val y1 = if (ty == numTilesY - 1) height else y0 + tileHeight
                
                // Build histogram for this tile
                val histogram = IntArray(256)
                var tilePixelCount = 0
                
                for (y in y0 until y1) {
                    for (x in x0 until x1) {
                        val idx = y * width + x
                        histogram[luminance[idx]]++
                        tilePixelCount++
                    }
                }
                
                // Clip histogram (CLAHE part)
                val clipValue = (clipLimit * tilePixelCount / 256).toInt()
                var excess = 0
                for (i in 0 until 256) {
                    if (histogram[i] > clipValue) {
                        excess += histogram[i] - clipValue
                        histogram[i] = clipValue
                    }
                }
                
                // Redistribute excess uniformly
                val redistrib = excess / 256
                val remainder = excess % 256
                for (i in 0 until 256) {
                    histogram[i] += redistrib
                    if (i < remainder) histogram[i]++
                }
                
                // Build CDF (cumulative distribution function)
                val cdf = IntArray(256)
                cdf[0] = histogram[0]
                for (i in 1 until 256) {
                    cdf[i] = cdf[i - 1] + histogram[i]
                }
                
                // Normalize CDF to 0-255
                val cdfMin = cdf.first { it > 0 }
                val cdfMax = cdf[255]
                val denominator = (cdfMax - cdfMin).coerceAtLeast(1)
                
                // Apply equalization to tile
                for (y in y0 until y1) {
                    for (x in x0 until x1) {
                        val idx = y * width + x
                        val oldLum = luminance[idx]
                        enhancedLuminance[idx] = ((cdf[oldLum] - cdfMin) * 255 / denominator).coerceIn(0, 255)
                    }
                }
            }
        }
        
        // Reconstruct RGB from enhanced luminance
        // Simple approach: scale each channel by the luminance ratio
        val resultPixels = IntArray(pixels.size)
        for (i in pixels.indices) {
            val oldLum = luminance[i].coerceAtLeast(1)
            val newLum = enhancedLuminance[i]
            val scale = newLum.toFloat() / oldLum
            
            val newR = (chromaR[i] * scale).toInt().coerceIn(0, 255)
            val newG = (chromaG[i] * scale).toInt().coerceIn(0, 255)
            val newB = (chromaB[i] * scale).toInt().coerceIn(0, 255)
            
            resultPixels[i] = (0xFF shl 24) or (newR shl 16) or (newG shl 8) or newB
        }
        
        // Create result bitmap
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        result.setPixels(resultPixels, 0, width, 0, 0, width, height)
        
        return result
    }
    
    /**
     * Release MediaPipe resources.
     */
    fun release() {
        try {
            landmarker?.close()
            landmarker = null
            Log.d(TAG, "MediaPipe PoseLandmarker released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing MediaPipe PoseLandmarker", e)
        }
    }
    
    companion object {
        private const val TAG = "MediaPipePoseBackend"
        
        // MediaPipe keypoint indices (same as PC - 33 landmarks)
        const val NOSE = 0
        const val LEFT_EYE_INNER = 1
        const val LEFT_EYE = 2
        const val LEFT_EYE_OUTER = 3
        const val RIGHT_EYE_INNER = 4
        const val RIGHT_EYE = 5
        const val RIGHT_EYE_OUTER = 6
        const val LEFT_EAR = 7
        const val RIGHT_EAR = 8
        const val MOUTH_LEFT = 9
        const val MOUTH_RIGHT = 10
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_ELBOW = 13
        const val RIGHT_ELBOW = 14
        const val LEFT_WRIST = 15
        const val RIGHT_WRIST = 16
        const val LEFT_PINKY = 17
        const val RIGHT_PINKY = 18
        const val LEFT_INDEX = 19
        const val RIGHT_INDEX = 20
        const val LEFT_THUMB = 21
        const val RIGHT_THUMB = 22
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24
        const val LEFT_KNEE = 25
        const val RIGHT_KNEE = 26
        const val LEFT_ANKLE = 27
        const val RIGHT_ANKLE = 28
        const val LEFT_HEEL = 29
        const val RIGHT_HEEL = 30
        const val LEFT_FOOT_INDEX = 31
        const val RIGHT_FOOT_INDEX = 32
    }
}
