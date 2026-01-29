package GaitVision.com.mediapipe

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import GaitVision.com.gait.GaitConfig
import kotlin.math.max
import kotlin.math.min

/**
 * ROI (Region of Interest) Tracker for improved pose detection.
 * 
 * Kotlin port of PC ROITracker (pose_backend_tasks.py).
 * Uses state machine: Detect → Track → Reacquire pattern.
 * 
 * Industry-standard pattern that:
 * - Crops to person ROI for higher effective resolution
 * - Handles mid-video degradation without reprocessing
 * - Maps all keypoints back to full-frame coordinates
 */
class ROITracker(
    private val margin: Float = GaitConfig.ROI_MARGIN,
    private val expandedMargin: Float = GaitConfig.ROI_EXPANDED_MARGIN,
    private val centerEmaAlpha: Float = GaitConfig.ROI_CENTER_EMA_ALPHA,
    private val sizeExpandAlpha: Float = GaitConfig.ROI_SIZE_EXPAND_ALPHA,
    private val sizeShrinkAlpha: Float = GaitConfig.ROI_SIZE_SHRINK_ALPHA,
    private val targetRoiSize: Int = GaitConfig.ROI_TARGET_SIZE,
    private val acquireStableFrames: Int = GaitConfig.ROI_ACQUIRE_STABLE_FRAMES,
    private val qualityWindowSize: Int = GaitConfig.ROI_QUALITY_WINDOW_SIZE,
    private val qualityThreshold: Float = GaitConfig.ROI_QUALITY_THRESHOLD,
    private val degradedRatioThreshold: Float = GaitConfig.ROI_DEGRADED_RATIO_THRESHOLD,
    private val reacquireFrames: Int = GaitConfig.ROI_REACQUIRE_FRAMES,
    private val failRatioThresholdTrack: Float = GaitConfig.ROI_FAIL_RATIO_TRACK,
    private val failRatioThresholdExpand: Float = GaitConfig.ROI_FAIL_RATIO_EXPAND,
    private val consecutiveFailReacquire: Int = GaitConfig.ROI_CONSECUTIVE_FAIL_REACQUIRE,
    private val minDwellFrames: Int = GaitConfig.ROI_MIN_DWELL_FRAMES
) {
    companion object {
        private const val TAG = "ROITracker"
    }
    
    /**
     * State machine states for ROI tracking.
     */
    enum class ROIState {
        ACQUIRE,    // Full-frame until stable pose
        TRACK,      // ROI tracking mode
        EXPAND,     // Expanded ROI (quality degrading)
        REACQUIRE   // Burst full-frame to recover
    }
    
    // State machine
    private var state = ROIState.ACQUIRE
    private var stateFrameCount = 0
    
    // ROI bounds (normalized 0-1)
    private var roiCenterX: Float? = null
    private var roiCenterY: Float? = null
    private var roiSize: Float? = null
    
    // Quality monitoring
    private val qualityHistory = mutableListOf<Float>()
    private val detectHistory = mutableListOf<Boolean>()
    private var consecutiveFailures = 0
    
    // Stats for logging
    var framesInAcquire = 0
        private set
    var framesInTrack = 0
        private set
    var framesInExpand = 0
        private set
    var framesInReacquire = 0
        private set
    var reacquireCount = 0
        private set
    var transitionCount = 0
        private set
    var maxConsecutiveFail = 0
        private set
    
    /**
     * Reset tracker state.
     */
    fun reset() {
        state = ROIState.ACQUIRE
        stateFrameCount = 0
        roiCenterX = null
        roiCenterY = null
        roiSize = null
        qualityHistory.clear()
        detectHistory.clear()
        consecutiveFailures = 0
        framesInAcquire = 0
        framesInTrack = 0
        framesInExpand = 0
        framesInReacquire = 0
        reacquireCount = 0
        transitionCount = 0
        maxConsecutiveFail = 0
    }
    
    /**
     * Get far-leg confidence (min of both ankles).
     */
    private fun computeFarLegConf(confidences: FloatArray): Float {
        return min(
            confidences[MediaPipePoseBackend.LEFT_ANKLE],
            confidences[MediaPipePoseBackend.RIGHT_ANKLE]
        )
    }
    
    /**
     * Update rolling quality window.
     */
    private fun updateQualityHistory(farLegConf: Float) {
        qualityHistory.add(farLegConf)
        while (qualityHistory.size > qualityWindowSize) {
            qualityHistory.removeAt(0)
        }
    }
    
    /**
     * Update rolling detection-failure window.
     */
    private fun updateDetectHistory(success: Boolean) {
        detectHistory.add(success)
        while (detectHistory.size > qualityWindowSize) {
            detectHistory.removeAt(0)
        }
    }
    
    /**
     * Fraction of recent frames that were detection failures.
     */
    private fun failureRate(): Float {
        if (detectHistory.isEmpty()) return 0f
        return detectHistory.count { !it }.toFloat() / detectHistory.size
    }
    
    /**
     * Check if quality is degraded based on rolling window.
     */
    private fun isQualityDegraded(): Boolean {
        if (qualityHistory.size < qualityWindowSize / 2) return false
        
        val degradedCount = qualityHistory.count { it < qualityThreshold }
        val degradedRatio = degradedCount.toFloat() / qualityHistory.size
        return degradedRatio >= degradedRatioThreshold
    }
    
    /**
     * Check if quality has recovered.
     */
    private fun isQualityRecovered(): Boolean {
        if (qualityHistory.size < 5) return false
        
        val recent = qualityHistory.takeLast(5)
        val goodCount = recent.count { it >= qualityThreshold }
        return goodCount >= 4
    }
    
    /**
     * Update ROI bounds from keypoints.
     * 
     * @param keypoints Normalized keypoints (33x2)
     * @param confidences Visibility scores (33)
     * @param useExpandedMargin Whether to use expanded margin
     * @return True if successful
     */
    private fun updateRoiBounds(
        keypoints: Array<FloatArray>,
        confidences: FloatArray,
        useExpandedMargin: Boolean = false
    ): Boolean {
        val coreIndices = intArrayOf(
            MediaPipePoseBackend.LEFT_SHOULDER, MediaPipePoseBackend.RIGHT_SHOULDER,
            MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE
        )
        
        val validKps = mutableListOf<FloatArray>()
        for (idx in coreIndices) {
            if (confidences[idx] > 0.3f) {
                validKps.add(keypoints[idx])
            }
        }
        
        if (validKps.size < 4) return false
        
        val xMin = validKps.minOf { it[0] }
        val xMax = validKps.maxOf { it[0] }
        val yMin = validKps.minOf { it[1] }
        val yMax = validKps.maxOf { it[1] }
        
        val newCenterX = (xMin + xMax) / 2f
        val newCenterY = (yMin + yMax) / 2f
        
        val currentMargin = if (useExpandedMargin) expandedMargin else margin
        var newSize = max(xMax - xMin, yMax - yMin) * currentMargin
        newSize = max(newSize, 0.3f)  // Minimum size
        
        if (roiCenterX == null) {
            roiCenterX = newCenterX
            roiCenterY = newCenterY
            roiSize = newSize
        } else {
            // EMA smoothing for center
            roiCenterX = centerEmaAlpha * newCenterX + (1 - centerEmaAlpha) * roiCenterX!!
            roiCenterY = centerEmaAlpha * newCenterY + (1 - centerEmaAlpha) * roiCenterY!!
            
            // Expand fast, shrink slow for size
            val alpha = if (newSize > roiSize!!) sizeExpandAlpha else sizeShrinkAlpha
            roiSize = alpha * newSize + (1 - alpha) * roiSize!!
        }
        
        return true
    }
    
    /**
     * Transition to a new state.
     */
    private fun transition(newState: ROIState) {
        state = newState
        stateFrameCount = 0
        transitionCount++
    }
    
    /**
     * Update state machine based on detection results.
     * 
     * @param keypoints Normalized keypoints (or null if detection failed)
     * @param confidences Visibility scores (or null if detection failed)
     * @param detectionSuccess Whether pose was detected
     * @return Pair(useRoiNextFrame, useExpandedMargin)
     */
    fun update(
        keypoints: Array<FloatArray>?,
        confidences: FloatArray?,
        detectionSuccess: Boolean
    ): Pair<Boolean, Boolean> {
        stateFrameCount++
        
        // Update quality history
        if (detectionSuccess && confidences != null) {
            val farLegConf = computeFarLegConf(confidences)
            updateQualityHistory(farLegConf)
        } else {
            updateQualityHistory(0f)  // Treat failure as worst quality
        }
        
        // Track consecutive failures
        if (detectionSuccess) {
            consecutiveFailures = 0
        } else {
            consecutiveFailures++
            maxConsecutiveFail = max(maxConsecutiveFail, consecutiveFailures)
        }
        
        // State machine transitions
        return when (state) {
            ROIState.ACQUIRE -> {
                framesInAcquire++
                
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                    
                    if (stateFrameCount >= acquireStableFrames && roiCenterX != null) {
                        transition(ROIState.TRACK)
                        Log.d(TAG, "ROI: ACQUIRE -> TRACK")
                    }
                }
                
                Pair(false, false)  // Full-frame during acquire
            }
            
            ROIState.TRACK -> {
                framesInTrack++
                updateDetectHistory(detectionSuccess)
                
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                }
                
                // Hard trigger: consecutive failures → REACQUIRE
                if (consecutiveFailures >= consecutiveFailReacquire) {
                    transition(ROIState.REACQUIRE)
                    reacquireCount++
                    detectHistory.clear()
                    Log.d(TAG, "ROI: TRACK -> REACQUIRE (consecutive failures)")
                    return Pair(false, false)
                }
                
                // Check failure rate after min dwell
                if (stateFrameCount >= minDwellFrames) {
                    val failRate = failureRate()
                    if (failRate >= failRatioThresholdTrack) {
                        transition(ROIState.EXPAND)
                        Log.d(TAG, "ROI: TRACK -> EXPAND (failure rate ${(failRate * 100).toInt()}%)")
                        return Pair(true, true)
                    }
                    if (isQualityDegraded()) {
                        transition(ROIState.EXPAND)
                        Log.d(TAG, "ROI: TRACK -> EXPAND (quality degraded)")
                        return Pair(true, true)
                    }
                }
                
                Pair(true, false)  // ROI with normal margin
            }
            
            ROIState.EXPAND -> {
                framesInExpand++
                updateDetectHistory(detectionSuccess)
                
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences, useExpandedMargin = true)
                }
                
                // Check if quality recovered
                if (isQualityRecovered()) {
                    transition(ROIState.TRACK)
                    Log.d(TAG, "ROI: EXPAND -> TRACK (quality recovered)")
                    return Pair(true, false)
                }
                
                // Timeout or continued failure → REACQUIRE
                if (stateFrameCount >= reacquireFrames) {
                    val failRate = failureRate()
                    if (failRate >= failRatioThresholdExpand) {
                        transition(ROIState.REACQUIRE)
                        reacquireCount++
                        detectHistory.clear()
                        Log.d(TAG, "ROI: EXPAND -> REACQUIRE (failure rate ${(failRate * 100).toInt()}%)")
                        return Pair(false, false)
                    }
                    // Timeout without recovery
                    transition(ROIState.REACQUIRE)
                    reacquireCount++
                    detectHistory.clear()
                    Log.d(TAG, "ROI: EXPAND -> REACQUIRE (timeout)")
                    return Pair(false, false)
                }
                
                Pair(true, true)  // ROI with expanded margin
            }
            
            ROIState.REACQUIRE -> {
                framesInReacquire++
                
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                    updateQualityHistory(computeFarLegConf(confidences))
                }
                
                // After burst, return to tracking or expand
                if (stateFrameCount >= reacquireFrames) {
                    if (isQualityRecovered()) {
                        transition(ROIState.TRACK)
                        Log.d(TAG, "ROI: REACQUIRE -> TRACK (recovered)")
                    } else {
                        transition(ROIState.EXPAND)
                        Log.d(TAG, "ROI: REACQUIRE -> EXPAND (still degraded)")
                    }
                    return Pair(true, state == ROIState.EXPAND)
                }
                
                Pair(false, false)  // Full-frame during reacquire
            }
        }
    }
    
    /**
     * Get ROI bounds in pixel coordinates.
     * 
     * @param frameWidth Frame width in pixels
     * @param frameHeight Frame height in pixels
     * @param useExpanded Whether to use expanded margin
     * @return Rect with ROI bounds, or full frame if not ready
     */
    fun getRoiBounds(frameWidth: Int, frameHeight: Int, useExpanded: Boolean = false): Rect {
        if (roiCenterX == null || roiSize == null) {
            return Rect(0, 0, frameWidth, frameHeight)
        }
        
        var size = roiSize!!
        if (useExpanded) {
            size *= (expandedMargin / margin)
        }
        
        val halfSize = size / 2f
        
        val x0 = ((roiCenterX!! - halfSize) * frameWidth).toInt().coerceAtLeast(0)
        val y0 = ((roiCenterY!! - halfSize) * frameHeight).toInt().coerceAtLeast(0)
        val x1 = ((roiCenterX!! + halfSize) * frameWidth).toInt().coerceAtMost(frameWidth)
        val y1 = ((roiCenterY!! + halfSize) * frameHeight).toInt().coerceAtMost(frameHeight)
        
        // Ensure minimum size
        if (x1 - x0 < 100 || y1 - y0 < 100) {
            return Rect(0, 0, frameWidth, frameHeight)
        }
        
        return Rect(x0, y0, x1, y1)
    }
    
    /**
     * Map keypoints from ROI-normalized coords to full-frame normalized coords.
     * 
     * @param keypoints ROI-normalized keypoints (33x2)
     * @param roiBounds ROI bounds in pixels
     * @param frameWidth Full frame width
     * @param frameHeight Full frame height
     * @return Full-frame normalized keypoints
     */
    fun mapKeypointsToFullFrame(
        keypoints: Array<FloatArray>,
        roiBounds: Rect,
        frameWidth: Int,
        frameHeight: Int
    ): Array<FloatArray> {
        val roiWidth = roiBounds.width().toFloat()
        val roiHeight = roiBounds.height().toFloat()
        
        return Array(keypoints.size) { i ->
            floatArrayOf(
                (roiBounds.left + keypoints[i][0] * roiWidth) / frameWidth,
                (roiBounds.top + keypoints[i][1] * roiHeight) / frameHeight
            )
        }
    }
    
    /**
     * Crop bitmap to ROI and resize to target size.
     * 
     * @param bitmap Full frame bitmap
     * @param roiBounds ROI bounds in pixels
     * @return Cropped and resized bitmap
     */
    fun cropToRoi(bitmap: Bitmap, roiBounds: Rect): Bitmap {
        // Ensure bounds are valid
        val x = roiBounds.left.coerceIn(0, bitmap.width - 1)
        val y = roiBounds.top.coerceIn(0, bitmap.height - 1)
        val width = roiBounds.width().coerceIn(1, bitmap.width - x)
        val height = roiBounds.height().coerceIn(1, bitmap.height - y)
        
        // Crop
        val cropped = Bitmap.createBitmap(bitmap, x, y, width, height)
        
        // Resize to target size if needed
        return if (targetRoiSize > 0 && (width != targetRoiSize || height != targetRoiSize)) {
            Bitmap.createScaledBitmap(cropped, targetRoiSize, targetRoiSize, true)
        } else {
            cropped
        }
    }
    
    /**
     * Get tracking statistics for logging.
     */
    fun getStats(): Map<String, Any> {
        val total = framesInAcquire + framesInTrack + framesInExpand + framesInReacquire
        if (total == 0) return emptyMap()
        
        return mapOf(
            "acquire_pct" to (framesInAcquire * 100f / total),
            "track_pct" to (framesInTrack * 100f / total),
            "expand_pct" to (framesInExpand * 100f / total),
            "reacquire_pct" to (framesInReacquire * 100f / total),
            "reacquire_count" to reacquireCount,
            "transition_count" to transitionCount,
            "max_consecutive_fail" to maxConsecutiveFail
        )
    }
    
    /**
     * Whether ROI tracking should be used for the next frame.
     */
    fun shouldUseRoi(): Boolean {
        return state == ROIState.TRACK || state == ROIState.EXPAND
    }
    
    /**
     * Current state name for debugging.
     */
    fun getCurrentState(): String = state.name
}
