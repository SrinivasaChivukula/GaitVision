package GaitVision.com.mediapipe

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import GaitVision.com.gait.GaitConfig
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * ROI (Region of Interest) Tracker for improved pose detection.
 *
 * Uses a conservative state machine:
 * ACQUIRE full-frame -> TRACK ROI -> EXPAND ROI -> REACQUIRE full-frame.
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

    enum class ROIState {
        ACQUIRE,
        TRACK,
        EXPAND,
        REACQUIRE
    }

    data class RoiCrop(
        val bitmap: Bitmap,
        val roiBounds: Rect,
        val squareSize: Int,
        val padLeft: Int,
        val padTop: Int,
        val contentWidth: Int,
        val contentHeight: Int
    )

    private var state = ROIState.ACQUIRE
    private var stateFrameCount = 0
    private var acquireStableGoodFrames = 0

    private var roiCenterX: Float? = null
    private var roiCenterY: Float? = null
    private var roiWidth: Float? = null
    private var roiHeight: Float? = null

    private val qualityHistory = mutableListOf<Float>()
    private val detectHistory = mutableListOf<Boolean>()
    private var consecutiveFailures = 0
    private var lowerBodyBadStreak = 0

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
    var maxLowerBodyBadStreak = 0
        private set

    fun reset() {
        state = ROIState.ACQUIRE
        stateFrameCount = 0
        acquireStableGoodFrames = 0
        roiCenterX = null
        roiCenterY = null
        roiWidth = null
        roiHeight = null
        qualityHistory.clear()
        detectHistory.clear()
        consecutiveFailures = 0
        lowerBodyBadStreak = 0
        framesInAcquire = 0
        framesInTrack = 0
        framesInExpand = 0
        framesInReacquire = 0
        reacquireCount = 0
        transitionCount = 0
        maxConsecutiveFail = 0
        maxLowerBodyBadStreak = 0
    }

    fun lowerBodyQuality(confidences: FloatArray?): Float {
        if (confidences == null) return 0f
        val indices = intArrayOf(
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.RIGHT_ANKLE,
            MediaPipePoseBackend.LEFT_HEEL,
            MediaPipePoseBackend.RIGHT_HEEL,
            MediaPipePoseBackend.LEFT_FOOT_INDEX,
            MediaPipePoseBackend.RIGHT_FOOT_INDEX
        )

        var sum = 0f
        var valid = 0
        for (idx in indices) {
            val confidence = confidences.getOrNull(idx)?.coerceIn(0f, 1f) ?: 0f
            sum += confidence
            if (confidence >= GaitConfig.MIN_CONFIDENCE) valid++
        }

        val mean = sum / indices.size
        val coverage = valid.toFloat() / indices.size
        return mean * coverage
    }

    private fun updateQualityHistory(quality: Float) {
        qualityHistory.add(quality)
        while (qualityHistory.size > qualityWindowSize) {
            qualityHistory.removeAt(0)
        }
    }

    private fun updateDetectHistory(success: Boolean) {
        detectHistory.add(success)
        while (detectHistory.size > qualityWindowSize) {
            detectHistory.removeAt(0)
        }
    }

    private fun failureRate(): Float {
        if (detectHistory.isEmpty()) return 0f
        return detectHistory.count { !it }.toFloat() / detectHistory.size
    }

    private fun isQualityDegraded(): Boolean {
        if (qualityHistory.size < qualityWindowSize / 2) return false
        val degradedCount = qualityHistory.count { it < qualityThreshold }
        val degradedRatio = degradedCount.toFloat() / qualityHistory.size
        return degradedRatio >= degradedRatioThreshold
    }

    private fun isQualityRecovered(): Boolean {
        if (qualityHistory.size < 5) return false
        val recent = qualityHistory.takeLast(5)
        return recent.count { it >= qualityThreshold } >= 4
    }

    private fun updateRoiBounds(
        keypoints: Array<FloatArray>,
        confidences: FloatArray
    ): Boolean {
        val coreIndices = intArrayOf(
            MediaPipePoseBackend.LEFT_SHOULDER,
            MediaPipePoseBackend.RIGHT_SHOULDER,
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.RIGHT_ANKLE,
            MediaPipePoseBackend.LEFT_HEEL,
            MediaPipePoseBackend.RIGHT_HEEL,
            MediaPipePoseBackend.LEFT_FOOT_INDEX,
            MediaPipePoseBackend.RIGHT_FOOT_INDEX
        )

        val validKps = mutableListOf<FloatArray>()
        for (idx in coreIndices) {
            val confidence = confidences.getOrNull(idx) ?: 0f
            val point = keypoints.getOrNull(idx)
            if (confidence > GaitConfig.ROI_BOUNDS_MIN_CONFIDENCE && point != null) {
                validKps.add(point)
            }
        }
        if (validKps.size < 4) return false

        val xMin = validKps.minOf { it[0] }
        val xMax = validKps.maxOf { it[0] }
        val yMin = validKps.minOf { it[1] }
        val yMax = validKps.maxOf { it[1] }

        val bodyWidth = max(xMax - xMin, 0.05f)
        val bodyHeight = max(yMax - yMin, 0.05f)
        val xPad = bodyWidth * (margin - 1f) / 2f
        val yPad = bodyHeight * (margin - 1f) / 2f
        val yTopPad = yPad + bodyHeight * GaitConfig.ROI_UPPER_BODY_EXTRA_PAD
        val yBottomPad = yPad + bodyHeight * GaitConfig.ROI_LOWER_BODY_EXTRA_PAD

        val paddedXMin = (xMin - xPad).coerceIn(0f, 1f)
        val paddedXMax = (xMax + xPad).coerceIn(0f, 1f)
        val paddedYMin = (yMin - yTopPad).coerceIn(0f, 1f)
        val paddedYMax = (yMax + yBottomPad).coerceIn(0f, 1f)

        val newCenterX = (paddedXMin + paddedXMax) / 2f
        val newCenterY = (paddedYMin + paddedYMax) / 2f
        val newWidth = max(paddedXMax - paddedXMin, 0.3f)
        val newHeight = max(paddedYMax - paddedYMin, 0.3f)

        if (roiCenterX == null) {
            roiCenterX = newCenterX
            roiCenterY = newCenterY
            roiWidth = newWidth
            roiHeight = newHeight
        } else {
            roiCenterX = centerEmaAlpha * newCenterX + (1 - centerEmaAlpha) * roiCenterX!!
            roiCenterY = centerEmaAlpha * newCenterY + (1 - centerEmaAlpha) * roiCenterY!!

            val widthAlpha = if (newWidth > roiWidth!!) sizeExpandAlpha else sizeShrinkAlpha
            val heightAlpha = if (newHeight > roiHeight!!) sizeExpandAlpha else sizeShrinkAlpha
            roiWidth = widthAlpha * newWidth + (1 - widthAlpha) * roiWidth!!
            roiHeight = heightAlpha * newHeight + (1 - heightAlpha) * roiHeight!!
        }

        return true
    }

    private fun transition(newState: ROIState) {
        state = newState
        stateFrameCount = 0
        transitionCount++
        if (newState == ROIState.ACQUIRE) {
            acquireStableGoodFrames = 0
        }
    }

    fun update(
        keypoints: Array<FloatArray>?,
        confidences: FloatArray?,
        detectionSuccess: Boolean
    ): Pair<Boolean, Boolean> {
        stateFrameCount++

        val quality = if (detectionSuccess) lowerBodyQuality(confidences) else 0f
        updateQualityHistory(quality)

        if (detectionSuccess) {
            consecutiveFailures = 0
        } else {
            consecutiveFailures++
            maxConsecutiveFail = max(maxConsecutiveFail, consecutiveFailures)
        }

        if (quality < qualityThreshold) {
            lowerBodyBadStreak++
            maxLowerBodyBadStreak = max(maxLowerBodyBadStreak, lowerBodyBadStreak)
        } else {
            lowerBodyBadStreak = 0
        }

        return when (state) {
            ROIState.ACQUIRE -> {
                framesInAcquire++
                if (detectionSuccess && keypoints != null && confidences != null) {
                    val updated = updateRoiBounds(keypoints, confidences)
                    if (updated && quality >= GaitConfig.ROI_ACQUIRE_MIN_QUALITY) {
                        acquireStableGoodFrames++
                    } else {
                        acquireStableGoodFrames = 0
                    }

                    if (acquireStableGoodFrames >= acquireStableFrames && roiCenterX != null) {
                        transition(ROIState.TRACK)
                        Log.d(TAG, "ROI: ACQUIRE -> TRACK after $acquireStableFrames stable lower-body frames")
                    }
                } else {
                    acquireStableGoodFrames = 0
                }
                Pair(false, false)
            }

            ROIState.TRACK -> {
                framesInTrack++
                updateDetectHistory(detectionSuccess)
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                }

                if (consecutiveFailures >= consecutiveFailReacquire ||
                    lowerBodyBadStreak >= GaitConfig.ROI_FAST_REACQUIRE_BAD_FRAMES) {
                    transition(ROIState.REACQUIRE)
                    reacquireCount++
                    detectHistory.clear()
                    Log.d(TAG, "ROI: TRACK -> REACQUIRE (failures=$consecutiveFailures, lowerBodyBad=$lowerBodyBadStreak)")
                    return Pair(false, false)
                }

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

                Pair(true, false)
            }

            ROIState.EXPAND -> {
                framesInExpand++
                updateDetectHistory(detectionSuccess)
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                }

                if (lowerBodyBadStreak >= GaitConfig.ROI_FAST_REACQUIRE_BAD_FRAMES) {
                    transition(ROIState.REACQUIRE)
                    reacquireCount++
                    detectHistory.clear()
                    Log.d(TAG, "ROI: EXPAND -> REACQUIRE (lower-body confidence stayed low)")
                    return Pair(false, false)
                }

                if (isQualityRecovered()) {
                    transition(ROIState.TRACK)
                    Log.d(TAG, "ROI: EXPAND -> TRACK (quality recovered)")
                    return Pair(true, false)
                }

                if (stateFrameCount >= reacquireFrames) {
                    val failRate = failureRate()
                    transition(ROIState.REACQUIRE)
                    reacquireCount++
                    detectHistory.clear()
                    val reason = if (failRate >= failRatioThresholdExpand) {
                        "failure rate ${(failRate * 100).toInt()}%"
                    } else {
                        "timeout"
                    }
                    Log.d(TAG, "ROI: EXPAND -> REACQUIRE ($reason)")
                    return Pair(false, false)
                }

                Pair(true, true)
            }

            ROIState.REACQUIRE -> {
                framesInReacquire++
                if (detectionSuccess && keypoints != null && confidences != null) {
                    updateRoiBounds(keypoints, confidences)
                }

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

                Pair(false, false)
            }
        }
    }

    fun getRoiBounds(frameWidth: Int, frameHeight: Int, useExpanded: Boolean = false): Rect {
        if (roiCenterX == null || roiWidth == null || roiHeight == null) {
            return Rect(0, 0, frameWidth, frameHeight)
        }

        var width = roiWidth!!
        var height = roiHeight!!
        if (useExpanded) {
            val expansion = expandedMargin / margin
            width *= expansion
            height *= expansion
        }

        val halfWidth = width / 2f
        val halfHeight = height / 2f
        val x0 = ((roiCenterX!! - halfWidth) * frameWidth).roundToInt().coerceAtLeast(0)
        val y0 = ((roiCenterY!! - halfHeight) * frameHeight).roundToInt().coerceAtLeast(0)
        val x1 = ((roiCenterX!! + halfWidth) * frameWidth).roundToInt().coerceAtMost(frameWidth)
        val y1 = ((roiCenterY!! + halfHeight) * frameHeight).roundToInt().coerceAtMost(frameHeight)

        if (x1 - x0 < 100 || y1 - y0 < 100) {
            return Rect(0, 0, frameWidth, frameHeight)
        }

        return Rect(x0, y0, x1, y1)
    }

    fun mapKeypointsToFullFrame(
        keypoints: Array<FloatArray>,
        roiCrop: RoiCrop,
        frameWidth: Int,
        frameHeight: Int
    ): Array<FloatArray> {
        val roiBounds = roiCrop.roiBounds
        val roiWidth = roiCrop.contentWidth.toFloat()
        val roiHeight = roiCrop.contentHeight.toFloat()
        val squareSize = roiCrop.squareSize.toFloat()

        return Array(keypoints.size) { i ->
            val squareX = keypoints[i][0] * squareSize
            val squareY = keypoints[i][1] * squareSize
            val cropX = (squareX - roiCrop.padLeft) / roiWidth
            val cropY = (squareY - roiCrop.padTop) / roiHeight
            floatArrayOf(
                ((roiBounds.left + cropX * roiWidth) / frameWidth).coerceIn(0f, 1f),
                ((roiBounds.top + cropY * roiHeight) / frameHeight).coerceIn(0f, 1f)
            )
        }
    }

    fun mapPoseFrameToFullFrame(
        frame: PoseFrame,
        roiCrop: RoiCrop,
        frameWidth: Int,
        frameHeight: Int
    ): PoseFrame {
        val mappedKeypoints = mapKeypointsToFullFrame(
            frame.keypoints,
            roiCrop,
            frameWidth,
            frameHeight
        )
        val mappedConfidences = frame.confidences.copyOf()
        val squareSize = roiCrop.squareSize.toFloat()
        val contentLeft = roiCrop.padLeft.toFloat()
        val contentTop = roiCrop.padTop.toFloat()
        val contentRight = contentLeft + roiCrop.contentWidth
        val contentBottom = contentTop + roiCrop.contentHeight

        for (i in frame.keypoints.indices) {
            val squareX = frame.keypoints[i][0] * squareSize
            val squareY = frame.keypoints[i][1] * squareSize
            val insideContent = squareX in contentLeft..contentRight && squareY in contentTop..contentBottom
            if (!insideContent) {
                mappedConfidences[i] = 0f
            }
        }

        return frame.copy(
            keypoints = mappedKeypoints,
            confidences = mappedConfidences
        )
    }

    fun cropToRoi(bitmap: Bitmap, roiBounds: Rect): RoiCrop {
        val x = roiBounds.left.coerceIn(0, bitmap.width - 1)
        val y = roiBounds.top.coerceIn(0, bitmap.height - 1)
        val width = roiBounds.width().coerceIn(1, bitmap.width - x)
        val height = roiBounds.height().coerceIn(1, bitmap.height - y)
        val safeBounds = Rect(x, y, x + width, y + height)

        val cropped = Bitmap.createBitmap(bitmap, x, y, width, height)
        val squareSize = max(width, height)
        val padLeft = (squareSize - width) / 2
        val padTop = (squareSize - height) / 2
        val square = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)
        Canvas(square).apply {
            drawColor(Color.BLACK)
            drawBitmap(cropped, padLeft.toFloat(), padTop.toFloat(), null)
        }

        val output = if (targetRoiSize > 0 && squareSize != targetRoiSize) {
            Bitmap.createScaledBitmap(square, targetRoiSize, targetRoiSize, true)
        } else {
            square
        }
        // createBitmap(source, ...) returns the SAME instance when the crop
        // covers the whole source with matching config — recycling it would
        // destroy the caller's frame bitmap, which is still needed for
        // wireframe drawing and encoding.
        if (cropped !== bitmap && !cropped.isRecycled) cropped.recycle()
        if (output !== square && !square.isRecycled) square.recycle()

        return RoiCrop(
            bitmap = output,
            roiBounds = safeBounds,
            squareSize = squareSize,
            padLeft = padLeft,
            padTop = padTop,
            contentWidth = width,
            contentHeight = height
        )
    }

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
            "max_consecutive_fail" to maxConsecutiveFail,
            "max_lower_body_bad_streak" to maxLowerBodyBadStreak
        )
    }

    fun shouldUseRoi(): Boolean {
        return state == ROIState.TRACK || state == ROIState.EXPAND
    }

    fun shouldUseExpandedRoi(): Boolean {
        return state == ROIState.EXPAND
    }

    fun getCurrentState(): String = state.name
}
