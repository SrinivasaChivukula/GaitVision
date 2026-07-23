package GaitVision.com.mediapipe

import kotlin.math.PI
import kotlin.math.abs

/**
 * Adaptive low-pass filter for noisy landmark coordinates.
 *
 * Coordinates are normalized 0..1 in this app, so beta is tuned on that scale.
 */
class OneEuroFilter(
    private val minCutoff: Float = 8.0f,
    private val beta: Float = 40.0f,
    private val dCutoff: Float = 1.0f
) {
    private var initialized = false
    private var xPrev = 0f
    private var dxPrev = 0f

    fun filter(x: Float, dt: Float, resetVelocity: Boolean = false): Float {
        // Degenerate samples (NaN coordinate or non-increasing timestamp) must
        // not leak a raw spike into the smoothed output: hold the last
        // filtered value if we have one, otherwise pass through untouched.
        if (x.isNaN() || dt <= 0f) {
            return if (initialized) xPrev else x
        }
        if (!initialized) {
            initialized = true
            xPrev = x
            dxPrev = 0f
            return x
        }

        val dx = if (resetVelocity) 0f else (x - xPrev) / dt
        val prevDx = if (resetVelocity) 0f else dxPrev
        val dxAlpha = alpha(dCutoff, dt)
        val dxHat = dxAlpha * dx + (1f - dxAlpha) * prevDx

        val cutoff = minCutoff + beta * abs(dxHat)
        val xAlpha = alpha(cutoff, dt)
        val xHat = xAlpha * x + (1f - xAlpha) * xPrev

        xPrev = xHat
        dxPrev = dxHat
        return xHat
    }

    fun reset() {
        initialized = false
        xPrev = 0f
        dxPrev = 0f
    }

    private fun alpha(cutoff: Float, dt: Float): Float {
        val safeCutoff = cutoff.coerceAtLeast(1e-6f)
        val tau = 1f / (2f * PI.toFloat() * safeCutoff)
        return 1f / (1f + tau / dt)
    }
}

/**
 * Filters high-confidence pose landmarks while preserving low-confidence
 * samples as raw coordinates with their original confidence.
 */
class PoseOneEuroSmoother(
    private val numLandmarks: Int = 33,
    private val minCutoff: Float = 8.0f,
    private val beta: Float = 40.0f,
    private val maxMissStreak: Int = 3
) {
    private val xFilters = Array(numLandmarks) { OneEuroFilter(minCutoff, beta) }
    private val yFilters = Array(numLandmarks) { OneEuroFilter(minCutoff, beta) }
    private val missStreak = IntArray(numLandmarks)

    fun smooth(frame: PoseFrame, dt: Float, minConfidence: Float): PoseFrame {
        val out = Array(frame.keypoints.size) { i ->
            val point = frame.keypoints[i]
            if (i < numLandmarks && frame.confidences.getOrNull(i)?.let { it >= minConfidence } == true) {
                val recovering = missStreak[i] in 1 until maxMissStreak
                missStreak[i] = 0
                floatArrayOf(
                    xFilters[i].filter(point[0], dt, resetVelocity = recovering),
                    yFilters[i].filter(point[1], dt, resetVelocity = recovering)
                )
            } else {
                if (i < numLandmarks) {
                    missStreak[i]++
                    if (missStreak[i] >= maxMissStreak) {
                        xFilters[i].reset()
                        yFilters[i].reset()
                    }
                }
                point.copyOf()
            }
        }
        return frame.copy(keypoints = out)
    }

    fun reset() {
        for (i in 0 until numLandmarks) {
            xFilters[i].reset()
            yFilters[i].reset()
            missStreak[i] = 0
        }
    }

    /**
     * Reset filter state for a subset of landmarks only. Used when an
     * upstream identity correction (e.g. a confirmed L/R leg-label flip)
     * introduces a legitimate discontinuity in those landmarks' trajectories:
     * without a reset, the filter would blend the pre-flip position of the
     * opposite leg into the first post-flip frames.
     */
    fun resetLandmarks(indices: IntArray) {
        for (i in indices) {
            if (i in 0 until numLandmarks) {
                xFilters[i].reset()
                yFilters[i].reset()
                missStreak[i] = 0
            }
        }
    }
}
