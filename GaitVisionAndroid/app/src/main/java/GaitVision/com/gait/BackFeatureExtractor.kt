package GaitVision.com.gait

import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*
import android.util.Log

/**
 * Back-view (posterior) gait feature extractor.
 * Mirrors PC pipeline (extract_back_features.py) for feature parity.
 */
class BackFeatureExtractor(
    private val minConfidence: Float = GaitConfig.MIN_CONFIDENCE,
    private val maxInterpGap: Int = GaitConfig.MAX_INTERP_GAP,
    private val emaAlpha: Float = GaitConfig.EMA_ALPHA,
    private val minStepTimeS: Float = GaitConfig.MIN_STEP_TIME_S
) {
    companion object {
        val BACK_FEATURE_COLUMNS = listOf(
            "cadence_spm", "hip_drop_mean", "hip_drop_std", "hip_sway_std",
            "stride_width_mean", "stride_width_cv", "step_width_asymmetry",
            "knee_separation_mean", "trunk_lean_mean", "trunk_lean_std"
        )
    }

    fun extract(poseSeq: PoseSequence): BackGaitFeatures? {
        Log.d("GaitDebug", "BackFeatureExtractor: frames=${poseSeq.frames.size}, numFramesTotal=${poseSeq.numFramesTotal}, detectionRate=${poseSeq.detectionRate}")
        if (poseSeq.frames.size < 20) {
            Log.d("GaitDebug", "BackFeatureExtractor: FAILED - too few frames")
            return null
        }
        if (poseSeq.detectionRate < 0.3f) {
            Log.d("GaitDebug", "BackFeatureExtractor: FAILED - low detection rate")
            return null
        }
        if (poseSeq.frames.size < 20) return null
        if (poseSeq.detectionRate < 0.3f) return null

        val n = poseSeq.numFramesTotal
        val hipLeftX = FloatArray(n) { Float.NaN }
        val hipRightX = FloatArray(n) { Float.NaN }
        val hipLeftY = FloatArray(n) { Float.NaN }
        val hipRightY = FloatArray(n) { Float.NaN }
        val kneeLeftX = FloatArray(n) { Float.NaN }
        val kneeRightX = FloatArray(n) { Float.NaN }
        val ankleLeftX = FloatArray(n) { Float.NaN }
        val ankleRightX = FloatArray(n) { Float.NaN }
        val shoulderLeftY = FloatArray(n) { Float.NaN }
        val shoulderRightY = FloatArray(n) { Float.NaN }
        val hipWidth = FloatArray(n) { Float.NaN }

        for (frame in poseSeq.frames) {
            val idx = frame.frameIdx
            if (idx >= n) continue

            val required = intArrayOf(
                MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP,
                MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.RIGHT_KNEE,
                MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE,
                MediaPipePoseBackend.LEFT_SHOULDER, MediaPipePoseBackend.RIGHT_SHOULDER
            )
            if (required.any { frame.confidences[it] < minConfidence }) continue

            val kp = frame.keypoints
            hipLeftX[idx] = kp[MediaPipePoseBackend.LEFT_HIP][0]
            hipRightX[idx] = kp[MediaPipePoseBackend.RIGHT_HIP][0]
            hipLeftY[idx] = kp[MediaPipePoseBackend.LEFT_HIP][1]
            hipRightY[idx] = kp[MediaPipePoseBackend.RIGHT_HIP][1]

            kneeLeftX[idx] = kp[MediaPipePoseBackend.LEFT_KNEE][0]
            kneeRightX[idx] = kp[MediaPipePoseBackend.RIGHT_KNEE][0]

            ankleLeftX[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][0]
            ankleRightX[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][0]

            shoulderLeftY[idx] = kp[MediaPipePoseBackend.LEFT_SHOULDER][1]
            shoulderRightY[idx] = kp[MediaPipePoseBackend.RIGHT_SHOULDER][1]

            hipWidth[idx] = abs(kp[MediaPipePoseBackend.RIGHT_HIP][0] - kp[MediaPipePoseBackend.LEFT_HIP][0])
        }

        val hipLx = interpolateAndSmooth(hipLeftX)
        val hipRx = interpolateAndSmooth(hipRightX)
        val hipLy = interpolateAndSmooth(hipLeftY)
        val hipRy = interpolateAndSmooth(hipRightY)
        val kneeLx = interpolateAndSmooth(kneeLeftX)
        val kneeRx = interpolateAndSmooth(kneeRightX)
        val ankLx = interpolateAndSmooth(ankleLeftX)
        val ankRx = interpolateAndSmooth(ankleRightX)
        val shLy = interpolateAndSmooth(shoulderLeftY)
        val shRy = interpolateAndSmooth(shoulderRightY)
        val hipW = interpolateAndSmooth(hipWidth)

        val ankleSepRaw = FloatArray(n) { abs(ankRx[it] - ankLx[it]) }
        val kneeSepRaw = FloatArray(n) { abs(kneeRx[it] - kneeLx[it]) }
        val hipCenter = FloatArray(n) { (hipLx[it] + hipRx[it]) / 2f }
        val hipDrop = FloatArray(n) { hipRy[it] - hipLy[it] }

        // Normalize by hip width to cancel depth/perspective drift
        val ankleSep = FloatArray(n) { ankleSepRaw[it] / (hipW[it] + 1e-8f) }
        val ankleSepValid = ankleSep.filter { !it.isNaN() }
        Log.d("GaitDebug", "BackFeatureExtractor: ankleSep valid=${ankleSepValid.size}, min=${ankleSepValid.minOrNull()}, max=${ankleSepValid.maxOrNull()}, mean=${ankleSepValid.average()}, std=${ankleSepValid.std()}")
        val kneeSep = FloatArray(n) { kneeSepRaw[it] / (hipW[it] + 1e-8f) }

        // Trunk lean angle - scale-invariant since it's an arctan ratio
        val trunkLeanAngle = FloatArray(n) {
            Math.toDegrees(atan2((shLy[it] - shRy[it]).toDouble(), (hipW[it] + 1e-8f).toDouble())).toFloat()
        }

        val steps = detectSteps(ankleSep, poseSeq.fps)
        Log.d("GaitDebug", "BackFeatureExtractor: detected ${steps.size} steps")
        if (steps.size < 4) {
            Log.d("GaitDebug", "BackFeatureExtractor: FAILED - insufficient steps")
            return null
        }

        val stepTimes = steps.map { it / poseSeq.fps }
        data class StrideRange(val start: Int, val end: Int)
        val strides = mutableListOf<StrideRange>()
        for (i in 0 until steps.size - 2) {
            strides.add(StrideRange(steps[i], steps[i + 2]))
        }
        if (strides.size < 2) return null

        // Cadence
        val stepIntervals = stepTimes.zipWithNext { a, b -> b - a }
        val meanStepTime = stepIntervals.average().toFloat()
        val cadenceSpm = if (meanStepTime > 0f) 60f / meanStepTime else 0f

        // Hip drop
        val hipDropValid = hipDrop.filter { !it.isNaN() }
        val hipDropMean = if (hipDropValid.isNotEmpty()) hipDropValid.map { abs(it) }.average().toFloat() else 0f
        val hipDropStd = hipDropValid.std()

        // Hip sway
        val hipCenterValid = hipCenter.filter { !it.isNaN() }
        val hipSwayStd = hipCenterValid.std()

        // Stride width mean + CV
        val strideWidths = strides.mapNotNull { s ->
            val end = min(s.end, ankleSep.size)
            val seg = ankleSep.slice(s.start until end).filter { !it.isNaN() }
            if (seg.isNotEmpty()) seg.average().toFloat() else null
        }
        val strideWidthMean = if (strideWidths.isNotEmpty()) strideWidths.average().toFloat() else 0f
        val strideWidthCv = if (strideWidths.isNotEmpty()) strideWidths.std() / (strideWidthMean + 1e-8f) else 0f

        // Step width asymmetry
        val leftExcursions = mutableListOf<Float>()
        val rightExcursions = mutableListOf<Float>()
        for (s in strides) {
            val end = min(s.end, ankLx.size - 1)
            val lSeg = ankLx.slice(s.start until end).filter { !it.isNaN() }
            val rSeg = ankRx.slice(s.start until end).filter { !it.isNaN() }
            if (lSeg.isNotEmpty()) leftExcursions.add(lSeg.max() - lSeg.min())
            if (rSeg.isNotEmpty()) rightExcursions.add(rSeg.max() - rSeg.min())
        }
        val meanL = if (leftExcursions.isNotEmpty()) leftExcursions.average().toFloat() else 0f
        val meanR = if (rightExcursions.isNotEmpty()) rightExcursions.average().toFloat() else 0f
        val stepWidthAsymmetry = (meanL - meanR) / (meanL + meanR + 1e-8f)

        // Knee separation
        val kneeSepValid = kneeSep.filter { !it.isNaN() }
        val kneeSeparationMean = if (kneeSepValid.isNotEmpty()) kneeSepValid.average().toFloat() else 0f

        // Trunk lean
        val trunkLeanValid = trunkLeanAngle.filter { !it.isNaN() }
        val trunkLeanMean = if (trunkLeanValid.isNotEmpty()) trunkLeanValid.map { abs(it) }.average().toFloat() else 0f
        val trunkLeanStd = trunkLeanValid.std()

        return BackGaitFeatures(
            cadence_spm = cadenceSpm,
            hip_drop_mean = hipDropMean,
            hip_drop_std = hipDropStd,
            hip_sway_std = hipSwayStd,
            stride_width_mean = strideWidthMean,
            stride_width_cv = strideWidthCv,
            step_width_asymmetry = stepWidthAsymmetry,
            knee_separation_mean = kneeSeparationMean,
            trunk_lean_mean = trunkLeanMean,
            trunk_lean_std = trunkLeanStd,
            valid_stride_count = strides.size
        )
    }

    private fun interpolateAndSmooth(arr: FloatArray, maxGap: Int = maxInterpGap, alpha: Float = emaAlpha): FloatArray {
        val n = arr.size
        val result = arr.copyOf()

        // Linear interpolate small gaps
        var lastValidIdx = -1
        for (i in 0 until n) {
            if (!result[i].isNaN()) {
                if (lastValidIdx != -1 && i - lastValidIdx - 1 <= maxGap && i - lastValidIdx > 1) {
                    val gapLen = i - lastValidIdx
                    val startVal = result[lastValidIdx]
                    val endVal = result[i]
                    for (j in 1 until gapLen) {
                        result[lastValidIdx + j] = startVal + (endVal - startVal) * (j.toFloat() / gapLen)
                    }
                }
                lastValidIdx = i
            }
        }

        // EMA smoothing
        var prev: Float? = null
        for (i in 0 until n) {
            if (result[i].isNaN()) continue
            prev = if (prev == null) result[i] else {
                val smoothed = alpha * result[i] + (1 - alpha) * prev
                result[i] = smoothed
                smoothed
            }
        }
        return result
    }

    private fun detectSteps(signal: FloatArray, fps: Float): List<Int> {
        val valid = signal.filter { !it.isNaN() }
        if (valid.size < 10) return emptyList()

        val clean = signal.map { if (it.isNaN()) 0f else it }
        val minDistance = max((fps * minStepTimeS).toInt(), 5)
        val std = valid.std()
        val prominence = std * 0.33f
        Log.d("GaitDebug", "detectSteps: validCount=${valid.size}, std=$std, prominence=$prominence, minDistance=$minDistance")

        val peaks = mutableListOf<Int>()
        var i = 1
        while (i < clean.size - 1) {
            if (clean[i] > clean[i - 1] && clean[i] > clean[i + 1]) {
                // Check prominence: value must exceed neighbors by prominence threshold
                val left = (i downTo max(0, i - minDistance)).map { clean[it] }.minOrNull() ?: clean[i]
                val right = (i..min(clean.size - 1, i + minDistance)).map { clean[it] }.minOrNull() ?: clean[i]
                val prom = clean[i] - max(left, right)
                if (prom >= prominence) {
                    if (peaks.isEmpty() || i - peaks.last() >= minDistance) {
                        peaks.add(i)
                    }
                }
            }
            i++
        }
        return peaks
    }

    private fun List<Float>.std(): Float {
        if (isEmpty()) return 0f
        val mean = average().toFloat()
        val variance = map { (it - mean).pow(2) }.average().toFloat()
        return sqrt(variance)
    }

    private fun FloatArray.std(): Float = this.toList().std()
}