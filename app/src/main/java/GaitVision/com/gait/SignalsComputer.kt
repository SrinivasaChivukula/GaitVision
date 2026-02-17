package GaitVision.com.gait

import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*

internal object SignalsComputer {
    private val CORE_KEYPOINTS = intArrayOf(
        MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP,
        MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.RIGHT_KNEE,
        MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE
    )

    fun computeAndProcess(
        poseSeq: PoseSequence,
        minConfidence: Float,
        maxInterpGap: Int,
        emaAlpha: Float
    ): Signals {
        var signals = computeSignals(poseSeq, minConfidence)
        signals = interpolateSignals(signals, maxInterpGap)
        signals = smoothSignals(signals, emaAlpha, maxInterpGap)
        return computeVelocities(signals, poseSeq.fps)
    }

    private fun computeSignals(poseSeq: PoseSequence, minConfidence: Float): Signals {
        val n = poseSeq.numFramesTotal
        val timestamps = FloatArray(n) { Float.NaN }
        val frameIndices = IntArray(n) { it }
        val isValid = BooleanArray(n) { false }
        val interAnkleDist = FloatArray(n) { Float.NaN }
        val kneeAngleLeft = FloatArray(n) { Float.NaN }
        val kneeAngleRight = FloatArray(n) { Float.NaN }
        val trunkAngle = FloatArray(n) { Float.NaN }
        val ankleAngleLeft = FloatArray(n) { Float.NaN }
        val ankleAngleRight = FloatArray(n) { Float.NaN }
        val hipAngleLeft = FloatArray(n) { Float.NaN }
        val hipAngleRight = FloatArray(n) { Float.NaN }
        val strideAngle = FloatArray(n) { Float.NaN }
        val ankleLeftX = FloatArray(n) { Float.NaN }
        val ankleRightX = FloatArray(n) { Float.NaN }
        val ankleLeftY = FloatArray(n) { Float.NaN }
        val ankleRightY = FloatArray(n) { Float.NaN }
        val hipLeftY = FloatArray(n) { Float.NaN }
        val hipRightY = FloatArray(n) { Float.NaN }
        val heelLeftY = FloatArray(n) { Float.NaN }
        val heelRightY = FloatArray(n) { Float.NaN }
        val toeLeftY = FloatArray(n) { Float.NaN }
        val toeRightY = FloatArray(n) { Float.NaN }
        val midHipX = FloatArray(n) { Float.NaN }

        for (frame in poseSeq.frames) {
            val idx = frame.frameIdx
            if (idx >= n) continue
            timestamps[idx] = frame.timestampS
            val coreValid = CORE_KEYPOINTS.all { frame.confidences[it] >= minConfidence }
            isValid[idx] = coreValid
            if (!coreValid) continue

            val kp = frame.keypoints
            ankleLeftX[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][0]
            ankleRightX[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][0]
            interAnkleDist[idx] = abs(kp[MediaPipePoseBackend.RIGHT_ANKLE][0] - kp[MediaPipePoseBackend.LEFT_ANKLE][0])
            ankleLeftY[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][1]
            ankleRightY[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][1]
            hipLeftY[idx] = kp[MediaPipePoseBackend.LEFT_HIP][1]
            hipRightY[idx] = kp[MediaPipePoseBackend.RIGHT_HIP][1]
            heelLeftY[idx] = kp[MediaPipePoseBackend.LEFT_HEEL][1]
            heelRightY[idx] = kp[MediaPipePoseBackend.RIGHT_HEEL][1]
            toeLeftY[idx] = kp[MediaPipePoseBackend.LEFT_FOOT_INDEX][1]
            toeRightY[idx] = kp[MediaPipePoseBackend.RIGHT_FOOT_INDEX][1]
            midHipX[idx] = (kp[MediaPipePoseBackend.LEFT_HIP][0] + kp[MediaPipePoseBackend.RIGHT_HIP][0]) / 2f

            kneeAngleLeft[idx] = computeAngle(kp[MediaPipePoseBackend.LEFT_HIP], kp[MediaPipePoseBackend.LEFT_KNEE], kp[MediaPipePoseBackend.LEFT_ANKLE])
            kneeAngleRight[idx] = computeAngle(kp[MediaPipePoseBackend.RIGHT_HIP], kp[MediaPipePoseBackend.RIGHT_KNEE], kp[MediaPipePoseBackend.RIGHT_ANKLE])
            val midShoulder = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][0] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][1] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][1]) / 2f
            )
            val midHip = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_HIP][0] + kp[MediaPipePoseBackend.RIGHT_HIP][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_HIP][1] + kp[MediaPipePoseBackend.RIGHT_HIP][1]) / 2f
            )
            trunkAngle[idx] = computeTrunkLean(midShoulder, midHip)
            ankleAngleLeft[idx] = computeAngle(kp[MediaPipePoseBackend.LEFT_FOOT_INDEX], kp[MediaPipePoseBackend.LEFT_ANKLE], kp[MediaPipePoseBackend.LEFT_KNEE]) - 90f
            ankleAngleRight[idx] = computeAngle(kp[MediaPipePoseBackend.RIGHT_FOOT_INDEX], kp[MediaPipePoseBackend.RIGHT_ANKLE], kp[MediaPipePoseBackend.RIGHT_KNEE]) - 90f
            hipAngleLeft[idx] = 180f - computeAngle(kp[MediaPipePoseBackend.LEFT_KNEE], kp[MediaPipePoseBackend.LEFT_HIP], kp[MediaPipePoseBackend.LEFT_SHOULDER])
            hipAngleRight[idx] = 180f - computeAngle(kp[MediaPipePoseBackend.RIGHT_KNEE], kp[MediaPipePoseBackend.RIGHT_HIP], kp[MediaPipePoseBackend.RIGHT_SHOULDER])
            strideAngle[idx] = computeAngle(kp[MediaPipePoseBackend.LEFT_HEEL], midHip, kp[MediaPipePoseBackend.RIGHT_HEEL])
        }

        return Signals(
            timestamps, frameIndices, isValid,
            interAnkleDist, kneeAngleLeft, kneeAngleRight, trunkAngle,
            ankleAngleLeft, ankleAngleRight, hipAngleLeft, hipAngleRight, strideAngle,
            ankleLeftX, ankleRightX, ankleLeftY, ankleRightY, hipLeftY, hipRightY,
            heelLeftY, heelRightY, toeLeftY, toeRightY, midHipX,
            FloatArray(n) { Float.NaN }, FloatArray(n) { Float.NaN }, FloatArray(n) { Float.NaN }
        )
    }

    private fun computeAngle(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
        val v1x = p1[0] - p2[0]; val v1y = p1[1] - p2[1]
        val v2x = p3[0] - p2[0]; val v2y = p3[1] - p2[1]
        val dot = v1x * v2x + v1y * v2y
        val mag1 = sqrt(v1x * v1x + v1y * v1y)
        val mag2 = sqrt(v2x * v2x + v2y * v2y)
        val cosAngle = (dot / (mag1 * mag2 + 1e-8f)).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(cosAngle).toDouble()).toFloat()
    }

    private fun computeTrunkLean(midShoulder: FloatArray, midHip: FloatArray): Float {
        val dx = midShoulder[0] - midHip[0]
        val dy = midShoulder[1] - midHip[1]
        return Math.toDegrees(atan2(dx, -dy).toDouble()).toFloat()
    }

    private fun interpolateSignals(signals: Signals, maxGap: Int): Signals {
        listOf(signals.interAnkleDist, signals.kneeAngleLeft, signals.kneeAngleRight,
            signals.trunkAngle, signals.ankleLeftX, signals.ankleRightX,
            signals.ankleLeftY, signals.ankleRightY, signals.hipLeftY, signals.hipRightY,
            signals.heelLeftY, signals.heelRightY, signals.toeLeftY, signals.toeRightY, signals.midHipX
        ).forEach { interpolateArray(it, maxGap) }
        return signals
    }

    private fun interpolateArray(arr: FloatArray, maxGap: Int) {
        var inGap = false
        var gapStart = 0
        for (i in arr.indices) {
            if (arr[i].isNaN()) {
                if (!inGap) { inGap = true; gapStart = i }
            } else {
                if (inGap) {
                    val gapLen = i - gapStart
                    if (gapLen <= maxGap && gapStart > 0) {
                        val prevVal = arr[gapStart - 1]
                        val nextVal = arr[i]
                        for (j in gapStart until i) {
                            val t = (j - gapStart + 1).toFloat() / (gapLen + 1)
                            arr[j] = prevVal + t * (nextVal - prevVal)
                        }
                    }
                    inGap = false
                }
            }
        }
    }

    private fun smoothSignals(signals: Signals, alpha: Float, maxBridgeGap: Int): Signals {
        listOf(signals.interAnkleDist, signals.kneeAngleLeft, signals.kneeAngleRight,
            signals.trunkAngle, signals.ankleLeftX, signals.ankleRightX,
            signals.ankleLeftY, signals.ankleRightY, signals.hipLeftY, signals.hipRightY,
            signals.heelLeftY, signals.heelRightY, signals.toeLeftY, signals.toeRightY, signals.midHipX
        ).forEach { emaSmoothGapAware(it, alpha, maxBridgeGap) }
        return signals
    }

    private fun emaSmoothGapAware(arr: FloatArray, alpha: Float, maxBridgeGap: Int) {
        val firstValid = arr.indices.firstOrNull { !arr[it].isNaN() } ?: return
        var prev = arr[firstValid]
        var nanRunLength = 0
        for (i in (firstValid + 1) until arr.size) {
            if (arr[i].isNaN()) {
                nanRunLength++
                if (nanRunLength <= maxBridgeGap) arr[i] = prev
            } else {
                if (nanRunLength > maxBridgeGap) prev = arr[i]
                else { arr[i] = alpha * arr[i] + (1 - alpha) * prev; prev = arr[i] }
                nanRunLength = 0
            }
        }
    }

    private fun computeVelocities(signals: Signals, fps: Float): Signals {
        val dt = 1f / fps
        val n = signals.ankleLeftY.size
        for (i in 0 until n - 1) {
            if (!signals.ankleLeftY[i].isNaN() && !signals.ankleLeftY[i + 1].isNaN())
                signals.ankleLeftVy[i] = (signals.ankleLeftY[i + 1] - signals.ankleLeftY[i]) / dt
            if (!signals.ankleRightY[i].isNaN() && !signals.ankleRightY[i + 1].isNaN())
                signals.ankleRightVy[i] = (signals.ankleRightY[i + 1] - signals.ankleRightY[i]) / dt
            val hipAvgCurr = (signals.hipLeftY[i] + signals.hipRightY[i]) / 2f
            val hipAvgNext = (signals.hipLeftY[i + 1] + signals.hipRightY[i + 1]) / 2f
            if (!hipAvgCurr.isNaN() && !hipAvgNext.isNaN())
                signals.hipAvgVy[i] = (hipAvgNext - hipAvgCurr) / dt
        }
        val last = n - 1
        if (last > 0) {
            if (!signals.ankleLeftY[last].isNaN() && !signals.ankleLeftY[last - 1].isNaN())
                signals.ankleLeftVy[last] = (signals.ankleLeftY[last] - signals.ankleLeftY[last - 1]) / dt
            if (!signals.ankleRightY[last].isNaN() && !signals.ankleRightY[last - 1].isNaN())
                signals.ankleRightVy[last] = (signals.ankleRightY[last] - signals.ankleRightY[last - 1]) / dt
            val hipAvgLast = (signals.hipLeftY[last] + signals.hipRightY[last]) / 2f
            val hipAvgPrev = (signals.hipLeftY[last - 1] + signals.hipRightY[last - 1]) / 2f
            if (!hipAvgLast.isNaN() && !hipAvgPrev.isNaN())
                signals.hipAvgVy[last] = (hipAvgLast - hipAvgPrev) / dt
        }
        return signals
    }
}
