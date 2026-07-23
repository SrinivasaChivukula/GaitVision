package GaitVision.com.gait

import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.hypot

/**
 * Computes pose-position jitter metrics for filter validation.
 *
 * This is not a clinical gait feature and is not fed into the scoring model.
 * It exists to make smoothing changes testable across videos.
 */
object PoseJitterAnalyzer {
    private val trackedLandmarks = intArrayOf(
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

    private val bodyScalePairs = arrayOf(
        MediaPipePoseBackend.LEFT_SHOULDER to MediaPipePoseBackend.RIGHT_SHOULDER,
        MediaPipePoseBackend.LEFT_HIP to MediaPipePoseBackend.RIGHT_HIP,
        MediaPipePoseBackend.LEFT_SHOULDER to MediaPipePoseBackend.LEFT_HIP,
        MediaPipePoseBackend.RIGHT_SHOULDER to MediaPipePoseBackend.RIGHT_HIP
    )

    private const val SNAP_DELTA_BODY_SCALE = 0.35f

    fun compare(
        raw: PoseSequence,
        smoothed: PoseSequence,
        minConfidence: Float = GaitConfig.MIN_CONFIDENCE
    ): PoseJitterComparison {
        return PoseJitterComparison(
            raw = compute(raw, minConfidence),
            smoothed = compute(smoothed, minConfidence)
        )
    }

    fun compute(
        poseSeq: PoseSequence,
        minConfidence: Float = GaitConfig.MIN_CONFIDENCE
    ): PoseJitterMetrics {
        val frames = poseSeq.frames.sortedBy { it.frameIdx }
        if (frames.isEmpty()) {
            return PoseJitterMetrics(
                numPoseFrames = 0,
                detectionRate = 0f,
                confidenceCoverage = 0f,
                medianBodyScale = Float.NaN,
                jitterSecondDiffNorm = Float.NaN,
                meanVelocityNorm = Float.NaN,
                snapRate = Float.NaN
            )
        }

        val validSampleCount = frames.sumOf { frame ->
            trackedLandmarks.count { hasLandmark(frame, it, minConfidence) }
        }
        val totalFrameCount = poseSeq.numFramesTotal.coerceAtLeast(frames.size)
        val totalPossibleSamples = totalFrameCount * trackedLandmarks.size
        val confidenceCoverage = if (totalPossibleSamples > 0) {
            validSampleCount.toFloat() / totalPossibleSamples
        } else {
            0f
        }

        val scalesByFrame = frames.map { bodyScale(it, minConfidence) }
        val medianBodyScale = median(scalesByFrame.filterFinitePositive())

        val secondDiffs = ArrayList<Float>()
        val velocities = ArrayList<Float>()
        var snapCount = 0
        var velocitySampleCount = 0

        for (i in 1 until frames.size) {
            val previous = frames[i - 1]
            val current = frames[i]
            if (current.frameIdx != previous.frameIdx + 1) continue

            val scale = representativeScale(
                medianBodyScale,
                scalesByFrame[i - 1],
                scalesByFrame[i]
            ) ?: continue

            for (landmark in trackedLandmarks) {
                if (!hasLandmark(previous, landmark, minConfidence) ||
                    !hasLandmark(current, landmark, minConfidence)
                ) {
                    continue
                }
                val delta = distance(previous, current, landmark) / scale
                velocities.add(delta)
                velocitySampleCount++
                if (delta > SNAP_DELTA_BODY_SCALE) snapCount++
            }
        }

        for (i in 1 until frames.lastIndex) {
            val previous = frames[i - 1]
            val current = frames[i]
            val next = frames[i + 1]
            if (current.frameIdx != previous.frameIdx + 1 ||
                next.frameIdx != current.frameIdx + 1
            ) {
                continue
            }

            val scale = representativeScale(
                medianBodyScale,
                scalesByFrame[i - 1],
                scalesByFrame[i],
                scalesByFrame[i + 1]
            ) ?: continue

            for (landmark in trackedLandmarks) {
                if (!hasLandmark(previous, landmark, minConfidence) ||
                    !hasLandmark(current, landmark, minConfidence) ||
                    !hasLandmark(next, landmark, minConfidence)
                ) {
                    continue
                }

                val p0 = previous.keypoints[landmark]
                val p1 = current.keypoints[landmark]
                val p2 = next.keypoints[landmark]
                val ddx = p2[0] - 2f * p1[0] + p0[0]
                val ddy = p2[1] - 2f * p1[1] + p0[1]
                secondDiffs.add(hypot(ddx, ddy) / scale)
            }
        }

        val meanVelocity = if (velocities.isNotEmpty()) velocities.average().toFloat() else Float.NaN
        val snapRate = if (velocitySampleCount > 0) snapCount.toFloat() / velocitySampleCount else Float.NaN

        return PoseJitterMetrics(
            numPoseFrames = frames.size,
            detectionRate = poseSeq.detectionRate,
            confidenceCoverage = confidenceCoverage,
            medianBodyScale = medianBodyScale,
            jitterSecondDiffNorm = median(secondDiffs),
            meanVelocityNorm = meanVelocity,
            snapRate = snapRate
        )
    }

    private fun hasLandmark(frame: PoseFrame, landmark: Int, minConfidence: Float): Boolean {
        val point = frame.keypoints.getOrNull(landmark) ?: return false
        val confidence = frame.confidences.getOrNull(landmark) ?: return false
        return confidence >= minConfidence &&
            point.size >= 2 &&
            point[0].isFinite() &&
            point[1].isFinite()
    }

    private fun bodyScale(frame: PoseFrame, minConfidence: Float): Float {
        val distances = bodyScalePairs.mapNotNull { (a, b) ->
            if (hasLandmark(frame, a, minConfidence) && hasLandmark(frame, b, minConfidence)) {
                distance(frame, a, b).takeIf { it.isFinite() && it > 1e-4f }
            } else {
                null
            }
        }
        return median(distances)
    }

    private fun representativeScale(medianBodyScale: Float, vararg candidates: Float): Float? {
        val localScale = median(candidates.toList().filterFinitePositive())
        return when {
            localScale.isFinite() && localScale > 0f -> localScale
            medianBodyScale.isFinite() && medianBodyScale > 0f -> medianBodyScale
            else -> null
        }
    }

    private fun distance(first: PoseFrame, second: PoseFrame, landmark: Int): Float {
        val a = first.keypoints[landmark]
        val b = second.keypoints[landmark]
        return hypot(b[0] - a[0], b[1] - a[1])
    }

    private fun distance(frame: PoseFrame, firstLandmark: Int, secondLandmark: Int): Float {
        val a = frame.keypoints[firstLandmark]
        val b = frame.keypoints[secondLandmark]
        return hypot(b[0] - a[0], b[1] - a[1])
    }

    private fun median(values: List<Float>): Float {
        val clean = values.filter { it.isFinite() }
        if (clean.isEmpty()) return Float.NaN
        val sorted = clean.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2f
        } else {
            sorted[middle]
        }
    }

    private fun List<Float>.filterFinitePositive(): List<Float> {
        return filter { it.isFinite() && it > 0f }
    }
}
