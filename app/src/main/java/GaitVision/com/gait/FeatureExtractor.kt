package GaitVision.com.gait

import android.util.Log
import GaitVision.com.AnalysisSession
import GaitVision.com.enableVerboseLogging
import GaitVision.com.video.getLRSwapIndex
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*

/**
 * Feature Extractor for Gait Analysis - thin orchestrator.
 * Delegates to SignalsComputer, CycleDetector, FeatureComputer.
 */
class FeatureExtractor(
    private val minConfidence: Float = GaitConfig.MIN_CONFIDENCE,
    private val maxInterpGap: Int = GaitConfig.MAX_INTERP_GAP,
    private val emaAlpha: Float = GaitConfig.EMA_ALPHA,
    private val minStepTimeS: Float = GaitConfig.MIN_STEP_TIME_S,
    private val maxStepTimeS: Float = GaitConfig.MAX_STEP_TIME_S,
    private val stepDistanceFactor: Float = GaitConfig.STEP_DISTANCE_FACTOR,
    private val stepProminenceFactor: Float = GaitConfig.STEP_PROMINENCE_FACTOR,
    private val useRobustExtrema: Boolean = GaitConfig.USE_ROBUST_EXTREMA,
    private val extremaPercentileLo: Float = GaitConfig.EXTREMA_PERCENTILE_LO,
    private val extremaPercentileHi: Float = GaitConfig.EXTREMA_PERCENTILE_HI
) {
    companion object {
        private const val TAG = "GaitLogging"
    }

    fun extract(poseSeq: PoseSequence): Pair<GaitFeatures?, GaitDiagnostics> {
        Log.d(TAG, "Video info: numFramesTotal=${poseSeq.numFramesTotal}, detected=${poseSeq.frames.size}, fps=${poseSeq.fps}, detectionRate=${String.format("%.1f", poseSeq.detectionRate * 100)}%")

        if (poseSeq.frames.size < 20) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.UNPROCESSABLE, "too_few_frames"))
        }
        if (poseSeq.detectionRate < 0.3f) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.LOW_DETECTION, "detection_rate_${(poseSeq.detectionRate * 100).toInt()}%"))
        }

        if (enableVerboseLogging) {
            Log.d(TAG, "VERBOSE logging ON")
        }

        var signals = SignalsComputer.computeAndProcess(poseSeq, minConfidence, maxInterpGap, emaAlpha)
        val steps = CycleDetector.detectSteps(signals.interAnkleDist, poseSeq.fps, minStepTimeS, maxStepTimeS, stepDistanceFactor, stepProminenceFactor)
        Log.d(TAG, "Step detection: inter_ankle, detected ${steps.size} steps")

        if (enableVerboseLogging) {
            Log.d(TAG, "DETECTED PEAKS: ${steps.map { it.frameIdx }}")
        }

        if (steps.size < 4) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, "only_${steps.size}_steps", steps.size))
        }

        val strides = CycleDetector.segmentStrides(steps, signals, poseSeq.fps)
        val validatedStrides = CycleDetector.validateStrides(strides, signals, poseSeq.fps, minStepTimeS, maxStepTimeS, useRobustExtrema, extremaPercentileLo, extremaPercentileHi)
        val validStrides = validatedStrides.filter { it.isValid }
        Log.d(TAG, "Scored strides: ${validStrides.size} valid of ${validatedStrides.size}")

        if (validStrides.size < 2) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, "only_${validStrides.size}_valid_strides", steps.size, validStrides.size))
        }

        val (features, selectionReason, selectedIndices) = FeatureComputer.compute(signals, validStrides, poseSeq, extremaPercentileLo, extremaPercentileHi)
        val qualityFlag = if (features.valid_stride_count == 0) QualityFlag.NO_CYCLES else QualityFlag.OK
        val diagnostics = createDiagnostics(poseSeq, qualityFlag, "", steps.size, validStrides.size, selectionReason)

        AnalysisSession.extractedSignals = signals
        AnalysisSession.extractedStrides = validatedStrides
        AnalysisSession.selectedStrideIndices = selectedIndices
        AnalysisSession.stepSignalMode = "inter_ankle"

        if (features.valid_stride_count == 0) return Pair(null, diagnostics)
        return Pair(features, diagnostics)
    }

    fun determineWalkingDirection(poseSeq: PoseSequence): String {
        val hipXPositions = mutableListOf<Float>()
        for (frame in poseSeq.frames) {
            val leftHip = frame.keypoints[MediaPipePoseBackend.LEFT_HIP]
            val rightHip = frame.keypoints[MediaPipePoseBackend.RIGHT_HIP]
            val leftConf = frame.confidences[MediaPipePoseBackend.LEFT_HIP]
            val rightConf = frame.confidences[MediaPipePoseBackend.RIGHT_HIP]
            if (leftConf > minConfidence && rightConf > minConfidence) {
                hipXPositions.add((leftHip[0] + rightHip[0]) / 2f)
            }
        }
        if (hipXPositions.size < 10) return "unknown"
        val first = hipXPositions.take(hipXPositions.size / 3).average().toFloat()
        val last = hipXPositions.takeLast(hipXPositions.size / 3).average().toFloat()
        return if (last > first) "left_to_right" else "right_to_left"
    }

    fun normalizeDirection(poseSeq: PoseSequence): PoseSequence {
        val direction = determineWalkingDirection(poseSeq)
        if (direction == "unknown") return poseSeq
        val shouldFlip = direction == "right_to_left"
        if (!shouldFlip) return poseSeq.copy(walkingDirection = direction, wasFlipped = false)
        val flippedFrames = poseSeq.frames.map { frame ->
            val newKp = Array(33) { i -> frame.keypoints[getLRSwapIndex(i)].clone() }
            val newConf = FloatArray(33) { i -> frame.confidences[getLRSwapIndex(i)] }
            frame.copy(keypoints = newKp, confidences = newConf)
        }
        return poseSeq.copy(frames = flippedFrames, walkingDirection = "left_to_right", wasFlipped = true)
    }

    private fun createDiagnostics(
        poseSeq: PoseSequence,
        qualityFlag: QualityFlag,
        reason: String,
        numSteps: Int = 0,
        numValidStrides: Int = 0,
        selectionReason: String = ""
    ): GaitDiagnostics = GaitDiagnostics(
        videoId = poseSeq.videoId,
        fpsDetected = poseSeq.fps,
        durationS = poseSeq.durationS,
        numFramesTotal = poseSeq.numFramesTotal,
        numFramesValid = poseSeq.frames.size,
        validFrameRate = poseSeq.detectionRate,
        numStepsDetected = numSteps,
        numStridesValid = numValidStrides,
        estimatedCadenceSpm = if (numSteps > 1 && poseSeq.durationS > 0) numSteps * 60f / poseSeq.durationS else 0f,
        walkingDirection = poseSeq.walkingDirection,
        wasFlipped = poseSeq.wasFlipped,
        qualityFlag = qualityFlag,
        rejectionReasons = if (reason.isNotEmpty()) listOf(reason) else emptyList()
    )
}
