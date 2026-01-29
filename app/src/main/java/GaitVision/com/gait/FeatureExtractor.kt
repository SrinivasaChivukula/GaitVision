package GaitVision.com.gait

import android.util.Log
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*

/**
 * Feature Extractor for Gait Analysis - Kotlin port of PC pipeline (pc_tasks/feature_extractor.py).
 * 
 * Extracts 16 scalar gait features from pose sequences.
 * Includes multi-mode step detection and quality-based stride selection.
 */
class FeatureExtractor(
    private val minConfidence: Float = GaitConfig.MIN_CONFIDENCE,
    private val maxInterpGap: Int = GaitConfig.MAX_INTERP_GAP,
    private val emaAlpha: Float = GaitConfig.EMA_ALPHA,
    private val minStepTimeS: Float = GaitConfig.MIN_STEP_TIME_S,
    private val maxStepTimeS: Float = GaitConfig.MAX_STEP_TIME_S,
    private val stepDistanceFactor: Float = GaitConfig.STEP_DISTANCE_FACTOR,
    private val stepProminenceFactor: Float = GaitConfig.STEP_PROMINENCE_FACTOR,
    private val validFramePct: Float = GaitConfig.VALID_FRAME_PCT,
    private val stepTimeTolerance: Float = GaitConfig.STEP_TIME_TOLERANCE,
    private val kneeRomMin: Float = GaitConfig.KNEE_ROM_MIN,
    private val kneeRomMax: Float = GaitConfig.KNEE_ROM_MAX,
    private val useRobustExtrema: Boolean = GaitConfig.USE_ROBUST_EXTREMA,
    private val extremaPercentileLo: Float = GaitConfig.EXTREMA_PERCENTILE_LO,
    private val extremaPercentileHi: Float = GaitConfig.EXTREMA_PERCENTILE_HI
) {
    companion object {
        private const val TAG = "FeatureExtractor"
        
        // Core keypoints for gait analysis
        val CORE_KEYPOINTS = intArrayOf(
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.RIGHT_ANKLE
        )
    }
    
    /**
     * Step signal detection modes - all are leg-swap robust.
     */
    enum class StepSignalMode(val value: String) {
        INTER_ANKLE("inter_ankle"),
        MAX_ANKLE_VY("max_ankle_vy"),
        MIN_KNEE_ANGLE("min_knee_angle")
    }
    
    /**
     * Quality metrics for a step signal mode.
     */
    data class StepSignalQuality(
        val mode: StepSignalMode,
        val peakCount: Int,
        val periodicityScore: Float,
        val confidenceCoverage: Float,
        val finalScore: Float,
        val rejectionReason: String = ""
    )
    
    /**
     * Extract gait features from pose sequence.
     */
    fun extract(poseSeq: PoseSequence): Pair<GaitFeatures?, GaitDiagnostics> {
        Log.d(TAG, "Extracting features from ${poseSeq.frames.size} frames")
        
        // Basic checks
        if (poseSeq.frames.size < 20) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.UNPROCESSABLE, "too_few_frames"))
        }
        
        if (poseSeq.detectionRate < 0.3f) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.LOW_DETECTION, 
                "detection_rate_${(poseSeq.detectionRate * 100).toInt()}%"))
        }
        
        // Step 1: Compute signals
        var signals = computeSignals(poseSeq)
        
        // Step 2: Interpolate gaps
        signals = interpolateSignals(signals)
        
        // Step 3: Light smoothing
        signals = smoothSignals(signals)
        
        // Step 4: Compute velocities
        signals = computeVelocities(signals, poseSeq.fps)
        
        // Step 5: Determine near leg
        val nearLeg = determineNearLeg(poseSeq)
        
        // Step 6: Multi-mode step detection
        val (stepMode, stepSignal, modeScores) = selectStepSignalMode(signals, poseSeq, nearLeg, poseSeq.fps)
        val steps = detectStepsFromSignal(stepSignal, poseSeq.fps)
        Log.d(TAG, "Step signal mode: ${stepMode.value}, detected ${steps.size} steps")
        
        if (steps.size < 4) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, 
                "only_${steps.size}_steps", steps.size, stepMode = stepMode.value))
        }
        
        // Step 7: Segment strides
        val strides = segmentStrides(steps, signals, poseSeq.fps)
        
        // Step 8: Validate strides
        val validatedStrides = validateStrides(strides, signals, poseSeq.fps)
        val validStrides = validatedStrides.filter { it.isValid }
        Log.d(TAG, "Valid strides: ${validStrides.size}")
        
        if (validStrides.size < 2) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, 
                "only_${validStrides.size}_valid_strides", steps.size, validStrides.size, stepMode = stepMode.value))
        }
        
        // Step 9: Quality-based stride selection and feature computation
        val (features, selectionReason, selectedIndices) = computeFeatures(signals, validStrides, poseSeq)
        
        val qualityFlag = when {
            features.valid_stride_count == 0 && validStrides.size >= 2 -> QualityFlag.NO_CYCLES
            features.valid_stride_count == 0 -> QualityFlag.NO_CYCLES
            else -> QualityFlag.OK
        }
        
        val diagnostics = createDiagnostics(
            poseSeq, qualityFlag, "",
            steps.size, validStrides.size,
            stepMode = stepMode.value,
            selectionReason = selectionReason
        )
        
        // Store signals and strides for visualization
        GaitVision.com.extractedSignals = signals
        GaitVision.com.extractedStrides = validatedStrides
        GaitVision.com.stepSignalMode = stepMode.value
        
        if (features.valid_stride_count == 0) {
            return Pair(null, diagnostics)
        }
        
        return Pair(features, diagnostics)
    }
    
    /**
     * Determine walking direction from hip positions.
     */
    fun determineWalkingDirection(poseSeq: PoseSequence): String {
        val hipXPositions = mutableListOf<Float>()
        
        for (frame in poseSeq.frames) {
            val leftHip = frame.keypoints[MediaPipePoseBackend.LEFT_HIP]
            val rightHip = frame.keypoints[MediaPipePoseBackend.RIGHT_HIP]
            val leftConf = frame.confidences[MediaPipePoseBackend.LEFT_HIP]
            val rightConf = frame.confidences[MediaPipePoseBackend.RIGHT_HIP]
            
            if (leftConf > minConfidence && rightConf > minConfidence) {
                val midHipX = (leftHip[0] + rightHip[0]) / 2f
                hipXPositions.add(midHipX)
            }
        }
        
        if (hipXPositions.size < 10) {
            return "left_to_right"
        }
        
        val firstQuarter = hipXPositions.take(hipXPositions.size / 4)
        val lastQuarter = hipXPositions.takeLast(hipXPositions.size / 4)
        
        val firstAvg = firstQuarter.average().toFloat()
        val lastAvg = lastQuarter.average().toFloat()
        
        return if (lastAvg > firstAvg) "left_to_right" else "right_to_left"
    }
    
    /**
     * Normalize walking direction - flip if needed so subject walks left to right.
     */
    fun normalizeDirection(poseSeq: PoseSequence): PoseSequence {
        val direction = determineWalkingDirection(poseSeq)
        
        if (direction == "left_to_right") {
            return poseSeq.copy(walkingDirection = direction, wasFlipped = false)
        }
        
        val flippedFrames = poseSeq.frames.map { frame ->
            val flippedKeypoints = Array(33) { floatArrayOf(0f, 0f) }
            val flippedConfidences = FloatArray(33)
            
            for (i in 0 until 33) {
                val swapIdx = getSwapIndex(i)
                flippedKeypoints[swapIdx][0] = 1f - frame.keypoints[i][0]
                flippedKeypoints[swapIdx][1] = frame.keypoints[i][1]
                flippedConfidences[swapIdx] = frame.confidences[i]
            }
            
            frame.copy(keypoints = flippedKeypoints, confidences = flippedConfidences)
        }
        
        return poseSeq.copy(
            frames = flippedFrames,
            walkingDirection = direction,
            wasFlipped = true
        )
    }
    
    private fun getSwapIndex(idx: Int): Int {
        return when (idx) {
            1 -> 4; 4 -> 1
            2 -> 5; 5 -> 2
            3 -> 6; 6 -> 3
            7 -> 8; 8 -> 7
            9 -> 10; 10 -> 9
            11 -> 12; 12 -> 11
            13 -> 14; 14 -> 13
            15 -> 16; 16 -> 15
            17 -> 18; 18 -> 17
            19 -> 20; 20 -> 19
            21 -> 22; 22 -> 21
            23 -> 24; 24 -> 23
            25 -> 26; 26 -> 25
            27 -> 28; 28 -> 27
            29 -> 30; 30 -> 29
            31 -> 32; 32 -> 31
            else -> idx
        }
    }
    
    // =========================================================================
    // Signal Computation
    // =========================================================================
    
    private fun computeSignals(poseSeq: PoseSequence): Signals {
        val n = poseSeq.numFramesTotal
        
        val timestamps = FloatArray(n) { Float.NaN }
        val frameIndices = IntArray(n) { it }
        val isValid = BooleanArray(n) { false }
        
        val interAnkleDist = FloatArray(n) { Float.NaN }
        val kneeAngleLeft = FloatArray(n) { Float.NaN }
        val kneeAngleRight = FloatArray(n) { Float.NaN }
        val trunkAngle = FloatArray(n) { Float.NaN }
        val ankleLeftX = FloatArray(n) { Float.NaN }
        val ankleRightX = FloatArray(n) { Float.NaN }
        val ankleLeftY = FloatArray(n) { Float.NaN }
        val ankleRightY = FloatArray(n) { Float.NaN }
        val hipLeftY = FloatArray(n) { Float.NaN }
        val hipRightY = FloatArray(n) { Float.NaN }
        
        for (frame in poseSeq.frames) {
            val idx = frame.frameIdx
            if (idx >= n) continue
            
            timestamps[idx] = frame.timestampS
            
            val coreValid = CORE_KEYPOINTS.all { kp ->
                frame.confidences[kp] >= minConfidence
            }
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
            
            kneeAngleLeft[idx] = computeAngle(
                kp[MediaPipePoseBackend.LEFT_HIP],
                kp[MediaPipePoseBackend.LEFT_KNEE],
                kp[MediaPipePoseBackend.LEFT_ANKLE]
            )
            kneeAngleRight[idx] = computeAngle(
                kp[MediaPipePoseBackend.RIGHT_HIP],
                kp[MediaPipePoseBackend.RIGHT_KNEE],
                kp[MediaPipePoseBackend.RIGHT_ANKLE]
            )
            
            val midShoulder = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][0] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][1] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][1]) / 2f
            )
            val midHip = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_HIP][0] + kp[MediaPipePoseBackend.RIGHT_HIP][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_HIP][1] + kp[MediaPipePoseBackend.RIGHT_HIP][1]) / 2f
            )
            trunkAngle[idx] = computeTrunkLean(midShoulder, midHip)
        }
        
        return Signals(
            timestamps = timestamps,
            frameIndices = frameIndices,
            isValid = isValid,
            interAnkleDist = interAnkleDist,
            kneeAngleLeft = kneeAngleLeft,
            kneeAngleRight = kneeAngleRight,
            trunkAngle = trunkAngle,
            ankleLeftX = ankleLeftX,
            ankleRightX = ankleRightX,
            ankleLeftY = ankleLeftY,
            ankleRightY = ankleRightY,
            hipLeftY = hipLeftY,
            hipRightY = hipRightY,
            ankleLeftVy = FloatArray(n) { Float.NaN },
            ankleRightVy = FloatArray(n) { Float.NaN },
            hipAvgVy = FloatArray(n) { Float.NaN }
        )
    }
    
    private fun computeAngle(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
        val v1x = p1[0] - p2[0]
        val v1y = p1[1] - p2[1]
        val v2x = p3[0] - p2[0]
        val v2y = p3[1] - p2[1]
        
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
    
    // =========================================================================
    // Signal Processing
    // =========================================================================
    
    private fun interpolateSignals(signals: Signals): Signals {
        interpolateArray(signals.interAnkleDist, maxInterpGap)
        interpolateArray(signals.kneeAngleLeft, maxInterpGap)
        interpolateArray(signals.kneeAngleRight, maxInterpGap)
        interpolateArray(signals.trunkAngle, maxInterpGap)
        interpolateArray(signals.ankleLeftX, maxInterpGap)
        interpolateArray(signals.ankleRightX, maxInterpGap)
        interpolateArray(signals.ankleLeftY, maxInterpGap)
        interpolateArray(signals.ankleRightY, maxInterpGap)
        interpolateArray(signals.hipLeftY, maxInterpGap)
        interpolateArray(signals.hipRightY, maxInterpGap)
        return signals
    }
    
    private fun interpolateArray(arr: FloatArray, maxGap: Int) {
        var inGap = false
        var gapStart = 0
        
        for (i in arr.indices) {
            if (arr[i].isNaN()) {
                if (!inGap) {
                    inGap = true
                    gapStart = i
                }
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
    
    private fun smoothSignals(signals: Signals): Signals {
        emaSmoothGapAware(signals.interAnkleDist, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeAngleLeft, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeAngleRight, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.trunkAngle, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.ankleLeftY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.ankleRightY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.hipLeftY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.hipRightY, emaAlpha, maxInterpGap)
        return signals
    }
    
    /**
     * Gap-aware EMA smoothing - doesn't bridge gaps longer than maxBridgeGap.
     */
    private fun emaSmoothGapAware(arr: FloatArray, alpha: Float, maxBridgeGap: Int) {
        var firstValid: Int? = null
        for (i in arr.indices) {
            if (!arr[i].isNaN()) {
                firstValid = i
                break
            }
        }
        if (firstValid == null) return
        
        var prev = arr[firstValid]
        var nanRunLength = 0
        
        for (i in (firstValid + 1) until arr.size) {
            if (arr[i].isNaN()) {
                nanRunLength++
                if (nanRunLength <= maxBridgeGap) {
                    arr[i] = prev
                }
            } else {
                if (nanRunLength > maxBridgeGap) {
                    prev = arr[i]
                } else {
                    arr[i] = alpha * arr[i] + (1 - alpha) * prev
                    prev = arr[i]
                }
                nanRunLength = 0
            }
        }
    }
    
    private fun computeVelocities(signals: Signals, fps: Float): Signals {
        val dt = 1f / fps
        
        for (i in 1 until signals.ankleLeftY.size - 1) {
            if (!signals.ankleLeftY[i-1].isNaN() && !signals.ankleLeftY[i+1].isNaN()) {
                signals.ankleLeftVy[i] = (signals.ankleLeftY[i+1] - signals.ankleLeftY[i-1]) / (2 * dt)
            }
            if (!signals.ankleRightY[i-1].isNaN() && !signals.ankleRightY[i+1].isNaN()) {
                signals.ankleRightVy[i] = (signals.ankleRightY[i+1] - signals.ankleRightY[i-1]) / (2 * dt)
            }
            
            val hipAvgPrev = (signals.hipLeftY[i-1] + signals.hipRightY[i-1]) / 2f
            val hipAvgNext = (signals.hipLeftY[i+1] + signals.hipRightY[i+1]) / 2f
            if (!hipAvgPrev.isNaN() && !hipAvgNext.isNaN()) {
                signals.hipAvgVy[i] = (hipAvgNext - hipAvgPrev) / (2 * dt)
            }
        }
        
        return signals
    }
    
    // =========================================================================
    // Multi-Mode Step Detection
    // =========================================================================
    
    private fun determineNearLeg(poseSeq: PoseSequence): String {
        var leftConfSum = 0f
        var rightConfSum = 0f
        
        for (frame in poseSeq.frames) {
            val leftConf = (frame.confidences[MediaPipePoseBackend.LEFT_HIP] +
                    frame.confidences[MediaPipePoseBackend.LEFT_KNEE] +
                    frame.confidences[MediaPipePoseBackend.LEFT_ANKLE]) / 3f
            val rightConf = (frame.confidences[MediaPipePoseBackend.RIGHT_HIP] +
                    frame.confidences[MediaPipePoseBackend.RIGHT_KNEE] +
                    frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE]) / 3f
            leftConfSum += leftConf
            rightConfSum += rightConf
        }
        
        return if (leftConfSum >= rightConfSum) "left" else "right"
    }
    
    private fun computeStepSignal(
        signals: Signals,
        poseSeq: PoseSequence,
        mode: StepSignalMode
    ): FloatArray {
        val n = poseSeq.numFramesTotal
        val minConf = 0.15f
        
        return when (mode) {
            StepSignalMode.INTER_ANKLE -> {
                signals.interAnkleDist.copyOf()
            }
            
            StepSignalMode.MAX_ANKLE_VY -> {
                val ankleLeftY = FloatArray(n) { Float.NaN }
                val ankleRightY = FloatArray(n) { Float.NaN }
                
                for (frame in poseSeq.frames) {
                    if (frame.confidences[MediaPipePoseBackend.LEFT_ANKLE] >= minConf) {
                        ankleLeftY[frame.frameIdx] = frame.keypoints[MediaPipePoseBackend.LEFT_ANKLE][1]
                    }
                    if (frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE] >= minConf) {
                        ankleRightY[frame.frameIdx] = frame.keypoints[MediaPipePoseBackend.RIGHT_ANKLE][1]
                    }
                }
                
                smoothSignalSegmentAware(ankleLeftY, 5)
                smoothSignalSegmentAware(ankleRightY, 5)
                
                val dt = 1f / poseSeq.fps
                val vyLeft = FloatArray(n) { Float.NaN }
                val vyRight = FloatArray(n) { Float.NaN }
                
                for (i in 1 until n) {
                    if (!ankleLeftY[i].isNaN() && !ankleLeftY[i-1].isNaN()) {
                        vyLeft[i] = -(ankleLeftY[i] - ankleLeftY[i-1]) / dt
                    }
                    if (!ankleRightY[i].isNaN() && !ankleRightY[i-1].isNaN()) {
                        vyRight[i] = -(ankleRightY[i] - ankleRightY[i-1]) / dt
                    }
                }
                
                FloatArray(n) { i ->
                    when {
                        vyLeft[i].isNaN() && vyRight[i].isNaN() -> Float.NaN
                        vyLeft[i].isNaN() -> vyRight[i]
                        vyRight[i].isNaN() -> vyLeft[i]
                        else -> maxOf(vyLeft[i], vyRight[i])
                    }
                }
            }
            
            StepSignalMode.MIN_KNEE_ANGLE -> {
                val kneeLeft = FloatArray(n) { Float.NaN }
                val kneeRight = FloatArray(n) { Float.NaN }
                
                for (frame in poseSeq.frames) {
                    val kp = frame.keypoints
                    
                    val leftConf = minOf(
                        frame.confidences[MediaPipePoseBackend.LEFT_HIP],
                        frame.confidences[MediaPipePoseBackend.LEFT_KNEE],
                        frame.confidences[MediaPipePoseBackend.LEFT_ANKLE]
                    )
                    if (leftConf >= minConf) {
                        kneeLeft[frame.frameIdx] = computeAngle(
                            kp[MediaPipePoseBackend.LEFT_HIP],
                            kp[MediaPipePoseBackend.LEFT_KNEE],
                            kp[MediaPipePoseBackend.LEFT_ANKLE]
                        )
                    }
                    
                    val rightConf = minOf(
                        frame.confidences[MediaPipePoseBackend.RIGHT_HIP],
                        frame.confidences[MediaPipePoseBackend.RIGHT_KNEE],
                        frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE]
                    )
                    if (rightConf >= minConf) {
                        kneeRight[frame.frameIdx] = computeAngle(
                            kp[MediaPipePoseBackend.RIGHT_HIP],
                            kp[MediaPipePoseBackend.RIGHT_KNEE],
                            kp[MediaPipePoseBackend.RIGHT_ANKLE]
                        )
                    }
                }
                
                smoothSignalSegmentAware(kneeLeft, 5)
                smoothSignalSegmentAware(kneeRight, 5)
                
                // Min knee angle, inverted so peaks = max flexion
                FloatArray(n) { i ->
                    when {
                        kneeLeft[i].isNaN() && kneeRight[i].isNaN() -> Float.NaN
                        kneeLeft[i].isNaN() -> 180f - kneeRight[i]
                        kneeRight[i].isNaN() -> 180f - kneeLeft[i]
                        else -> 180f - minOf(kneeLeft[i], kneeRight[i])
                    }
                }
            }
        }
    }
    
    private fun smoothSignalSegmentAware(signal: FloatArray, maxGap: Int, alpha: Float = 0.3f) {
        var i = 0
        while (i < signal.size) {
            while (i < signal.size && signal[i].isNaN()) i++
            if (i >= signal.size) break
            
            val start = i
            var gapCount = 0
            while (i < signal.size) {
                if (!signal[i].isNaN()) {
                    gapCount = 0
                    i++
                } else if (gapCount < maxGap) {
                    gapCount++
                    i++
                } else {
                    break
                }
            }
            val end = i - gapCount
            
            if (end - start >= 3) {
                var prev = Float.NaN
                for (j in start until end) {
                    if (!signal[j].isNaN()) {
                        if (prev.isNaN()) {
                            prev = signal[j]
                        } else {
                            signal[j] = alpha * signal[j] + (1 - alpha) * prev
                            prev = signal[j]
                        }
                    }
                }
            }
        }
    }
    
    private fun computeConfidenceCoverage(poseSeq: PoseSequence, mode: StepSignalMode): Float {
        val minConf = 0.15f
        val minConfBoth = 0.3f
        var validCount = 0
        
        for (frame in poseSeq.frames) {
            val isValid = when (mode) {
                StepSignalMode.INTER_ANKLE -> {
                    minOf(frame.confidences[MediaPipePoseBackend.LEFT_ANKLE],
                          frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE]) >= minConfBoth
                }
                StepSignalMode.MAX_ANKLE_VY -> {
                    frame.confidences[MediaPipePoseBackend.LEFT_ANKLE] >= minConf ||
                    frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE] >= minConf
                }
                StepSignalMode.MIN_KNEE_ANGLE -> {
                    val leftConf = minOf(
                        frame.confidences[MediaPipePoseBackend.LEFT_HIP],
                        frame.confidences[MediaPipePoseBackend.LEFT_KNEE],
                        frame.confidences[MediaPipePoseBackend.LEFT_ANKLE]
                    )
                    val rightConf = minOf(
                        frame.confidences[MediaPipePoseBackend.RIGHT_HIP],
                        frame.confidences[MediaPipePoseBackend.RIGHT_KNEE],
                        frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE]
                    )
                    leftConf >= minConf || rightConf >= minConf
                }
            }
            if (isValid) validCount++
        }
        
        return if (poseSeq.numFramesTotal > 0) validCount.toFloat() / poseSeq.numFramesTotal else 0f
    }
    
    private fun computePeriodicityScore(signal: FloatArray, fps: Float): Float {
        val validMask = signal.map { !it.isNaN() }.toBooleanArray()
        val validCount = validMask.count { it }
        if (validCount < 20) return 0f
        
        val indices = signal.indices.toList()
        val validIndices = indices.filter { validMask[it] }
        val validValues = validIndices.map { signal[it] }
        
        val signalInterp = FloatArray(signal.size) { i ->
            if (validMask[i]) signal[i]
            else {
                val lower = validIndices.lastOrNull { it < i }
                val upper = validIndices.firstOrNull { it > i }
                when {
                    lower == null -> validValues.first()
                    upper == null -> validValues.last()
                    else -> {
                        val t = (i - lower).toFloat() / (upper - lower)
                        signal[lower] + t * (signal[upper] - signal[lower])
                    }
                }
            }
        }
        
        val mean = signalInterp.average().toFloat()
        val centered = FloatArray(signalInterp.size) { signalInterp[it] - mean }
        val std = sqrt(centered.map { it * it }.average().toFloat())
        if (std < 1e-6f) return 0f
        
        // Autocorrelation
        val n = centered.size
        val acf = FloatArray(n)
        for (lag in 0 until n) {
            var sum = 0f
            for (i in 0 until n - lag) {
                sum += centered[i] * centered[i + lag]
            }
            acf[lag] = sum / (acf[0] + 1e-8f)
        }
        acf[0] = 1f
        
        val minLag = (fps * minStepTimeS).toInt()
        val maxLag = minOf((fps * maxStepTimeS).toInt(), n - 1)
        if (minLag >= maxLag) return 0f
        
        val search = acf.slice(minLag..maxLag)
        val peaks = findPeaks(search.toFloatArray(), 1, 0.05f)
        
        return if (peaks.isNotEmpty()) {
            search[peaks[0]].coerceIn(0f, 1f)
        } else {
            (search.maxOrNull() ?: 0f).coerceIn(0f, 1f) * 0.5f
        }
    }
    
    private fun evaluateStepSignalQuality(
        signal: FloatArray,
        poseSeq: PoseSequence,
        mode: StepSignalMode,
        fps: Float
    ): StepSignalQuality {
        val validPct = signal.count { !it.isNaN() }.toFloat() / signal.size
        if (validPct < 0.3f) {
            return StepSignalQuality(mode, 0, 0f, 0f, 0f, "insufficient_valid_data")
        }
        
        val signalClean = signal.map { if (it.isNaN()) 0f else it }.toFloatArray()
        val stepFrames = estimateStepPeriod(signalClean, fps)
        val minDistance = maxOf((stepFrames * stepDistanceFactor).toInt(), 5)
        
        val validVals = signalClean.filter { it != 0f }
        val minProminence = if (validVals.isNotEmpty()) {
            validVals.std() * stepProminenceFactor
        } else 0.01f
        
        val peaks = findPeaks(signalClean, minDistance, minProminence)
        val peakCount = peaks.size
        
        val periodicity = computePeriodicityScore(signal, fps)
        val coverage = computeConfidenceCoverage(poseSeq, mode)
        
        val rejectionReason = if (peakCount < 4) "insufficient_peaks_$peakCount" else ""
        
        var finalScore = (
            0.3f * minOf(peakCount / 6f, 1f) +
            0.4f * periodicity +
            0.3f * coverage
        )
        
        if (rejectionReason.isNotEmpty()) {
            finalScore *= 0.1f
        }
        
        return StepSignalQuality(mode, peakCount, periodicity, coverage, finalScore, rejectionReason)
    }
    
    private fun selectStepSignalMode(
        signals: Signals,
        poseSeq: PoseSequence,
        nearLeg: String,
        fps: Float
    ): Triple<StepSignalMode, FloatArray, Map<String, StepSignalQuality>> {
        val modes = StepSignalMode.values()
        val candidates = mutableMapOf<StepSignalMode, Pair<FloatArray, StepSignalQuality>>()
        val modeScores = mutableMapOf<String, StepSignalQuality>()
        
        for (mode in modes) {
            val signal = computeStepSignal(signals, poseSeq, mode)
            val quality = evaluateStepSignalQuality(signal, poseSeq, mode, fps)
            candidates[mode] = Pair(signal, quality)
            modeScores[mode.value] = quality
        }
        
        val bestMode = candidates.maxByOrNull { it.value.second.finalScore }!!.key
        val bestQuality = candidates[bestMode]!!.second
        
        val selected = if (bestQuality.peakCount >= 4 && bestQuality.finalScore >= 0.2f) {
            bestMode
        } else {
            candidates.maxByOrNull { it.value.second.peakCount }!!.key
        }
        
        Log.d(TAG, "Step signal: ${selected.value} selected (score=${candidates[selected]!!.second.finalScore})")
        
        return Triple(selected, candidates[selected]!!.first, modeScores)
    }
    
    private fun detectStepsFromSignal(stepSignal: FloatArray, fps: Float): List<StepEvent> {
        val validPct = stepSignal.count { !it.isNaN() }.toFloat() / stepSignal.size
        if (validPct < 0.3f) return emptyList()
        
        val signalClean = stepSignal.map { if (it.isNaN()) 0f else it }.toFloatArray()
        val stepFrames = estimateStepPeriod(signalClean, fps)
        val minDistance = maxOf((stepFrames * stepDistanceFactor).toInt(), 5)
        
        val validVals = signalClean.filter { it != 0f }
        val minProminence = if (validVals.isNotEmpty()) {
            validVals.std() * stepProminenceFactor
        } else 0.01f
        
        val peaks = findPeaks(signalClean, minDistance, minProminence)
        
        return peaks.map { idx ->
            StepEvent(frameIdx = idx, timeS = idx.toFloat() / fps)
        }
    }
    
    private fun estimateStepPeriod(signal: FloatArray, fps: Float): Float {
        val mean = signal.average().toFloat()
        val centered = signal.map { it - mean }.toFloatArray()
        val std = sqrt(centered.map { it * it }.average().toFloat())
        
        if (std < 1e-6f) return fps * 0.5f
        
        val n = centered.size
        val acf = FloatArray(n)
        for (lag in 0 until n) {
            var sum = 0f
            for (i in 0 until n - lag) {
                sum += centered[i] * centered[i + lag]
            }
            acf[lag] = sum
        }
        val acf0 = acf[0] + 1e-8f
        for (i in acf.indices) acf[i] /= acf0
        
        val minLag = (fps * minStepTimeS).toInt()
        val maxLag = minOf((fps * maxStepTimeS).toInt(), n - 1)
        if (minLag >= maxLag) return fps * 0.5f
        
        val search = acf.slice(minLag..maxLag)
        val peaks = findPeaks(search.toFloatArray(), 1, 0.1f)
        
        return if (peaks.isNotEmpty()) {
            (minLag + peaks[0]).toFloat()
        } else {
            (minLag + search.indices.maxByOrNull { search[it] }!!).toFloat()
        }
    }
    
    private fun findPeaks(signal: FloatArray, minDistance: Int, minProminence: Float): List<Int> {
        val peaks = mutableListOf<Int>()
        
        for (i in 1 until signal.size - 1) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                val leftMin = (maxOf(0, i - minDistance) until i).minOfOrNull { signal[it] } ?: signal[i]
                val rightMin = ((i + 1)..minOf(signal.size - 1, i + minDistance)).minOfOrNull { signal[it] } ?: signal[i]
                val prominence = signal[i] - maxOf(leftMin, rightMin)
                
                if (prominence >= minProminence) {
                    if (peaks.isEmpty() || i - peaks.last() >= minDistance) {
                        peaks.add(i)
                    }
                }
            }
        }
        
        return peaks
    }
    
    // =========================================================================
    // Stride Segmentation and Validation
    // =========================================================================
    
    private fun segmentStrides(steps: List<StepEvent>, signals: Signals, fps: Float): List<Stride> {
        val strides = mutableListOf<Stride>()
        
        var i = 0
        while (i < steps.size - 1) {
            val step1 = steps[i]
            val step2 = steps[i + 1]
            
            val (endFrame, endTime) = if (i + 2 < steps.size) {
                Pair(steps[i + 2].frameIdx, steps[i + 2].timeS)
            } else {
                val stepDur = step2.frameIdx - step1.frameIdx
                Pair(step2.frameIdx + stepDur, step2.timeS + stepDur / fps)
            }
            
            strides.add(Stride(
                startFrame = step1.frameIdx,
                endFrame = minOf(endFrame, signals.timestamps.size - 1),
                startTimeS = step1.timeS,
                endTimeS = endTime,
                step1Frame = step1.frameIdx,
                step2Frame = step2.frameIdx,
                step1TimeS = step1.timeS,
                step2TimeS = step2.timeS
            ))
            
            i += 2
        }
        
        return strides
    }
    
    private fun validateStrides(strides: List<Stride>, signals: Signals, fps: Float): List<Stride> {
        if (strides.isEmpty()) return strides
        
        val stepTimes = strides.map { it.step2TimeS - it.step1TimeS }
        val globalStepTime = stepTimes.median()
        
        return strides.map { stride ->
            val start = stride.startFrame
            val end = minOf(stride.endFrame, signals.isValid.size - 1)
            
            if (start >= end) {
                return@map stride.copy(isValid = false, invalidReason = "invalid_frame_range", qualityScore = 0f)
            }
            
            // Check 1: Valid frame percentage
            val validFrames = (start..end).count { signals.isValid[it] }
            val totalFrames = end - start + 1
            val validPct = validFrames.toFloat() / totalFrames
            
            if (validPct < validFramePct) {
                return@map stride.copy(
                    isValid = false,
                    invalidReason = "low_valid_frames_${(validPct * 100).toInt()}%",
                    validFramePct = validPct,
                    qualityScore = 0f
                )
            }
            
            // Check 2: Step time consistency
            val stepTime = stride.step2TimeS - stride.step1TimeS
            val stepTimeDev = abs(stepTime - globalStepTime) / (globalStepTime + 1e-8f)
            
            if (stepTimeDev > stepTimeTolerance) {
                return@map stride.copy(
                    isValid = false,
                    invalidReason = "inconsistent_step_time",
                    validFramePct = validPct,
                    qualityScore = 0f
                )
            }
            
            // Check 3: Knee ROM
            val kneeLeftSlice = signals.kneeAngleLeft.slice(start..end).filter { !it.isNaN() }
            val kneeRightSlice = signals.kneeAngleRight.slice(start..end).filter { !it.isNaN() }
            
            if (kneeLeftSlice.size < 5 || kneeRightSlice.size < 5) {
                return@map stride.copy(
                    isValid = false,
                    invalidReason = "insufficient_knee_data",
                    validFramePct = validPct,
                    qualityScore = 0f
                )
            }
            
            val romLeft = computeRom(kneeLeftSlice)
            val romRight = computeRom(kneeRightSlice)
            val maxRom = maxOf(romLeft, romRight)
            
            if (maxRom < kneeRomMin || maxRom > kneeRomMax) {
                return@map stride.copy(
                    isValid = false,
                    invalidReason = "abnormal_knee_rom_${maxRom.toInt()}deg",
                    validFramePct = validPct,
                    kneeRomLeft = romLeft,
                    kneeRomRight = romRight,
                    qualityScore = 0f
                )
            }
            
            val kneeMaxLeft = computeMaxAngle(kneeLeftSlice)
            val kneeMaxRight = computeMaxAngle(kneeRightSlice)
            
            // Compute quality score
            val romMargin = minOf(
                (maxRom - kneeRomMin) / (kneeRomMax - kneeRomMin + 1e-8f),
                (kneeRomMax - maxRom) / (kneeRomMax - kneeRomMin + 1e-8f)
            ).coerceIn(0f, 1f) * 2f
            
            val timingScore = maxOf(0f, 1f - stepTimeDev / stepTimeTolerance)
            
            val interAnkleSegment = signals.interAnkleDist.slice(start..end).filter { !it.isNaN() }
            val signalQuality = computeSignalQualityScore(interAnkleSegment)
            
            val qualityScore = (
                0.20f * validPct +
                0.20f * timingScore +
                0.20f * romMargin.coerceIn(0f, 1f) +
                0.40f * signalQuality
            )
            
            stride.copy(
                isValid = true,
                validFramePct = validPct,
                kneeRomLeft = romLeft,
                kneeRomRight = romRight,
                kneeMaxLeft = kneeMaxLeft,
                kneeMaxRight = kneeMaxRight,
                qualityScore = qualityScore
            )
        }
    }
    
    private fun computeSignalQualityScore(segment: List<Float>): Float {
        if (segment.size < 5) return 0f
        
        val peakVal = segment.maxOrNull() ?: return 0f
        val troughVal = segment.minOrNull() ?: return 0f
        val signalRange = peakVal - troughVal
        
        val amplitudeScore = when {
            signalRange < 0.05f -> 0.1f
            signalRange < 0.08f -> 0.3f
            signalRange < 0.10f -> 0.6f
            signalRange < 0.15f -> 1.0f
            else -> 0.9f
        }
        
        val peakScore = when {
            peakVal < 0.06f -> 0.2f
            peakVal < 0.10f -> 0.5f
            peakVal < 0.15f -> 1.0f
            else -> 0.9f
        }
        
        val smoothnessScore = if (segment.size > 2 && signalRange > 0.01f) {
            val diffs = (1 until segment.size).map { segment[it] - segment[it - 1] }
            val jitterRatio = diffs.std() / signalRange
            when {
                jitterRatio < 0.12f -> 1.0f
                jitterRatio < 0.20f -> 0.8f
                jitterRatio < 0.30f -> 0.5f
                else -> 0.2f
            }
        } else 0.3f
        
        return 0.45f * amplitudeScore + 0.30f * peakScore + 0.25f * smoothnessScore
    }
    
    private fun computeRom(values: List<Float>): Float {
        if (values.size < 2) return 0f
        
        return if (useRobustExtrema && values.size >= 10) {
            val sorted = values.sorted()
            val loIdx = (values.size * extremaPercentileLo / 100).toInt().coerceIn(0, values.size - 1)
            val hiIdx = (values.size * extremaPercentileHi / 100).toInt().coerceIn(0, values.size - 1)
            sorted[hiIdx] - sorted[loIdx]
        } else {
            (values.maxOrNull() ?: 0f) - (values.minOrNull() ?: 0f)
        }
    }
    
    private fun computeMaxAngle(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        
        return if (useRobustExtrema && values.size >= 10) {
            val sorted = values.sorted()
            val hiIdx = (values.size * extremaPercentileHi / 100).toInt().coerceIn(0, values.size - 1)
            sorted[hiIdx]
        } else {
            values.maxOrNull() ?: 0f
        }
    }
    
    // =========================================================================
    // Quality-Based Stride Selection
    // =========================================================================
    
    private fun select2InnerCycles(strides: List<Stride>): Triple<List<Stride>, String, List<Int>> {
        val validWithIdx = strides.mapIndexedNotNull { idx, s ->
            if (s.isValid) Pair(idx, s) else null
        }
        
        if (validWithIdx.size < 2) return Triple(emptyList(), "", emptyList())
        
        // Find consecutive runs of valid strides
        val runs = mutableListOf<List<Pair<Int, Stride>>>()
        var currentRun = mutableListOf<Pair<Int, Stride>>()
        
        for ((idx, stride) in strides.withIndex()) {
            if (stride.isValid) {
                currentRun.add(Pair(idx, stride))
            } else {
                if (currentRun.isNotEmpty()) {
                    runs.add(currentRun.toList())
                    currentRun = mutableListOf()
                }
            }
        }
        if (currentRun.isNotEmpty()) runs.add(currentRun.toList())
        
        // Find best consecutive pair
        var bestPair: List<Stride>? = null
        var bestPairScore = -1f
        var bestPairIndices = listOf<Int>()
        var bestPairReason = ""
        
        for (run in runs) {
            if (run.size >= 2) {
                for (j in 0 until run.size - 1) {
                    val (idx1, s1) = run[j]
                    val (idx2, s2) = run[j + 1]
                    val pairScore = s1.qualityScore + s2.qualityScore
                    
                    if (pairScore > bestPairScore) {
                        bestPairScore = pairScore
                        bestPair = listOf(s1, s2)
                        bestPairIndices = listOf(idx1, idx2)
                        bestPairReason = "best_consecutive_pair"
                    }
                }
            }
        }
        
        if (bestPair != null) {
            return Triple(bestPair, bestPairReason, bestPairIndices)
        }
        
        // Fallback: best non-consecutive pair by quality
        if (validWithIdx.size >= 2) {
            val sorted = validWithIdx.sortedByDescending { it.second.qualityScore }
            val (idx1, s1) = sorted[0]
            val (idx2, s2) = sorted[1]
            
            val orderedPair = if (idx1 < idx2) listOf(s1, s2) else listOf(s2, s1)
            val orderedIndices = if (idx1 < idx2) listOf(idx1, idx2) else listOf(idx2, idx1)
            
            return Triple(orderedPair, "fallback_nonconsecutive_pair", orderedIndices)
        }
        
        return Triple(emptyList(), "", emptyList())
    }
    
    // =========================================================================
    // Feature Computation
    // =========================================================================
    
    private fun computeFeatures(
        signals: Signals,
        validStrides: List<Stride>,
        poseSeq: PoseSequence
    ): Triple<GaitFeatures, String, List<Int>> {
        
        val (selectedStrides, selectionReason, selectedIndices) = select2InnerCycles(validStrides)
        
        if (selectedStrides.size < 2) {
            return Triple(GaitFeatures.empty(), "", emptyList())
        }
        
        // Temporal features
        val allStepTimes = mutableListOf<Float>()
        for (s in selectedStrides) {
            allStepTimes.add(s.step2TimeS - s.startTimeS)
            allStepTimes.add(s.endTimeS - s.step2TimeS)
        }
        
        val meanStepTime = allStepTimes.average().toFloat()
        val cadenceSpm = 60f / meanStepTime
        
        val strideTimes = selectedStrides.map { it.endTimeS - it.startTimeS }
        val strideTimeS = strideTimes.average().toFloat()
        val strideTimeCv = if (strideTimes.size > 1) strideTimes.std() / strideTimeS else 0f
        
        val legAStepTimes = allStepTimes.filterIndexed { idx, _ -> idx % 2 == 0 }
        val legBStepTimes = allStepTimes.filterIndexed { idx, _ -> idx % 2 == 1 }
        val stepTimeAsymmetry = asymmetryIndex(
            legAStepTimes.average().toFloat(),
            legBStepTimes.average().toFloat()
        )
        
        // Stride length (displacement / body width)
        val strideLengths = mutableListOf<Float>()
        for (s in selectedStrides) {
            val startX = signals.ankleLeftX[s.startFrame]
            val endX = signals.ankleLeftX[minOf(s.endFrame, signals.ankleLeftX.size - 1)]
            if (!startX.isNaN() && !endX.isNaN()) {
                strideLengths.add(abs(endX - startX))
            }
        }
        val bodyWidth = computeBodyWidth(poseSeq)
        val strideLengthNorm = if (strideLengths.isNotEmpty()) {
            strideLengths.average().toFloat() / (bodyWidth + 1e-8f)
        } else 0f
        
        // Step length asymmetry
        val ankleALengths = mutableListOf<Float>()
        val ankleBLengths = mutableListOf<Float>()
        for (s in selectedStrides) {
            val aStart = signals.ankleLeftX[s.startFrame]
            val aEnd = signals.ankleLeftX[minOf(s.endFrame, signals.ankleLeftX.size - 1)]
            val bStart = signals.ankleRightX[s.startFrame]
            val bEnd = signals.ankleRightX[minOf(s.endFrame, signals.ankleRightX.size - 1)]
            
            if (!aStart.isNaN() && !aEnd.isNaN()) ankleALengths.add(abs(aEnd - aStart))
            if (!bStart.isNaN() && !bEnd.isNaN()) ankleBLengths.add(abs(bEnd - bStart))
        }
        val meanAnkleA = if (ankleALengths.isNotEmpty()) ankleALengths.average().toFloat() else 0f
        val meanAnkleB = if (ankleBLengths.isNotEmpty()) ankleBLengths.average().toFloat() else 0f
        val stepLengthAsymmetry = asymmetryIndex(meanAnkleA, meanAnkleB)
        
        // Knee ROM and max (from stride validation)
        val kneeLeftRom = selectedStrides.map { it.kneeRomLeft }.average().toFloat()
        val kneeRightRom = selectedStrides.map { it.kneeRomRight }.average().toFloat()
        val kneeLeftMax = selectedStrides.map { it.kneeMaxLeft }.average().toFloat()
        val kneeRightMax = selectedStrides.map { it.kneeMaxRight }.average().toFloat()
        
        // Stride amplitude (max inter-ankle in stride / body width)
        val maxInterAnkleValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.interAnkleDist.slice(s.startFrame..minOf(s.endFrame, signals.interAnkleDist.size - 1))
                .filter { !it.isNaN() }
            if (segment.isNotEmpty()) {
                maxInterAnkleValues.add(segment.maxOrNull() ?: 0f)
            }
        }
        val strideAmpNorm = if (maxInterAnkleValues.isNotEmpty()) {
            maxInterAnkleValues.average().toFloat() / (bodyWidth + 1e-8f)
        } else 0f
        
        // Inter-ankle CV (within stride windows)
        val interAnkleValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.interAnkleDist.slice(s.startFrame..minOf(s.endFrame, signals.interAnkleDist.size - 1))
                .filter { !it.isNaN() }
            interAnkleValues.addAll(segment)
        }
        val interAnkleCv = if (interAnkleValues.isNotEmpty() && interAnkleValues.average() > 0) {
            interAnkleValues.std() / interAnkleValues.average().toFloat()
        } else 0f
        
        // LDJ (within stride windows)
        val ldjKneeLeftValues = mutableListOf<Float>()
        val ldjKneeRightValues = mutableListOf<Float>()
        val ldjHipValues = mutableListOf<Float>()
        
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.kneeAngleLeft.size - 1)
            
            val ldjL = computeLDJ(signals.kneeAngleLeft.slice(start..end), poseSeq.fps)
            val ldjR = computeLDJ(signals.kneeAngleRight.slice(start..end), poseSeq.fps)
            
            if (!ldjL.isNaN() && ldjL > 0) ldjKneeLeftValues.add(ldjL)
            if (!ldjR.isNaN() && ldjR > 0) ldjKneeRightValues.add(ldjR)
            
            val trunkSlice = signals.trunkAngle.slice(start..end)
            val ldjH = computeLDJ(trunkSlice, poseSeq.fps)
            if (!ldjH.isNaN() && ldjH > 0) ldjHipValues.add(ldjH)
        }
        
        val ldjKneeLeft = if (ldjKneeLeftValues.isNotEmpty()) ldjKneeLeftValues.average().toFloat() else 0f
        val ldjKneeRight = if (ldjKneeRightValues.isNotEmpty()) ldjKneeRightValues.average().toFloat() else 0f
        val ldjHip = if (ldjHipValues.isNotEmpty()) ldjHipValues.average().toFloat() else 0f
        
        // Trunk lean std (within stride windows)
        val trunkValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.trunkAngle.slice(s.startFrame..minOf(s.endFrame, signals.trunkAngle.size - 1))
                .filter { !it.isNaN() }
            trunkValues.addAll(segment)
        }
        val trunkValuesAbs = trunkValues.map { abs(it) }
        val trunkLeanStdDeg = if (trunkValuesAbs.isNotEmpty()) trunkValuesAbs.std() else 0f
        
        return Triple(
            GaitFeatures(
                cadence_spm = cadenceSpm,
                stride_time_s = strideTimeS,
                stride_time_cv = strideTimeCv,
                step_time_asymmetry = stepTimeAsymmetry,
                stride_length_norm = strideLengthNorm,
                stride_amp_norm = strideAmpNorm,
                step_length_asymmetry = stepLengthAsymmetry,
                knee_left_rom = kneeLeftRom,
                knee_right_rom = kneeRightRom,
                knee_left_max = kneeLeftMax,
                knee_right_max = kneeRightMax,
                ldj_knee_left = ldjKneeLeft,
                ldj_knee_right = ldjKneeRight,
                ldj_hip = ldjHip,
                trunk_lean_std_deg = trunkLeanStdDeg,
                inter_ankle_cv = interAnkleCv,
                valid_stride_count = selectedStrides.size
            ),
            selectionReason,
            selectedIndices
        )
    }
    
    private fun asymmetryIndex(left: Float, right: Float): Float {
        val total = left + right
        return if (total == 0f) 0f else (left - right) / total
    }
    
    private fun computeBodyWidth(poseSeq: PoseSequence): Float {
        val widths = mutableListOf<Float>()
        for (frame in poseSeq.frames) {
            val shoulderWidth = abs(
                frame.keypoints[MediaPipePoseBackend.LEFT_SHOULDER][0] -
                frame.keypoints[MediaPipePoseBackend.RIGHT_SHOULDER][0]
            )
            val hipWidth = abs(
                frame.keypoints[MediaPipePoseBackend.LEFT_HIP][0] -
                frame.keypoints[MediaPipePoseBackend.RIGHT_HIP][0]
            )
            widths.add((shoulderWidth + hipWidth) / 2f)
        }
        return if (widths.isNotEmpty()) widths.average().toFloat() else 0.1f
    }
    
    private fun computeLDJ(signal: List<Float>, fps: Float): Float {
        val valid = signal.filter { !it.isNaN() }
        if (valid.size < 3) return Float.NaN
        
        val dt = 1f / fps
        
        val velocity = (1 until valid.size).map { (valid[it] - valid[it - 1]) / dt }
        if (velocity.size < 2) return Float.NaN
        
        val accel = (1 until velocity.size).map { (velocity[it] - velocity[it - 1]) / dt }
        if (accel.isEmpty()) return Float.NaN
        
        val ldj = sqrt(accel.map { it * it }.average().toFloat())
        return ldj
    }
    
    // =========================================================================
    // Diagnostics
    // =========================================================================
    
    private fun createDiagnostics(
        poseSeq: PoseSequence,
        qualityFlag: QualityFlag,
        reason: String,
        numSteps: Int = 0,
        numValidStrides: Int = 0,
        stepMode: String = "inter_ankle",
        selectionReason: String = ""
    ): GaitDiagnostics {
        return GaitDiagnostics(
            videoId = poseSeq.videoId,
            fpsDetected = poseSeq.fps,
            durationS = poseSeq.durationS,
            numFramesTotal = poseSeq.numFramesTotal,
            numFramesValid = poseSeq.frames.size,
            validFrameRate = poseSeq.detectionRate,
            numStepsDetected = numSteps,
            numStridesValid = numValidStrides,
            estimatedCadenceSpm = if (numSteps > 1 && poseSeq.durationS > 0) {
                numSteps * 60f / poseSeq.durationS
            } else 0f,
            walkingDirection = poseSeq.walkingDirection,
            wasFlipped = poseSeq.wasFlipped,
            qualityFlag = qualityFlag,
            rejectionReasons = if (reason.isNotEmpty()) listOf(reason) else emptyList()
        )
    }
}

// Extension functions
private fun List<Float>.std(): Float {
    if (size < 2) return 0f
    val mean = average().toFloat()
    val variance = sumOf { ((it - mean) * (it - mean)).toDouble() } / (size - 1)
    return sqrt(variance).toFloat()
}

private fun List<Float>.median(): Float {
    if (isEmpty()) return 0f
    val sorted = sorted()
    return if (size % 2 == 0) {
        (sorted[size / 2 - 1] + sorted[size / 2]) / 2f
    } else {
        sorted[size / 2]
    }
}

private fun FloatArray.std(): Float = this.toList().filter { !it.isNaN() }.std()
