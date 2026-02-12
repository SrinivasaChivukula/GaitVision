package GaitVision.com.gait

import android.util.Log
import GaitVision.com.enableVerboseLogging
import GaitVision.com.getLRSwapIndex
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*

/**
 * Feature Extractor for Gait Analysis - Kotlin port of PC pipeline (pc_tasks/feature_extractor.py).
 * 
 * Extracts gait features from pose sequences.
 * Uses inter-ankle distance as the single step detection signal.
 * Quality-based stride selection picks the best two consecutive cycles.
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
        private const val TAG = "GaitDebug"
        
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
     * Extract gait features from pose sequence.
     */
    fun extract(poseSeq: PoseSequence): Pair<GaitFeatures?, GaitDiagnostics> {
        Log.d(TAG, "=== VIDEO INFO ===")
        Log.d(TAG, "  numFramesTotal: ${poseSeq.numFramesTotal}")
        Log.d(TAG, "  detectedFrames: ${poseSeq.frames.size}")
        Log.d(TAG, "  fps: ${poseSeq.fps}")
        Log.d(TAG, "  detectionRate: ${String.format("%.1f", poseSeq.detectionRate * 100)}%")
        Log.d(TAG, "Extracting features from ${poseSeq.frames.size} frames")
        
        // Basic checks
        if (poseSeq.frames.size < 20) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.UNPROCESSABLE, "too_few_frames"))
        }
        
        if (poseSeq.detectionRate < 0.3f) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.LOW_DETECTION, 
                "detection_rate_${(poseSeq.detectionRate * 100).toInt()}%"))
        }
        
        // =====================================================================
        // FULL PIPELINE DEBUG - Compare with PC (controlled by enableVerboseLogging)
        // =====================================================================
        
        // Step 1: Compute signals
        var signals = computeSignals(poseSeq)
        
        // Step 2: Interpolate gaps
        signals = interpolateSignals(signals)
        
        // Step 3: Light smoothing
        signals = smoothSignals(signals)
        
        // Step 4: Compute velocities
        signals = computeVelocities(signals, poseSeq.fps)
        
        // Step 5: Detect steps from inter-ankle distance (single signal, always)
        val steps = detectStepsFromSignal(signals.interAnkleDist, poseSeq.fps)
        Log.d(TAG, "Step detection: inter_ankle (single signal), detected ${steps.size} steps")
        
        // Verbose per-frame logging (expensive - only enable for debugging)
        if (enableVerboseLogging) {
            Log.d(TAG, "")
            Log.d(TAG, "========== VERBOSE DEBUG: RAW LANDMARKS ==========")
            Log.d(TAG, "frame,la_x,la_y,ra_x,ra_y,lk_x,lk_y,rk_x,rk_y,lh_x,lh_y,rh_x,rh_y,la_conf,ra_conf,lk_conf,rk_conf")
            for (frameIdx in 0 until poseSeq.numFramesTotal) {
                val frame = poseSeq.frames.find { it.frameIdx == frameIdx }
                if (frame != null) {
                    val la = frame.keypoints[MediaPipePoseBackend.LEFT_ANKLE]
                    val ra = frame.keypoints[MediaPipePoseBackend.RIGHT_ANKLE]
                    val lk = frame.keypoints[MediaPipePoseBackend.LEFT_KNEE]
                    val rk = frame.keypoints[MediaPipePoseBackend.RIGHT_KNEE]
                    val lh = frame.keypoints[MediaPipePoseBackend.LEFT_HIP]
                    val rh = frame.keypoints[MediaPipePoseBackend.RIGHT_HIP]
                    val laC = frame.confidences[MediaPipePoseBackend.LEFT_ANKLE]
                    val raC = frame.confidences[MediaPipePoseBackend.RIGHT_ANKLE]
                    val lkC = frame.confidences[MediaPipePoseBackend.LEFT_KNEE]
                    val rkC = frame.confidences[MediaPipePoseBackend.RIGHT_KNEE]
                    Log.d(TAG, "$frameIdx,${String.format("%.6f", la[0])},${String.format("%.6f", la[1])},${String.format("%.6f", ra[0])},${String.format("%.6f", ra[1])},${String.format("%.6f", lk[0])},${String.format("%.6f", lk[1])},${String.format("%.6f", rk[0])},${String.format("%.6f", rk[1])},${String.format("%.6f", lh[0])},${String.format("%.6f", lh[1])},${String.format("%.6f", rh[0])},${String.format("%.6f", rh[1])},${String.format("%.3f", laC)},${String.format("%.3f", raC)},${String.format("%.3f", lkC)},${String.format("%.3f", rkC)}")
                } else {
                    Log.d(TAG, "$frameIdx,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,0")
                }
            }
            
            Log.d(TAG, "")
            Log.d(TAG, "========== VERBOSE DEBUG: SIGNALS ==========")
            Log.d(TAG, "frame,inter_ankle,knee_left,knee_right")
            for (i in 0 until signals.timestamps.size) {
                Log.d(TAG, "$i,${String.format("%.6f", signals.interAnkleDist[i])},${String.format("%.6f", signals.kneeAngleLeft[i])},${String.format("%.6f", signals.kneeAngleRight[i])}")
            }
            
            val interAnkle = signals.interAnkleDist
            val validCount = interAnkle.count { !it.isNaN() }
            val validVals = interAnkle.filter { !it.isNaN() }
            if (validVals.isNotEmpty()) {
                Log.d(TAG, "INTER_ANKLE stats: count=$validCount, mean=${String.format("%.4f", validVals.average())}, std=${String.format("%.4f", validVals.std())}, min=${String.format("%.4f", validVals.minOrNull())}, max=${String.format("%.4f", validVals.maxOrNull())}")
            }
            
            Log.d(TAG, "")
            Log.d(TAG, "========== VERBOSE DEBUG: STEP SIGNAL ==========")
            Log.d(TAG, "Mode: inter_ankle (single signal)")
            Log.d(TAG, "frame,value")
            for (i in signals.interAnkleDist.indices) {
                Log.d(TAG, "$i,${String.format("%.6f", signals.interAnkleDist[i])}")
            }
            Log.d(TAG, "DETECTED PEAKS: ${steps.map { it.frameIdx }}")
            Log.d(TAG, "PEAK TIMES: ${steps.map { String.format("%.3f", it.timeS) }}")
        }
        
        if (steps.size < 4) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, 
                "only_${steps.size}_steps", steps.size))
        }
        
        // Step 6: Segment strides
        val strides = segmentStrides(steps, signals, poseSeq.fps)
        
        // Step 7: Soft-score all strides (no hard gates — only degenerate strides filtered)
        val validatedStrides = validateStrides(strides, signals, poseSeq.fps)
        val validStrides = validatedStrides.filter { it.isValid }
        Log.d(TAG, "Scored strides: ${validStrides.size} valid of ${validatedStrides.size}")
        
        if (validStrides.size < 2) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, 
                "only_${validStrides.size}_valid_strides", steps.size, validStrides.size))
        }
        
        // Step 8: Quality-based stride selection and feature computation
        val (features, selectionReason, selectedIndices) = computeFeatures(signals, validStrides, poseSeq)
        
        val qualityFlag = when {
            features.valid_stride_count == 0 -> QualityFlag.NO_CYCLES
            else -> QualityFlag.OK
        }
        
        val diagnostics = createDiagnostics(
            poseSeq, qualityFlag, "",
            steps.size, validStrides.size,
            selectionReason = selectionReason
        )
        
        // Store signals and strides for visualization
        GaitVision.com.extractedSignals = signals
        GaitVision.com.extractedStrides = validatedStrides
        GaitVision.com.selectedStrideIndices = selectedIndices
        GaitVision.com.stepSignalMode = "inter_ankle"
        
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
                val swapIdx = getLRSwapIndex(i)
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
    
    // =========================================================================
    // Signal Computation
    // =========================================================================
    
    private fun computeSignals(poseSeq: PoseSequence): Signals {
        val n = poseSeq.numFramesTotal
        
        val timestamps = FloatArray(n) { Float.NaN }
        val frameIndices = IntArray(n) { it }
        val isValid = BooleanArray(n) { false }
        
        // Core signals (used in features)
        val interAnkleDist = FloatArray(n) { Float.NaN }
        val kneeAngleLeft = FloatArray(n) { Float.NaN }
        val kneeAngleRight = FloatArray(n) { Float.NaN }
        val trunkAngle = FloatArray(n) { Float.NaN }
        
        // Visualization-only angles (for charts)
        val ankleAngleLeft = FloatArray(n) { Float.NaN }
        val ankleAngleRight = FloatArray(n) { Float.NaN }
        val hipAngleLeft = FloatArray(n) { Float.NaN }
        val hipAngleRight = FloatArray(n) { Float.NaN }
        val strideAngle = FloatArray(n) { Float.NaN }
        
        // Positions
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
            
            // Positions
            ankleLeftX[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][0]
            ankleRightX[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][0]
            interAnkleDist[idx] = abs(kp[MediaPipePoseBackend.RIGHT_ANKLE][0] - kp[MediaPipePoseBackend.LEFT_ANKLE][0])
            ankleLeftY[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][1]
            ankleRightY[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][1]
            hipLeftY[idx] = kp[MediaPipePoseBackend.LEFT_HIP][1]
            hipRightY[idx] = kp[MediaPipePoseBackend.RIGHT_HIP][1]
            
            // Knee angles (hip-knee-ankle) - used in features
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
            
            // Trunk angle - used in features
            val midShoulder = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][0] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_SHOULDER][1] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][1]) / 2f
            )
            val midHip = floatArrayOf(
                (kp[MediaPipePoseBackend.LEFT_HIP][0] + kp[MediaPipePoseBackend.RIGHT_HIP][0]) / 2f,
                (kp[MediaPipePoseBackend.LEFT_HIP][1] + kp[MediaPipePoseBackend.RIGHT_HIP][1]) / 2f
            )
            trunkAngle[idx] = computeTrunkLean(midShoulder, midHip)
            
            // Ankle angles (foot-ankle-knee) - visualization only
            ankleAngleLeft[idx] = computeAngle(
                kp[MediaPipePoseBackend.LEFT_FOOT_INDEX],
                kp[MediaPipePoseBackend.LEFT_ANKLE],
                kp[MediaPipePoseBackend.LEFT_KNEE]
            ) - 90f  // Offset like original: 0° = foot flat
            ankleAngleRight[idx] = computeAngle(
                kp[MediaPipePoseBackend.RIGHT_FOOT_INDEX],
                kp[MediaPipePoseBackend.RIGHT_ANKLE],
                kp[MediaPipePoseBackend.RIGHT_KNEE]
            ) - 90f
            
            // Hip angles (knee-hip-shoulder) - visualization only
            hipAngleLeft[idx] = 180f - computeAngle(
                kp[MediaPipePoseBackend.LEFT_KNEE],
                kp[MediaPipePoseBackend.LEFT_HIP],
                kp[MediaPipePoseBackend.LEFT_SHOULDER]
            )
            hipAngleRight[idx] = 180f - computeAngle(
                kp[MediaPipePoseBackend.RIGHT_KNEE],
                kp[MediaPipePoseBackend.RIGHT_HIP],
                kp[MediaPipePoseBackend.RIGHT_SHOULDER]
            )
            
            // Stride angle (left heel - hip center - right heel) - visualization only
            strideAngle[idx] = computeAngle(
                kp[MediaPipePoseBackend.LEFT_HEEL],
                midHip,
                kp[MediaPipePoseBackend.RIGHT_HEEL]
            )
        }
        
        return Signals(
            timestamps = timestamps,
            frameIndices = frameIndices,
            isValid = isValid,
            interAnkleDist = interAnkleDist,
            kneeAngleLeft = kneeAngleLeft,
            kneeAngleRight = kneeAngleRight,
            trunkAngle = trunkAngle,
            ankleAngleLeft = ankleAngleLeft,
            ankleAngleRight = ankleAngleRight,
            hipAngleLeft = hipAngleLeft,
            hipAngleRight = hipAngleRight,
            strideAngle = strideAngle,
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
        val n = signals.ankleLeftY.size
        
        // Use forward difference to match PC: v[i] = (s[i+1] - s[i]) / dt
        for (i in 0 until n - 1) {
            if (!signals.ankleLeftY[i].isNaN() && !signals.ankleLeftY[i+1].isNaN()) {
                signals.ankleLeftVy[i] = (signals.ankleLeftY[i+1] - signals.ankleLeftY[i]) / dt
            }
            if (!signals.ankleRightY[i].isNaN() && !signals.ankleRightY[i+1].isNaN()) {
                signals.ankleRightVy[i] = (signals.ankleRightY[i+1] - signals.ankleRightY[i]) / dt
            }
            
            val hipAvgCurr = (signals.hipLeftY[i] + signals.hipRightY[i]) / 2f
            val hipAvgNext = (signals.hipLeftY[i+1] + signals.hipRightY[i+1]) / 2f
            if (!hipAvgCurr.isNaN() && !hipAvgNext.isNaN()) {
                signals.hipAvgVy[i] = (hipAvgNext - hipAvgCurr) / dt
            }
        }
        
        // Backward difference for last frame (like PC)
        val last = n - 1
        if (last > 0) {
            if (!signals.ankleLeftY[last].isNaN() && !signals.ankleLeftY[last-1].isNaN()) {
                signals.ankleLeftVy[last] = (signals.ankleLeftY[last] - signals.ankleLeftY[last-1]) / dt
            }
            if (!signals.ankleRightY[last].isNaN() && !signals.ankleRightY[last-1].isNaN()) {
                signals.ankleRightVy[last] = (signals.ankleRightY[last] - signals.ankleRightY[last-1]) / dt
            }
            val hipAvgLast = (signals.hipLeftY[last] + signals.hipRightY[last]) / 2f
            val hipAvgPrev = (signals.hipLeftY[last-1] + signals.hipRightY[last-1]) / 2f
            if (!hipAvgLast.isNaN() && !hipAvgPrev.isNaN()) {
                signals.hipAvgVy[last] = (hipAvgLast - hipAvgPrev) / dt
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
        
        // Peak snap: recenter each peak to local maximum within ±2 frames
        // This neutralizes ±1 frame edge-case differences vs PC
        val snappedPeaks = snapPeaksToLocalMax(peaks, signalClean, snapWindow = 2)
        
        
        // Log if any peaks were snapped
        if (peaks != snappedPeaks) {
            Log.d(TAG, "Peak snap applied: $peaks -> $snappedPeaks")
        } else {
            Log.d(TAG, "Peak snap: no changes (peaks already at local max)")
        }
        
        return snappedPeaks.map { idx ->
            StepEvent(frameIdx = idx, timeS = idx.toFloat() / fps)
        }
    }
    
    /**
     * Snap each peak to the true local maximum within ±window frames.
     * Tie-break: leftmost index wins (deterministic, matches typical SciPy behavior).
     * This helps achieve parity with PC when peaks differ by ±1 frame due to
     * floating-point or edge-case differences in prominence calculation.
     */
    private fun snapPeaksToLocalMax(peaks: List<Int>, signal: FloatArray, snapWindow: Int): List<Int> {
        if (peaks.isEmpty()) return peaks
        
        return peaks.map { peakIdx ->
            val start = maxOf(0, peakIdx - snapWindow)
            val end = minOf(signal.size - 1, peakIdx + snapWindow)
            
            // Find the index with maximum value in the window
            // Tie-break: leftmost (first encountered)
            var bestIdx = start
            var bestVal = signal[start]
            for (i in (start + 1)..end) {
                if (signal[i] > bestVal) {  // Strict > means leftmost wins on tie
                    bestVal = signal[i]
                    bestIdx = i
                }
            }
            bestIdx
        }.distinct()  // Remove duplicates if multiple peaks snap to same index
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
        // Step 1: Find all local maxima
        val localMaxima = mutableListOf<Int>()
        for (i in 1 until signal.size - 1) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
                localMaxima.add(i)
            }
        }
        
        if (localMaxima.isEmpty()) return emptyList()
        
        // Step 2: Compute prominence for each peak (scipy-style algorithm)
        // For each peak, extend left/right until hitting a higher peak or edge
        // Then find minimum within that range on each side
        val prominences = FloatArray(localMaxima.size)
        
        for ((idx, peakIdx) in localMaxima.withIndex()) {
            val peakHeight = signal[peakIdx]
            
            // Extend left until hitting higher peak or edge
            var leftBound = peakIdx - 1
            while (leftBound > 0 && signal[leftBound] < peakHeight) {
                leftBound--
            }
            // Find minimum from leftBound to peak
            var leftMin = signal[peakIdx]
            for (j in leftBound until peakIdx) {
                if (signal[j] < leftMin) leftMin = signal[j]
            }
            
            // Extend right until hitting higher peak or edge
            var rightBound = peakIdx + 1
            while (rightBound < signal.size - 1 && signal[rightBound] < peakHeight) {
                rightBound++
            }
            // Find minimum from peak to rightBound
            var rightMin = signal[peakIdx]
            for (j in (peakIdx + 1)..rightBound) {
                if (signal[j] < rightMin) rightMin = signal[j]
            }
            
            // Prominence = peak height - higher of the two bases
            prominences[idx] = peakHeight - maxOf(leftMin, rightMin)
        }
        
        // Step 3: Filter by prominence, then by distance
        val filteredPeaks = mutableListOf<Int>()
        for ((idx, peakIdx) in localMaxima.withIndex()) {
            if (prominences[idx] >= minProminence) {
                if (filteredPeaks.isEmpty() || peakIdx - filteredPeaks.last() >= minDistance) {
                    filteredPeaks.add(peakIdx)
                } else if (prominences[idx] > prominences[localMaxima.indexOf(filteredPeaks.last())]) {
                    // If this peak has higher prominence, replace the last one
                    filteredPeaks[filteredPeaks.size - 1] = peakIdx
                }
            }
        }
        
        return filteredPeaks
    }
    
    // =========================================================================
    // Stride Segmentation and Validation
    // =========================================================================
    
    private fun segmentStrides(steps: List<StepEvent>, signals: Signals, fps: Float): List<Stride> {
        if (steps.size < 3) return emptyList()
        val maxFrame = signals.timestamps.size - 1
        
        // Overlapping candidates: peak[i] → peak[i+2] for every i
        return (0..steps.size - 3).map { i ->
            Stride(
                startFrame = steps[i].frameIdx,
                endFrame = minOf(steps[i + 2].frameIdx, maxFrame),
                startTimeS = steps[i].timeS,
                endTimeS = steps[i + 2].timeS,
                step1Frame = steps[i].frameIdx,
                step2Frame = steps[i + 1].frameIdx,
                step1TimeS = steps[i].timeS,
                step2TimeS = steps[i + 1].timeS
            )
        }
    }
    
    private fun validateStrides(strides: List<Stride>, signals: Signals, fps: Float): List<Stride> {
        if (strides.isEmpty()) return strides
        
        val stepTimes = strides.map { it.step2TimeS - it.step1TimeS }
        val globalStepTime = stepTimes.median()
        
        // Global inter-ankle range for motion_energy scoring
        val allInterAnkle = signals.interAnkleDist.filter { !it.isNaN() }
        val expectedRange = if (allInterAnkle.size >= 2) {
            allInterAnkle.maxOrNull()!! - allInterAnkle.minOrNull()!!
        } else 1f
        
        return strides.map { stride ->
            val start = stride.startFrame
            val end = minOf(stride.endFrame, signals.isValid.size - 1)
            
            // Truly degenerate — hard exit only for these
            if (end - start < 5) {
                return@map stride.copy(isValid = false, invalidReason = "degenerate", qualityScore = 0f)
            }
            
            val validFrames = (start until end).count { signals.isValid[it] }
            val totalFrames = end - start
            val validPct = if (totalFrames > 0) validFrames.toFloat() / totalFrames else 0f
            
            if (validPct == 0f) {
                return@map stride.copy(isValid = false, invalidReason = "degenerate", qualityScore = 0f)
            }
            
            // --- Soft scoring: 5 components, all [0,1], no hard gates ---
            
            val coverageScore = validPct
            
            val stepTimeDev = abs((stride.step2TimeS - stride.step1TimeS) - globalStepTime) / (globalStepTime + 1e-8f)
            val timingScore = maxOf(0f, 1f - stepTimeDev / 0.5f)
            
            val kneeLeftSlice = signals.kneeAngleLeft.slice(start until end).filter { !it.isNaN() }
            val kneeRightSlice = signals.kneeAngleRight.slice(start until end).filter { !it.isNaN() }
            val romLeft = if (kneeLeftSlice.size >= 5) computeRom(kneeLeftSlice) else 0f
            val romRight = if (kneeRightSlice.size >= 5) computeRom(kneeRightSlice) else 0f
            val romScore = sigmoidRomScore(maxOf(romLeft, romRight))
            
            val interAnkleSegment = signals.interAnkleDist.slice(start until end).filter { !it.isNaN() }
            val signalQuality = computeSignalQualityScore(interAnkleSegment)
            
            val segmentRange = if (interAnkleSegment.size >= 2) {
                interAnkleSegment.maxOrNull()!! - interAnkleSegment.minOrNull()!!
            } else 0f
            val motionEnergy = minOf(1f, segmentRange / (expectedRange + 1e-8f))
            
            val qualityScore = (
                0.20f * coverageScore +
                0.20f * timingScore +
                0.15f * romScore +
                0.30f * signalQuality +
                0.15f * motionEnergy
            ).coerceIn(0f, 1f)
            
            Log.d(TAG, "  Stride [${start}..${end}] quality: " +
                "cov=${String.format("%.3f", coverageScore)} tim=${String.format("%.3f", timingScore)} " +
                "rom=${String.format("%.3f", romScore)} sig=${String.format("%.3f", signalQuality)} " +
                "mot=${String.format("%.3f", motionEnergy)} -> ${String.format("%.4f", qualityScore)}")
            
            stride.copy(
                isValid = true,
                validFramePct = validPct,
                kneeRomLeft = romLeft,
                kneeRomRight = romRight,
                kneeMaxLeft = if (kneeLeftSlice.size >= 5) computeMaxAngle(kneeLeftSlice) else 0f,
                kneeMaxRight = if (kneeRightSlice.size >= 5) computeMaxAngle(kneeRightSlice) else 0f,
                qualityScore = qualityScore
            )
        }
    }
    
    /** Sigmoid-shaped ROM score: 1.0 inside [10°,55°], smooth decay outside, never hard-cuts. */
    private fun sigmoidRomScore(rom: Float): Float {
        val lo = 10f; val hi = 55f
        return when {
            rom in lo..hi -> 1f
            rom < lo -> 1f / (1f + ((lo - rom) / 10f).let { it * it })
            else -> 1f / (1f + ((rom - hi) / 10f).let { it * it })
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
            // Use linear interpolation like numpy's np.percentile()
            val pLo = percentile(sorted, extremaPercentileLo)
            val pHi = percentile(sorted, extremaPercentileHi)
            pHi - pLo
        } else {
            (values.maxOrNull() ?: 0f) - (values.minOrNull() ?: 0f)
        }
    }
    
    // Helper to match numpy's percentile with linear interpolation
    private fun percentile(sortedValues: List<Float>, p: Float): Float {
        if (sortedValues.isEmpty()) return 0f
        if (sortedValues.size == 1) return sortedValues[0]
        
        val n = sortedValues.size
        val idx = (n - 1) * p / 100f
        val lower = idx.toInt().coerceIn(0, n - 2)
        val upper = lower + 1
        val fraction = idx - lower
        
        return sortedValues[lower] + fraction * (sortedValues[upper] - sortedValues[lower])
    }
    
    private fun computeMaxAngle(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        
        return if (useRobustExtrema && values.size >= 10) {
            percentile(values.sorted(), extremaPercentileHi)
        } else {
            values.maxOrNull() ?: 0f
        }
    }
    
    // =========================================================================
    // Quality-Based Stride Selection
    // =========================================================================
    
    private fun select2InnerCycles(strides: List<Stride>): Triple<List<Stride>, String, List<Int>> {
        val valid = strides.withIndex().filter { it.value.isValid }
        if (valid.size < 2) return Triple(emptyList(), "", emptyList())
        
        // Find best pair sharing exactly one boundary peak (cycleA.endFrame == cycleB.startFrame)
        var bestScore = -1f
        var bestA = -1
        var bestB = -1
        
        Log.d(TAG, "=== PAIR SELECTION (boundary-sharing) ===")
        for (i in valid.indices) {
            for (j in i + 1 until valid.size) {
                val a = valid[i]; val b = valid[j]
                if (a.value.endFrame != b.value.startFrame) continue  // must share boundary peak
                val score = a.value.qualityScore + b.value.qualityScore
                Log.d(TAG, "  Pair [${a.index}, ${b.index}]: score=${String.format("%.4f", score)} " +
                    "frames [${a.value.startFrame}..${a.value.endFrame}] + [${b.value.startFrame}..${b.value.endFrame}]")
                if (score > bestScore) {
                    bestScore = score; bestA = i; bestB = j
                }
            }
        }
        
        if (bestA < 0) return Triple(emptyList(), "", emptyList())
        
        val a = valid[bestA]; val b = valid[bestB]
        Log.d(TAG, "  Best: [${a.index}, ${b.index}] score=${String.format("%.4f", bestScore)}")
        return Triple(listOf(a.value, b.value), "best_pair", listOf(a.index, b.index))
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
        
        // Log stride selection details
        Log.d(TAG, "=== STRIDE SELECTION ===")
        Log.d(TAG, "Total strides: ${validStrides.size}, Valid: ${validStrides.count { it.isValid }}")
        validStrides.forEachIndexed { idx, s ->
            val marker = if (selectedIndices.contains(idx)) " [SELECTED]" else ""
            Log.d(TAG, "  Stride $idx: valid=${s.isValid}, frames=${s.startFrame}-${s.endFrame}, " +
                "quality=${String.format("%.2f", s.qualityScore)}, " +
                "kneeL=${String.format("%.1f", s.kneeRomLeft)}°, kneeR=${String.format("%.1f", s.kneeRomRight)}°" +
                "${s.invalidReason?.let { ", reason=$it" } ?: ""}$marker")
        }
        Log.d(TAG, "Selection: $selectionReason, indices=$selectedIndices")
        
        if (selectedStrides.size < 2) {
            Log.w(TAG, "Not enough valid strides selected!")
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
        
        // Stride amplitude (max inter-ankle in stride / body width, exclusive range to match PC)
        val maxInterAnkleValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.interAnkleDist.slice(s.startFrame until minOf(s.endFrame, signals.interAnkleDist.size))
                .filter { !it.isNaN() }
            if (segment.isNotEmpty()) {
                maxInterAnkleValues.add(segment.maxOrNull() ?: 0f)
            }
        }
        val strideAmpNorm = if (maxInterAnkleValues.isNotEmpty()) {
            maxInterAnkleValues.average().toFloat() / (bodyWidth + 1e-8f)
        } else 0f
        
        // Inter-ankle CV (within stride windows, exclusive range to match PC)
        val interAnkleValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.interAnkleDist.slice(s.startFrame until minOf(s.endFrame, signals.interAnkleDist.size))
                .filter { !it.isNaN() }
            interAnkleValues.addAll(segment)
        }
        // Use epsilon like PC for numerical stability
        val interAnkleCv = if (interAnkleValues.isNotEmpty()) {
            interAnkleValues.std() / (interAnkleValues.average().toFloat() + 1e-8f)
        } else 0f
        
        // LDJ (within stride windows)
        val ldjKneeLeftValues = mutableListOf<Float>()
        val ldjKneeRightValues = mutableListOf<Float>()
        val ldjHipValues = mutableListOf<Float>()
        
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.kneeAngleLeft.size - 1)
            
            val ldjL = computeLDJ(signals.kneeAngleLeft.slice(start until end), poseSeq.fps)
            val ldjR = computeLDJ(signals.kneeAngleRight.slice(start until end), poseSeq.fps)
            
            if (!ldjL.isNaN() && ldjL > 0) ldjKneeLeftValues.add(ldjL)
            if (!ldjR.isNaN() && ldjR > 0) ldjKneeRightValues.add(ldjR)
            
            val trunkSlice = signals.trunkAngle.slice(start until end)
            val ldjH = computeLDJ(trunkSlice, poseSeq.fps)
            if (!ldjH.isNaN() && ldjH > 0) ldjHipValues.add(ldjH)
        }
        
        val ldjKneeLeft = if (ldjKneeLeftValues.isNotEmpty()) ldjKneeLeftValues.average().toFloat() else 0f
        val ldjKneeRight = if (ldjKneeRightValues.isNotEmpty()) ldjKneeRightValues.average().toFloat() else 0f
        val ldjHip = if (ldjHipValues.isNotEmpty()) ldjHipValues.average().toFloat() else 0f
        
        // Trunk lean std (within stride windows, exclusive range to match PC)
        val trunkValues = mutableListOf<Float>()
        for (s in selectedStrides) {
            val segment = signals.trunkAngle.slice(s.startFrame until minOf(s.endFrame, signals.trunkAngle.size))
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
        if (valid.size < 3) return 0f  // Match PC: return 0, not NaN
        
        val dt = if (fps > 0) 1f / fps else 0.033f  // Add fallback like PC
        
        val velocity = (1 until valid.size).map { (valid[it] - valid[it - 1]) / dt }
        if (velocity.size < 2) return 0f  // Match PC
        
        val accel = (1 until velocity.size).map { (velocity[it] - velocity[it - 1]) / dt }
        if (accel.isEmpty()) return 0f  // Match PC
        
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
    // Use population std (divide by n) to match numpy's default np.std()
    val variance = sumOf { ((it - mean) * (it - mean)).toDouble() } / size
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
