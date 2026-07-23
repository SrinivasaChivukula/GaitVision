package GaitVision.com.gait

import android.util.Log
import GaitVision.com.enableVerboseLogging
import GaitVision.com.AnalysisSession
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
    private val minStepPeakWidthS: Float = GaitConfig.MIN_STEP_PEAK_WIDTH_S,
    private val validFramePct: Float = GaitConfig.VALID_FRAME_PCT,
    private val stepTimeTolerance: Float = GaitConfig.STEP_TIME_TOLERANCE,
    private val kneeRomMin: Float = GaitConfig.KNEE_ROM_MIN,
    private val kneeRomMax: Float = GaitConfig.KNEE_ROM_MAX,
    private val useRobustExtrema: Boolean = GaitConfig.USE_ROBUST_EXTREMA,
    private val extremaPercentileLo: Float = GaitConfig.EXTREMA_PERCENTILE_LO,
    private val extremaPercentileHi: Float = GaitConfig.EXTREMA_PERCENTILE_HI
) {
    companion object {
        private const val TAG = "GaitDebug"
        private const val MAX_INTERVAL_CLEANING_ITERATIONS = 20
        
        // Core keypoints for gait analysis (matches PC pose_backend_tasks.CORE_KEYPOINTS)
        val CORE_KEYPOINTS = intArrayOf(
            MediaPipePoseBackend.LEFT_SHOULDER,
            MediaPipePoseBackend.RIGHT_SHOULDER,
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.RIGHT_ANKLE
        )
    }
    
    // Debug counter to avoid log spam
    private var periodicityDebugCount = 0
    
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
        periodicityDebugCount = 0  // Reset for each video
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
        // FULL PIPELINE DEBUG - Compare with PC (controlled by enableVerboseLogging) // =====================================================================
        
        // Step 1: Compute signals
        var signals = computeSignals(poseSeq)
        
        // Step 2: Drop isolated angle spikes before interpolation.
        // MediaPipe can briefly put a knee/ankle landmark on the wrong edge of
        // the silhouette without changing L/R identity. Those single-frame
        // cliffs look exactly like the dashboard jitter users report; converting
        // them to short gaps lets interpolation/EMA handle them as uncertainty.
        signals = removeSignalOutliers(signals)
        
        // Step 3: Interpolate gaps
        signals = interpolateSignals(signals)
        
        // Step 4: Segment-aware smoothing
        signals = smoothSignals(signals)
        
        // Step 5: Compute velocities
        signals = computeVelocities(signals, poseSeq.fps)

        // Step 6: Determine near leg
        val nearLeg = determineNearLeg(poseSeq)
        
        // Step 7: Multi-mode step detection
        val (stepMode, stepSignal, modeScores) = selectStepSignalMode(signals, poseSeq, nearLeg, poseSeq.fps)
        val steps = detectStepsFromSignal(stepSignal, poseSeq.fps)
        Log.d(TAG, "Step signal mode: ${stepMode.value}, detected ${steps.size} steps")
        
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
            Log.d(TAG, "Near leg: $nearLeg, Mode: ${stepMode.value}")
            Log.d(TAG, "frame,value")
            for (i in stepSignal.indices) {
                Log.d(TAG, "$i,${String.format("%.6f", stepSignal[i])}")
            }
            Log.d(TAG, "DETECTED PEAKS: ${steps.map { it.frameIdx }}")
            Log.d(TAG, "PEAK TIMES: ${steps.map { String.format("%.3f", it.timeS) }}")
        }
        
        if (steps.size < 4) {
            return Pair(null, createDiagnostics(poseSeq, QualityFlag.NO_CYCLES, 
                "only_${steps.size}_steps", steps.size, stepMode = stepMode.value))
        }
        
        // Step 7: Segment strides
        val strides = segmentStrides(steps, signals, poseSeq.fps)
        
        // Step 8: Validate strides (use selected step signal for quality scoring)
        val validatedStrides = validateStrides(strides, signals, poseSeq.fps, stepSignal, stepMode)
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
        
        val numFramesValid = signals.isValid.count { it }
        val validFrameRate = if (poseSeq.numFramesTotal > 0) numFramesValid.toFloat() / poseSeq.numFramesTotal else 0f
        val diagnostics = createDiagnostics(
            poseSeq, qualityFlag, "",
            steps.size, validStrides.size,
            stepMode = stepMode.value,
            selectionReason = selectionReason,
            numFramesValid = numFramesValid,
            validFrameRate = validFrameRate
        )
        
        // Store signals and strides for visualization
        AnalysisSession.extractedSignals = signals
        AnalysisSession.extractedStrides = validatedStrides
        AnalysisSession.selectedStrideIndices = selectedIndices
        AnalysisSession.stepSignalMode = stepMode.value
        
        if (features.valid_stride_count == 0) {
            return Pair(null, diagnostics)
        }
        
        // Summary for PC comparison (filter logcat with: adb logcat -s GaitDebug)
        Log.d(TAG, "=== EXTRACTION SUMMARY (compare with PC) ===")
        Log.d(TAG, "  num_frames_valid=${diagnostics.numFramesValid}/${diagnostics.numFramesTotal}, valid_rate=${String.format("%.1f", diagnostics.validFrameRate * 100)}%")
        Log.d(TAG, "  step_mode=$stepMode, peaks=[${steps.map { it.frameIdx }}], peak_times=[${steps.map { String.format("%.2f", it.timeS) }}]")
        Log.d(TAG, "  selected_stride_indices=$selectedIndices, cadence=${String.format("%.1f", features.cadence_spm)} spm, stride_time=${String.format("%.2f", features.stride_time_s)} s")
        Log.d(TAG, "  knee_rom_L=${String.format("%.1f", features.knee_left_rom)}°, knee_rom_R=${String.format("%.1f", features.knee_right_rom)}°")
        
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
        val frontalKneeOffsetLeft = FloatArray(n) { Float.NaN }
        val frontalKneeOffsetRight = FloatArray(n) { Float.NaN }
        
        // Positions
        val ankleLeftX = FloatArray(n) { Float.NaN }
        val ankleRightX = FloatArray(n) { Float.NaN }
        val ankleLeftY = FloatArray(n) { Float.NaN }
        val ankleRightY = FloatArray(n) { Float.NaN }
        val kneeLeftX = FloatArray(n) { Float.NaN }
        val kneeRightX = FloatArray(n) { Float.NaN }
        val kneeLeftY = FloatArray(n) { Float.NaN }
        val kneeRightY = FloatArray(n) { Float.NaN }
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
            
            val kp = frame.keypoints
            val conf = frame.confidences
            
            // Positions
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_ANKLE)) {
                ankleLeftX[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][0]
                ankleLeftY[idx] = kp[MediaPipePoseBackend.LEFT_ANKLE][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_ANKLE)) {
                ankleRightX[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][0]
                ankleRightY[idx] = kp[MediaPipePoseBackend.RIGHT_ANKLE][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_KNEE)) {
                kneeLeftX[idx] = kp[MediaPipePoseBackend.LEFT_KNEE][0]
                kneeLeftY[idx] = kp[MediaPipePoseBackend.LEFT_KNEE][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_KNEE)) {
                kneeRightX[idx] = kp[MediaPipePoseBackend.RIGHT_KNEE][0]
                kneeRightY[idx] = kp[MediaPipePoseBackend.RIGHT_KNEE][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.RIGHT_ANKLE)) {
                interAnkleDist[idx] = abs(kp[MediaPipePoseBackend.RIGHT_ANKLE][0] - kp[MediaPipePoseBackend.LEFT_ANKLE][0])
            }
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_HIP)) {
                hipLeftY[idx] = kp[MediaPipePoseBackend.LEFT_HIP][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_HIP)) {
                hipRightY[idx] = kp[MediaPipePoseBackend.RIGHT_HIP][1]
            }
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.LEFT_ANKLE)) {
                frontalKneeOffsetLeft[idx] = computeFrontalKneeOffset(
                    kp[MediaPipePoseBackend.LEFT_HIP],
                    kp[MediaPipePoseBackend.LEFT_KNEE],
                    kp[MediaPipePoseBackend.LEFT_ANKLE]
                )
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_HIP, MediaPipePoseBackend.RIGHT_KNEE, MediaPipePoseBackend.RIGHT_ANKLE)) {
                frontalKneeOffsetRight[idx] = computeFrontalKneeOffset(
                    kp[MediaPipePoseBackend.RIGHT_HIP],
                    kp[MediaPipePoseBackend.RIGHT_KNEE],
                    kp[MediaPipePoseBackend.RIGHT_ANKLE]
                )
            }
            
            // Knee angles (hip-knee-ankle) - used in features
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.LEFT_ANKLE)) {
                kneeAngleLeft[idx] = computeAngle(
                    kp[MediaPipePoseBackend.LEFT_HIP],
                    kp[MediaPipePoseBackend.LEFT_KNEE],
                    kp[MediaPipePoseBackend.LEFT_ANKLE]
                )
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_HIP, MediaPipePoseBackend.RIGHT_KNEE, MediaPipePoseBackend.RIGHT_ANKLE)) {
                kneeAngleRight[idx] = computeAngle(
                    kp[MediaPipePoseBackend.RIGHT_HIP],
                    kp[MediaPipePoseBackend.RIGHT_KNEE],
                    kp[MediaPipePoseBackend.RIGHT_ANKLE]
                )
            }
            
            // Trunk angle - used in features
            val hasTrunkConfidence = hasConfidence(
                conf,
                MediaPipePoseBackend.LEFT_SHOULDER,
                MediaPipePoseBackend.RIGHT_SHOULDER,
                MediaPipePoseBackend.LEFT_HIP,
                MediaPipePoseBackend.RIGHT_HIP
            )
            val midShoulder = if (hasTrunkConfidence) {
                floatArrayOf(
                    (kp[MediaPipePoseBackend.LEFT_SHOULDER][0] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][0]) / 2f,
                    (kp[MediaPipePoseBackend.LEFT_SHOULDER][1] + kp[MediaPipePoseBackend.RIGHT_SHOULDER][1]) / 2f
                )
            } else {
                null
            }
            val midHip = if (hasConfidence(conf, MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.RIGHT_HIP)) {
                floatArrayOf(
                    (kp[MediaPipePoseBackend.LEFT_HIP][0] + kp[MediaPipePoseBackend.RIGHT_HIP][0]) / 2f,
                    (kp[MediaPipePoseBackend.LEFT_HIP][1] + kp[MediaPipePoseBackend.RIGHT_HIP][1]) / 2f
                )
            } else {
                null
            }
            if (midShoulder != null && midHip != null) {
                trunkAngle[idx] = computeTrunkLean(midShoulder, midHip)
            }
            
            // Ankle angles (foot-ankle-knee) - visualization only
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_FOOT_INDEX, MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.LEFT_KNEE)) {
                ankleAngleLeft[idx] = computeAngle(
                    kp[MediaPipePoseBackend.LEFT_FOOT_INDEX],
                    kp[MediaPipePoseBackend.LEFT_ANKLE],
                    kp[MediaPipePoseBackend.LEFT_KNEE]
                ) - 90f  // Offset like original: 0° = foot flat
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_FOOT_INDEX, MediaPipePoseBackend.RIGHT_ANKLE, MediaPipePoseBackend.RIGHT_KNEE)) {
                ankleAngleRight[idx] = computeAngle(
                    kp[MediaPipePoseBackend.RIGHT_FOOT_INDEX],
                    kp[MediaPipePoseBackend.RIGHT_ANKLE],
                    kp[MediaPipePoseBackend.RIGHT_KNEE]
                ) - 90f
            }
            
            // Hip angles (knee-hip-shoulder) - visualization only
            if (hasConfidence(conf, MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.LEFT_SHOULDER)) {
                hipAngleLeft[idx] = 180f - computeAngle(
                    kp[MediaPipePoseBackend.LEFT_KNEE],
                    kp[MediaPipePoseBackend.LEFT_HIP],
                    kp[MediaPipePoseBackend.LEFT_SHOULDER]
                )
            }
            if (hasConfidence(conf, MediaPipePoseBackend.RIGHT_KNEE, MediaPipePoseBackend.RIGHT_HIP, MediaPipePoseBackend.RIGHT_SHOULDER)) {
                hipAngleRight[idx] = 180f - computeAngle(
                    kp[MediaPipePoseBackend.RIGHT_KNEE],
                    kp[MediaPipePoseBackend.RIGHT_HIP],
                    kp[MediaPipePoseBackend.RIGHT_SHOULDER]
                )
            }
            
            // Stride angle (left heel - hip center - right heel) - visualization only
            if (midHip != null && hasConfidence(conf, MediaPipePoseBackend.LEFT_HEEL, MediaPipePoseBackend.RIGHT_HEEL)) {
                strideAngle[idx] = computeAngle(
                    kp[MediaPipePoseBackend.LEFT_HEEL],
                    midHip,
                    kp[MediaPipePoseBackend.RIGHT_HEEL]
                )
            }
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
            frontalKneeOffsetLeft = frontalKneeOffsetLeft,
            frontalKneeOffsetRight = frontalKneeOffsetRight,
            ankleLeftX = ankleLeftX,
            ankleRightX = ankleRightX,
            ankleLeftY = ankleLeftY,
            ankleRightY = ankleRightY,
            kneeLeftX = kneeLeftX,
            kneeRightX = kneeRightX,
            kneeLeftY = kneeLeftY,
            kneeRightY = kneeRightY,
            hipLeftY = hipLeftY,
            hipRightY = hipRightY,
            ankleLeftVy = FloatArray(n) { Float.NaN },
            ankleRightVy = FloatArray(n) { Float.NaN },
            hipAvgVy = FloatArray(n) { Float.NaN }
        )
    }

    private fun hasConfidence(confidences: FloatArray, vararg keypoints: Int): Boolean {
        return keypoints.all { kp -> confidences.getOrNull(kp)?.let { it >= minConfidence } == true }
    }

    /**
     * Safe single-landmark confidence read. Returns 0f for out-of-range indices
     * instead of throwing — MediaPipe variants can return fewer than 33 slots,
     * and step detection reads confidence in hot loops where a defensive
     * fallback is preferable to a per-frame crash. Mirrors the bounds-safe
     * access used in [GaitVision.com.mediapipe.ROITracker].
     */
    private fun confidenceAt(confidences: FloatArray, idx: Int): Float =
        confidences.getOrNull(idx) ?: 0f
    
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

    private fun computeFrontalKneeOffset(hip: FloatArray, knee: FloatArray, ankle: FloatArray): Float {
        val axisX = ankle[0] - hip[0]
        val axisY = ankle[1] - hip[1]
        val kneeX = knee[0] - hip[0]
        val kneeY = knee[1] - hip[1]
        val axisLen = sqrt(axisX * axisX + axisY * axisY)
        if (axisLen < 1e-6f) return Float.NaN

        return (axisX * kneeY - axisY * kneeX) / axisLen
    }
    
    // =========================================================================
    // Signal Processing
    // =========================================================================

    private fun removeSignalOutliers(signals: Signals): Signals {
        removeIsolatedSpikes(signals.kneeAngleLeft, radius = 2, minAbsDeviation = 10f)
        removeIsolatedSpikes(signals.kneeAngleRight, radius = 2, minAbsDeviation = 10f)
        removeIsolatedSpikes(signals.trunkAngle, radius = 2, minAbsDeviation = 8f)
        removeIsolatedSpikes(signals.ankleAngleLeft, radius = 2, minAbsDeviation = 14f)
        removeIsolatedSpikes(signals.ankleAngleRight, radius = 2, minAbsDeviation = 14f)
        removeIsolatedSpikes(signals.hipAngleLeft, radius = 2, minAbsDeviation = 10f)
        removeIsolatedSpikes(signals.hipAngleRight, radius = 2, minAbsDeviation = 10f)
        removeIsolatedSpikes(signals.strideAngle, radius = 2, minAbsDeviation = 12f)
        return signals
    }

    private fun removeIsolatedSpikes(
        signal: FloatArray,
        radius: Int,
        minAbsDeviation: Float,
        thresholdMad: Float = 3.5f
    ) {
        val cleaned = signal.copyOf()
        for (i in signal.indices) {
            val value = signal[i]
            if (value.isNaN()) continue

            val neighbors = mutableListOf<Float>()
            val start = maxOf(0, i - radius)
            val end = minOf(signal.size - 1, i + radius)
            for (j in start..end) {
                if (j == i || signal[j].isNaN()) continue
                neighbors.add(signal[j])
            }
            if (neighbors.size < 3) continue

            val sortedNeighbors = neighbors.sorted()
            val median = percentile(sortedNeighbors, 50f)
            val deviations = sortedNeighbors.map { abs(it - median) }.sorted()
            val mad = percentile(deviations, 50f).coerceAtLeast(0.5f)
            val absDeviation = abs(value - median)

            if (absDeviation >= minAbsDeviation && absDeviation > thresholdMad * mad) {
                cleaned[i] = Float.NaN
            }
        }

        for (i in signal.indices) {
            signal[i] = cleaned[i]
        }
    }
    
    private fun interpolateSignals(signals: Signals): Signals {
        interpolateArray(signals.interAnkleDist, maxInterpGap)
        interpolateArray(signals.kneeAngleLeft, maxInterpGap)
        interpolateArray(signals.kneeAngleRight, maxInterpGap)
        interpolateArray(signals.trunkAngle, maxInterpGap)
        interpolateArray(signals.ankleAngleLeft, maxInterpGap)
        interpolateArray(signals.ankleAngleRight, maxInterpGap)
        interpolateArray(signals.hipAngleLeft, maxInterpGap)
        interpolateArray(signals.hipAngleRight, maxInterpGap)
        interpolateArray(signals.strideAngle, maxInterpGap)
        interpolateArray(signals.frontalKneeOffsetLeft, maxInterpGap)
        interpolateArray(signals.frontalKneeOffsetRight, maxInterpGap)
        interpolateArray(signals.ankleLeftX, maxInterpGap)
        interpolateArray(signals.ankleRightX, maxInterpGap)
        interpolateArray(signals.ankleLeftY, maxInterpGap)
        interpolateArray(signals.ankleRightY, maxInterpGap)
        interpolateArray(signals.kneeLeftX, maxInterpGap)
        interpolateArray(signals.kneeRightX, maxInterpGap)
        interpolateArray(signals.kneeLeftY, maxInterpGap)
        interpolateArray(signals.kneeRightY, maxInterpGap)
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
        smoothAngleSignal(signals.kneeAngleLeft, alpha = 0.32f)
        smoothAngleSignal(signals.kneeAngleRight, alpha = 0.32f)
        smoothAngleSignal(signals.trunkAngle, alpha = 0.40f)
        smoothAngleSignal(signals.ankleAngleLeft, alpha = 0.36f)
        smoothAngleSignal(signals.ankleAngleRight, alpha = 0.36f)
        smoothAngleSignal(signals.hipAngleLeft, alpha = 0.36f)
        smoothAngleSignal(signals.hipAngleRight, alpha = 0.36f)
        smoothAngleSignal(signals.strideAngle, alpha = 0.36f)
        emaSmoothGapAware(signals.frontalKneeOffsetLeft, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.frontalKneeOffsetRight, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.ankleLeftY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.ankleRightY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeLeftX, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeRightX, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeLeftY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.kneeRightY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.hipLeftY, emaAlpha, maxInterpGap)
        emaSmoothGapAware(signals.hipRightY, emaAlpha, maxInterpGap)
        return signals
    }

    private fun smoothAngleSignal(signal: FloatArray, alpha: Float) {
        medianSmooth(signal, radius = 2)
        forwardBackwardEmaSmooth(signal, alpha.coerceAtMost(emaAlpha), maxInterpGap)
    }

    private fun medianSmooth(signal: FloatArray, radius: Int) {
        val source = signal.copyOf()
        for (i in signal.indices) {
            if (source[i].isNaN()) continue

            val window = mutableListOf<Float>()
            val start = maxOf(0, i - radius)
            val end = minOf(signal.size - 1, i + radius)
            for (j in start..end) {
                if (!source[j].isNaN()) {
                    window.add(source[j])
                }
            }
            if (window.size >= 3) {
                signal[i] = percentile(window.sorted(), 50f)
            }
        }
    }

    private fun forwardBackwardEmaSmooth(arr: FloatArray, alpha: Float, maxBridgeGap: Int) {
        emaSmoothGapAware(arr, alpha, maxBridgeGap)
        arr.reverse()
        emaSmoothGapAware(arr, alpha, maxBridgeGap)
        arr.reverse()
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
            val conf = frame.confidences
            val leftConf = (
                confidenceAt(conf, MediaPipePoseBackend.LEFT_HIP) +
                confidenceAt(conf, MediaPipePoseBackend.LEFT_KNEE) +
                confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE)) / 3f
            val rightConf = (
                confidenceAt(conf, MediaPipePoseBackend.RIGHT_HIP) +
                confidenceAt(conf, MediaPipePoseBackend.RIGHT_KNEE) +
                confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE)) / 3f
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
        val minConf = GaitConfig.STEP_DETECT_MIN_CONF
        
        return when (mode) {
            StepSignalMode.INTER_ANKLE -> {
                signals.interAnkleDist.copyOf()
            }
            
            StepSignalMode.MAX_ANKLE_VY -> {
                val ankleLeftY = FloatArray(n) { Float.NaN }
                val ankleRightY = FloatArray(n) { Float.NaN }
                
                for (frame in poseSeq.frames) {
                    val conf = frame.confidences
                    val idx = frame.frameIdx
                    if (idx >= n) continue
                    if (confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE) >= minConf) {
                        ankleLeftY[idx] = frame.keypoints[MediaPipePoseBackend.LEFT_ANKLE][1]
                    }
                    if (confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE) >= minConf) {
                        ankleRightY[idx] = frame.keypoints[MediaPipePoseBackend.RIGHT_ANKLE][1]
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
                    val conf = frame.confidences
                    val idx = frame.frameIdx
                    if (idx >= n) continue
                    
                    val leftConf = minOf(
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_HIP),
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_KNEE),
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE)
                    )
                    if (leftConf >= minConf) {
                        kneeLeft[idx] = computeAngle(
                            kp[MediaPipePoseBackend.LEFT_HIP],
                            kp[MediaPipePoseBackend.LEFT_KNEE],
                            kp[MediaPipePoseBackend.LEFT_ANKLE]
                        )
                    }
                    
                    val rightConf = minOf(
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_HIP),
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_KNEE),
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE)
                    )
                    if (rightConf >= minConf) {
                        kneeRight[idx] = computeAngle(
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
        val minConf = GaitConfig.STEP_DETECT_MIN_CONF
        val minConfBoth = GaitConfig.STEP_DETECT_MIN_CONF_BOTH
        var validCount = 0
        
        for (frame in poseSeq.frames) {
            val conf = frame.confidences
            val isValid = when (mode) {
                StepSignalMode.INTER_ANKLE -> {
                    minOf(confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE),
                          confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE)) >= minConfBoth
                }
                StepSignalMode.MAX_ANKLE_VY -> {
                    confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE) >= minConf ||
                    confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE) >= minConf
                }
                StepSignalMode.MIN_KNEE_ANGLE -> {
                    val leftConf = minOf(
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_HIP),
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_KNEE),
                        confidenceAt(conf, MediaPipePoseBackend.LEFT_ANKLE)
                    )
                    val rightConf = minOf(
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_HIP),
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_KNEE),
                        confidenceAt(conf, MediaPipePoseBackend.RIGHT_ANKLE)
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
        
        // First compute acf[0] (the energy/variance at lag 0)
        var acf0 = 0f
        for (i in 0 until n) {
            acf0 += centered[i] * centered[i]
        }
        if (acf0 < 1e-8f) return 0f
        
        // Now compute normalized autocorrelation for all lags
        for (lag in 0 until n) {
            var sum = 0f
            for (i in 0 until n - lag) {
                sum += centered[i] * centered[i + lag]
            }
            acf[lag] = sum / acf0
        }
        
        val minLag = (fps * minStepTimeS).toInt()
        val maxLag = minOf((fps * maxStepTimeS).toInt(), n - 1)
        if (minLag >= maxLag) return 0f
        
        // Debug ACF values (first call only to avoid spam)
        if (periodicityDebugCount < 1) {
            Log.d(TAG, "=== PERIODICITY ACF DEBUG ===")
            Log.d(TAG, "  ACF[0] (energy): ${String.format("%.4f", acf0)}")
            Log.d(TAG, "  ACF normalized[0]: ${String.format("%.4f", acf[0])}")
            Log.d(TAG, "  ACF normalized[10]: ${String.format("%.4f", acf.getOrElse(10) { 0f })}")
            Log.d(TAG, "  ACF normalized[20]: ${String.format("%.4f", acf.getOrElse(20) { 0f })}")
            Log.d(TAG, "  ACF normalized[30]: ${String.format("%.4f", acf.getOrElse(30) { 0f })}")
            Log.d(TAG, "  minLag: $minLag, maxLag: $maxLag")
            periodicityDebugCount++
        }
        
        val search = acf.slice(minLag..maxLag)
        val maxInSearch = search.maxOrNull() ?: 0f
        val peaks = findPeaks(search.toFloatArray(), 1, 0.05f)
        
        val result = if (peaks.isNotEmpty()) {
            search[peaks[0]].coerceIn(0f, 1f)
        } else {
            maxInSearch.coerceIn(0f, 1f) * 0.5f
        }
        
        // Debug final result (first call only)
        if (periodicityDebugCount == 1) {
            Log.d(TAG, "  Max in search: ${String.format("%.4f", maxInSearch)}")
            Log.d(TAG, "  Peaks found: ${peaks.size}, result: ${String.format("%.4f", result)}")
            periodicityDebugCount++
        }
        
        return result
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
        
        // Log all mode scores for debugging
        Log.d(TAG, "=== STEP SIGNAL MODE SELECTION ===")
        for ((mode, pair) in candidates) {
            val q = pair.second
            val marker = if (mode == selected) " [SELECTED]" else ""
            Log.d(TAG, "  ${mode.value}: peaks=${q.peakCount}, periodicity=${String.format("%.3f", q.periodicityScore)}, " +
                "coverage=${String.format("%.3f", q.confidenceCoverage)}, finalScore=${String.format("%.3f", q.finalScore)}$marker")
        }
        
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
        
        val minWidth = maxOf((fps * minStepPeakWidthS).toInt(), 2)
        var peaks = findPeaks(signalClean, minDistance, minProminence)
        peaks = filterPeaksByWidth(signalClean, peaks, minWidth)
        
        // Peak snap: recenter each peak to local maximum within ±2 frames
        // This neutralizes ±1 frame edge-case differences vs PC
        val snappedPeaks = snapPeaksToLocalMax(peaks, signalClean, snapWindow = 2)
        
        
        // Log if any peaks were snapped
        if (peaks != snappedPeaks) {
            Log.d(TAG, "Peak snap applied: $peaks -> $snappedPeaks")
        } else {
            Log.d(TAG, "Peak snap: no changes (peaks already at local max)")
        }
        
        val steps = snappedPeaks.map { idx ->
            StepEvent(frameIdx = idx, timeS = idx.toFloat() / fps)
        }
        return cleanStepIntervals(steps, fps)
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
    
    /**
     * Filter peaks by minimum width (scipy-style: width at half prominence).
     * Rejects narrow jitter spikes.
     */
    private fun filterPeaksByWidth(
        signal: FloatArray,
        peaks: List<Int>,
        minWidth: Int
    ): List<Int> {
        if (peaks.isEmpty() || minWidth <= 1) return peaks
        return peaks.filter { peakIdx ->
            val peakHeight = signal[peakIdx]
            var leftMin = signal[peakIdx]
            for (j in maxOf(0, peakIdx - 1) downTo 0) {
                if (signal[j] < leftMin) leftMin = signal[j]
                if (signal[j] > peakHeight) break
            }
            var rightMin = signal[peakIdx]
            for (j in (peakIdx + 1) until signal.size) {
                if (signal[j] < rightMin) rightMin = signal[j]
                if (signal[j] > peakHeight) break
            }
            val prominence = peakHeight - maxOf(leftMin, rightMin)
            val halfHeight = peakHeight - prominence / 2f
            var leftCross = peakIdx
            while (leftCross > 0 && signal[leftCross - 1] >= halfHeight) leftCross--
            var rightCross = peakIdx
            while (rightCross < signal.size - 1 && signal[rightCross + 1] >= halfHeight) rightCross++
            (rightCross - leftCross + 1) >= minWidth
        }
    }
    
    /**
     * Post-hoc cleaning: remove peaks that create outlier intervals (likely jitter).
     * Drop when BOTH adjacent intervals are < 0.5× median.
     */
    private fun cleanStepIntervals(steps: List<StepEvent>, fps: Float): List<StepEvent> {
        var current = steps
        for (_iter in 0 until MAX_INTERVAL_CLEANING_ITERATIONS) {
            if (current.size < 3) break
            val intervals = (0 until current.size - 1).map {
                current[it + 1].timeS - current[it].timeS
            }
            val medianInt = intervals.sorted()[intervals.size / 2]
            if (medianInt < 1e-6f) break
            val toDrop = (1 until current.size - 1).filter { i ->
                intervals[i - 1] < 0.5f * medianInt && intervals[i] < 0.5f * medianInt
            }
            if (toDrop.isEmpty()) break
            current = current.filterIndexed { j, _ -> j !in toDrop }
        }
        return current
    }
    
    // =========================================================================
    // Stride Segmentation and Validation
    // =========================================================================
    
    /**
     * Segment steps into strides (2 steps each). Overlapping: peak[i]->peak[i+2] for all i.
     * Matches PC pipeline _segment_strides. N peaks produce N-2 strides.
     */
    private fun segmentStrides(steps: List<StepEvent>, signals: Signals, fps: Float): List<Stride> {
        val strides = mutableListOf<Stride>()
        if (steps.size < 3) return strides

        val maxFrame = signals.timestamps.size - 1
        for (i in 0 until steps.size - 2) {
            val step1 = steps[i]
            val step2 = steps[i + 1]
            val endStep = steps[i + 2]
            val endFrame = minOf(endStep.frameIdx, maxFrame)
            val endTime = endStep.timeS

            strides.add(Stride(
                startFrame = step1.frameIdx,
                endFrame = endFrame,
                startTimeS = step1.timeS,
                endTimeS = endTime,
                step1Frame = step1.frameIdx,
                step2Frame = step2.frameIdx,
                step1TimeS = step1.timeS,
                step2TimeS = step2.timeS
            ))
        }
        return strides
    }
    
    private fun validateStrides(
        strides: List<Stride>,
        signals: Signals,
        fps: Float,
        stepSignal: FloatArray? = null,
        stepMode: StepSignalMode = StepSignalMode.INTER_ANKLE
    ): List<Stride> {
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
            // Use exclusive range [start, end) to match PC's Python slicing
            val validFrames = (start until end).count { signals.isValid[it] }
            val totalFrames = end - start
            val validPct = if (totalFrames > 0) validFrames.toFloat() / totalFrames else 0f
            
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
            
            // Check 3: Knee ROM (exclusive range to match PC)
            val kneeLeftSlice = signals.kneeAngleLeft.slice(start until end).filter { !it.isNaN() }
            val kneeRightSlice = signals.kneeAngleRight.slice(start until end).filter { !it.isNaN() }
            
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
            
            // Max invalid run (gap) - duration-based, FPS-independent
            val segValid = signals.isValid.slice(start until end)
            var maxInvalidRun = 0
            var current = 0
            for (v in segValid) {
                if (!v) {
                    current++
                    maxInvalidRun = maxOf(maxInvalidRun, current)
                } else {
                    current = 0
                }
            }
            val maxGapS = if (fps > 0) maxInvalidRun / fps else 0f
            val gapScore = when {
                maxGapS <= 0.15f -> 1.0f
                maxGapS <= 0.4f -> 0.5f
                else -> 0.4f
            }
            
            // Signal quality: use step signal if available, scale by valid_pct
            val segmentForQuality = when {
                stepSignal != null && start < stepSignal.size && end <= stepSignal.size ->
                    stepSignal.slice(start until end).filter { !it.isNaN() }
                else ->
                    signals.interAnkleDist.slice(start until end).filter { !it.isNaN() }
            }
            val signalQualityRaw = computeSignalQualityScore(segmentForQuality, stepMode, stepSignal)
            val signalQuality = signalQualityRaw * validPct
            
            val qualityScore = (
                0.28f * validPct +
                0.20f * timingScore +
                0.20f * romMargin.coerceIn(0f, 1f) +
                0.05f * gapScore +
                0.27f * signalQuality
            )
            
            // Debug logging for quality score breakdown
            Log.d(TAG, "  Stride [${start}..${end}] quality breakdown: " +
                "validPct=${String.format("%.4f", validPct)}, " +
                "timingScore=${String.format("%.4f", timingScore)}, " +
                "romMargin=${String.format("%.4f", romMargin.coerceIn(0f, 1f))}, " +
                "gapScore=${String.format("%.4f", gapScore)}, " +
                "signalQuality=${String.format("%.4f", signalQuality)} -> total=${String.format("%.4f", qualityScore)}")
            
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
    
    private fun computeSignalQualityScore(
        segment: List<Float>,
        stepMode: StepSignalMode = StepSignalMode.INTER_ANKLE,
        fullSignal: FloatArray? = null
    ): Float {
        if (segment.size < 5) return 0f
        
        val peakVal = segment.maxOrNull() ?: return 0f
        val troughVal = segment.minOrNull() ?: return 0f
        val signalRange = peakVal - troughVal
        
        val useRelative = stepMode != StepSignalMode.INTER_ANKLE && fullSignal != null &&
            fullSignal.count { !it.isNaN() } >= 5
        
        val (amplitudeScore, peakScore) = if (useRelative) {
            val fullValid = fullSignal!!.filter { !it.isNaN() }
            val globalRange = (fullValid.maxOrNull() ?: 0f) - (fullValid.minOrNull() ?: 0f)
            val globalMax = fullValid.maxOrNull() ?: 1e-8f
            val amplitudeRatio = if (globalRange > 1e-8f) minOf(signalRange / globalRange, 1f) else 1f
            val peakRatio = if (globalMax > 1e-8f) minOf(peakVal / globalMax, 1f) else 1f
            Pair(0.3f + 0.7f * amplitudeRatio, 0.3f + 0.7f * peakRatio)
        } else {
            val amp = when {
                signalRange < 0.05f -> 0.1f
                signalRange < 0.08f -> 0.3f
                signalRange < 0.10f -> 0.6f
                signalRange < 0.15f -> 1.0f
                else -> 0.9f
            }
            val peak = when {
                peakVal < 0.06f -> 0.2f
                peakVal < 0.10f -> 0.5f
                peakVal < 0.15f -> 1.0f
                else -> 0.9f
            }
            Pair(amp, peak)
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
        
        // Find best (i, i+2) pair. With overlapping strides, stride i and i+1 overlap;
        // we need stride i and i+2 for 4 consecutive steps. For each j, take first
        // non-overlapping stride after it (which is i+2), then break.
        var bestPair: List<Stride>? = null
        var bestPairScore = -1f
        var bestPairIndices = listOf<Int>()
        var bestPairReason = ""
        
        Log.d(TAG, "=== CONSECUTIVE PAIR SCORES (i, i+2 only) ===")
        
        for (run in runs) {
            if (run.size >= 2) {
                for (j in 0 until run.size) {
                    val (idx1, s1) = run[j]
                    for (m in j + 1 until run.size) {
                        val (idx2, s2) = run[m]
                        if (s1.endFrame > s2.startFrame) continue  // overlapping, skip
                        val pairScore = s1.qualityScore + s2.qualityScore
                        Log.d(TAG, "  Pair [$idx1, $idx2]: score=${String.format("%.4f", pairScore)} " +
                            "(q1=${String.format("%.4f", s1.qualityScore)}, q2=${String.format("%.4f", s2.qualityScore)}) " +
                            "frames [${s1.startFrame}..${s1.endFrame}] + [${s2.startFrame}..${s2.endFrame}]")
                        if (pairScore > bestPairScore) {
                            bestPairScore = pairScore
                            bestPair = listOf(s1, s2)
                            bestPairIndices = listOf(idx1, idx2)
                            bestPairReason = "best_consecutive_pair"
                        }
                        break  // s2 is first non-overlapping after s1 (consecutive strides only)
                    }
                }
            }
        }
        
        if (bestPair != null) {
            Log.d(TAG, "  Best: [$bestPairIndices] score=$bestPairScore")
            return Triple(bestPair, bestPairReason, bestPairIndices)
        }
        
        // No consecutive pair found - do not fall back to non-consecutive (PC parity)
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
        stepMode: String = "inter_ankle",
        selectionReason: String = "",
        numFramesValid: Int? = null,
        validFrameRate: Float? = null
    ): GaitDiagnostics {
        val validCount = numFramesValid ?: poseSeq.frames.size
        val validRate = validFrameRate ?: poseSeq.detectionRate
        return GaitDiagnostics(
            videoId = poseSeq.videoId,
            fpsDetected = poseSeq.fps,
            durationS = poseSeq.durationS,
            numFramesTotal = poseSeq.numFramesTotal,
            numFramesValid = validCount,
            validFrameRate = validRate,
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
