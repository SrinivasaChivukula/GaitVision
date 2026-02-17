package GaitVision.com.gait

import android.util.Log
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseSequence
import kotlin.math.*

internal object FeatureComputer {
    private const val TAG = "GaitLogging"

    fun compute(
        signals: Signals,
        validStrides: List<Stride>,
        poseSeq: PoseSequence,
        extremaPercentileLo: Float,
        extremaPercentileHi: Float
    ): Triple<GaitFeatures, String, List<Int>> {
        val (selectedStrides, selectionReason, selectedIndices) = CycleDetector.select2InnerCycles(validStrides)

        Log.d(TAG, "Stride selection: total=${validStrides.size}, valid=${validStrides.count { it.isValid }}, selected=$selectedIndices")
        if (selectedStrides.size < 2) {
            Log.w(TAG, "Not enough valid strides selected!")
            return Triple(GaitFeatures.empty(), "", emptyList())
        }

        val bodyWidth = computeBodyWidth(poseSeq)
        val bw = bodyWidth + 1e-8f

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
        val stepTimeAsymmetry = asymmetryIndex(legAStepTimes.average().toFloat(), legBStepTimes.average().toFloat())

        val strideLengths = mutableListOf<Float>()
        for (s in selectedStrides) {
            val startX = signals.ankleLeftX[s.startFrame]
            val endX = signals.ankleLeftX[minOf(s.endFrame, signals.ankleLeftX.size - 1)]
            if (!startX.isNaN() && !endX.isNaN()) strideLengths.add(abs(endX - startX))
        }
        val strideLengthNorm = if (strideLengths.isNotEmpty()) strideLengths.average().toFloat() / bw else 0f

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
        val stepLengthAsymmetry = asymmetryIndex(ankleALengths.averageOrZero(), ankleBLengths.averageOrZero())

        val kneeLeftRom = selectedStrides.map { it.kneeRomLeft }.average().toFloat()
        val kneeRightRom = selectedStrides.map { it.kneeRomRight }.average().toFloat()
        val kneeLeftMax = selectedStrides.map { it.kneeMaxLeft }.average().toFloat()
        val kneeRightMax = selectedStrides.map { it.kneeMaxRight }.average().toFloat()

        val maxInterAnkleValues = selectedStrides.mapNotNull { sliceStrideSignal(signals.interAnkleDist, it).maxOrNull() }
        val strideAmpNorm = maxInterAnkleValues.averageOrZero() / bw
        val interAnkleValues = selectedStrides.flatMap { sliceStrideSignal(signals.interAnkleDist, it) }
        val interAnkleCv = if (interAnkleValues.isNotEmpty()) interAnkleValues.std() / (interAnkleValues.average().toFloat() + 1e-8f) else 0f

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
            val ldjH = computeLDJ(signals.trunkAngle.slice(start until end), poseSeq.fps)
            if (!ldjH.isNaN() && ldjH > 0) ldjHipValues.add(ldjH)
        }
        val ldjKneeLeft = ldjKneeLeftValues.averageOrZero()
        val ldjKneeRight = ldjKneeRightValues.averageOrZero()
        val ldjHip = ldjHipValues.averageOrZero()

        val trunkValuesAbs = selectedStrides.flatMap { sliceStrideSignal(signals.trunkAngle, it) }.map { abs(it) }
        val trunkLeanStdDeg = if (trunkValuesAbs.isNotEmpty()) trunkValuesAbs.std() else 0f

        val strideLengthRelL = mutableListOf<Float>()
        val strideLengthRelR = mutableListOf<Float>()
        val ankleApRangePerStride = mutableListOf<Float>()
        val ankleApRangePerStrideCombined = mutableListOf<Float>()
        val midHipDriftPerStride = mutableListOf<Float>()
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.ankleLeftX.size - 1)
            if (end <= start) continue
            val aLStart = signals.ankleLeftX[start] - signals.midHipX[start]
            val aLEnd = signals.ankleLeftX[end] - signals.midHipX[end]
            val aRStart = signals.ankleRightX[start] - signals.midHipX[start]
            val aREnd = signals.ankleRightX[end] - signals.midHipX[end]
            if (!aLStart.isNaN() && !aLEnd.isNaN()) strideLengthRelL.add(abs(aLEnd - aLStart))
            if (!aRStart.isNaN() && !aREnd.isNaN()) strideLengthRelR.add(abs(aREnd - aRStart))
            var leftRng: Float? = null
            var rightRng: Float? = null
            val ankleRelL = (start until end).mapNotNull { i -> val v = signals.ankleLeftX[i] - signals.midHipX[i]; if (!v.isNaN()) v else null }
            val ankleRelR = (start until end).mapNotNull { i -> val v = signals.ankleRightX[i] - signals.midHipX[i]; if (!v.isNaN()) v else null }
            if (ankleRelL.size >= 5) {
                val sorted = ankleRelL.sorted()
                val rng = percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo)
                ankleApRangePerStride.add(rng); leftRng = rng
            }
            if (ankleRelR.size >= 5) {
                val sorted = ankleRelR.sorted()
                val rng = percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo)
                ankleApRangePerStride.add(rng); rightRng = rng
            }
            when {
                leftRng != null && rightRng != null -> ankleApRangePerStrideCombined.add((leftRng + rightRng) / 2f)
                leftRng != null -> ankleApRangePerStrideCombined.add(leftRng)
                rightRng != null -> ankleApRangePerStrideCombined.add(rightRng)
            }
            val mStart = signals.midHipX[start]; val mEnd = signals.midHipX[end]
            if (!mStart.isNaN() && !mEnd.isNaN()) midHipDriftPerStride.add(abs(mEnd - mStart))
        }
        val strideLengthRelLNorm = strideLengthRelL.averageOrZero() / bw
        val strideLengthRelRNorm = strideLengthRelR.averageOrZero() / bw
        val strideLengthRelAsym = asymmetryIndex(strideLengthRelLNorm, strideLengthRelRNorm)
        val ankleApRangeRelNorm = ankleApRangePerStride.averageOrZero() / bw
        val midHipApDriftNorm = midHipDriftPerStride.averageOrZero() / bw

        val tKneeLeftPeak = mutableListOf<Float>()
        val tKneeRightPeak = mutableListOf<Float>()
        val tTrunkPeakAbs = mutableListOf<Float>()
        val tToeClearanceLeft = mutableListOf<Float>()
        val tToeClearanceRight = mutableListOf<Float>()
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.kneeAngleLeft.size - 1)
            val cycleLen = (end - start).coerceAtLeast(1)
            fun argmaxPct(sig: FloatArray, transform: (Float) -> Float = { it }): Float? {
                val best = (start until end).filter { !sig[it].isNaN() }.maxByOrNull { transform(sig[it]) } ?: return null
                return (best - start).toFloat() / cycleLen
            }
            argmaxPct(signals.kneeAngleLeft)?.let { tKneeLeftPeak.add(it) }
            argmaxPct(signals.kneeAngleRight)?.let { tKneeRightPeak.add(it) }
            argmaxPct(signals.trunkAngle) { abs(it) }?.let { tTrunkPeakAbs.add(it) }
            argmaxPct(signals.toeLeftY) { -it }?.let { tToeClearanceLeft.add(it) }
            argmaxPct(signals.toeRightY) { -it }?.let { tToeClearanceRight.add(it) }
        }

        val toeClearanceLeftMax = mutableListOf<Float>()
        val toeClearanceRightMax = mutableListOf<Float>()
        val toeClearanceLeftRange = mutableListOf<Float>()
        val toeClearanceRightRange = mutableListOf<Float>()
        val footPitchLeftMean = mutableListOf<Float>()
        val footPitchRightMean = mutableListOf<Float>()
        val footPitchLeftRange = mutableListOf<Float>()
        val footPitchRightRange = mutableListOf<Float>()
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.toeLeftY.size - 1)
            val toeClearL = (start until end).mapNotNull { i -> val v = -signals.toeLeftY[i]; if (!v.isNaN()) v else null }
            val toeClearR = (start until end).mapNotNull { i -> val v = -signals.toeRightY[i]; if (!v.isNaN()) v else null }
            val footPitchL = (start until end).mapNotNull { i ->
                val tc = -signals.toeLeftY[i]; val hc = -signals.heelLeftY[i]
                if (!tc.isNaN() && !hc.isNaN()) tc - hc else null
            }
            val footPitchR = (start until end).mapNotNull { i ->
                val tc = -signals.toeRightY[i]; val hc = -signals.heelRightY[i]
                if (!tc.isNaN() && !hc.isNaN()) tc - hc else null
            }
            if (toeClearL.size >= 5) {
                val sorted = toeClearL.sorted()
                toeClearanceLeftMax.add(percentile(sorted, extremaPercentileHi))
                toeClearanceLeftRange.add(percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo))
            }
            if (toeClearR.size >= 5) {
                val sorted = toeClearR.sorted()
                toeClearanceRightMax.add(percentile(sorted, extremaPercentileHi))
                toeClearanceRightRange.add(percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo))
            }
            if (footPitchL.size >= 5) {
                footPitchLeftMean.add(footPitchL.average().toFloat())
                val sorted = footPitchL.sorted()
                footPitchLeftRange.add(percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo))
            }
            if (footPitchR.size >= 5) {
                footPitchRightMean.add(footPitchR.average().toFloat())
                val sorted = footPitchR.sorted()
                footPitchRightRange.add(percentile(sorted, extremaPercentileHi) - percentile(sorted, extremaPercentileLo))
            }
        }

        val trunkPooled = selectedStrides.flatMap { sliceStrideSignal(signals.trunkAngle, it) }
        val trunkAbsPooled = trunkPooled.map { abs(it) }
        val dt = if (poseSeq.fps > 0) 1f / poseSeq.fps else 0.033f
        val trunkVelPooled = mutableListOf<Float>()
        for (s in selectedStrides) {
            val start = s.startFrame
            val end = minOf(s.endFrame, signals.trunkAngle.size - 1)
            for (i in start until end - 1) {
                val v = (signals.trunkAngle[i + 1] - signals.trunkAngle[i]) / dt
                if (!v.isNaN()) trunkVelPooled.add(v)
            }
        }
        val trunkAbsMeanDeg = if (trunkAbsPooled.isNotEmpty()) trunkAbsPooled.average().toFloat() else 0f
        val trunkAbsP95Deg = if (trunkAbsPooled.size >= 5) percentile(trunkAbsPooled.sorted(), extremaPercentileHi) else 0f
        val trunkVelAbs = trunkVelPooled.map { abs(it) }
        val trunkAngVelMeanAbs = if (trunkVelAbs.isNotEmpty()) trunkVelAbs.average().toFloat() else 0f
        val trunkAngVelP95Abs = if (trunkVelAbs.size >= 5) percentile(trunkVelAbs.sorted(), extremaPercentileHi) else 0f

        fun diffOfTwo(list: List<Float>): Float = abs(list[0] - list[1])
        val cadencePerStride = strideTimes.map { 60f / it }
        val stepTimeAsymPerStride = selectedStrides.map { s -> asymmetryIndex(s.step2TimeS - s.startTimeS, s.endTimeS - s.step2TimeS) }
        val strideLengthNormPerStride = strideLengths.map { it / bw }
        val strideAmpNormPerStride = maxInterAnkleValues.map { it / bw }
        val stepLengthAsymPerStride = selectedStrides.indices.map { i ->
            asymmetryIndex(ankleALengths.getOrNull(i) ?: 0f, ankleBLengths.getOrNull(i) ?: 0f)
        }
        val strideLengthRelLNormPerStride = strideLengthRelL.map { it / bw }
        val strideLengthRelRNormPerStride = strideLengthRelR.map { it / bw }
        val strideLengthRelAsymPerStride = strideLengthRelL.zip(strideLengthRelR) { l, r -> asymmetryIndex(l, r) }
        val ankleApRangeNormPerStride = ankleApRangePerStrideCombined.map { it / bw }
        val midHipDriftNormPerStride = midHipDriftPerStride.map { it / bw }

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
                stride_length_relL_norm = strideLengthRelLNorm,
                stride_length_relR_norm = strideLengthRelRNorm,
                stride_length_rel_asym = strideLengthRelAsym,
                ankle_ap_range_rel_norm = ankleApRangeRelNorm,
                midHip_ap_drift_norm = midHipApDriftNorm,
                t_knee_left_peak_pct = tKneeLeftPeak.averageOrZero(),
                t_knee_right_peak_pct = tKneeRightPeak.averageOrZero(),
                t_trunk_peak_abs_pct = tTrunkPeakAbs.averageOrZero(),
                t_toe_clearance_left_pct = tToeClearanceLeft.averageOrZero(),
                t_toe_clearance_right_pct = tToeClearanceRight.averageOrZero(),
                toe_clearance_left_max = toeClearanceLeftMax.averageOrZero(),
                toe_clearance_right_max = toeClearanceRightMax.averageOrZero(),
                toe_clearance_left_range = toeClearanceLeftRange.averageOrZero(),
                toe_clearance_right_range = toeClearanceRightRange.averageOrZero(),
                foot_pitch_left_mean = footPitchLeftMean.averageOrZero(),
                foot_pitch_right_mean = footPitchRightMean.averageOrZero(),
                foot_pitch_left_range = footPitchLeftRange.averageOrZero(),
                foot_pitch_right_range = footPitchRightRange.averageOrZero(),
                trunk_abs_mean_deg = trunkAbsMeanDeg,
                trunk_abs_p95_deg = trunkAbsP95Deg,
                trunk_ang_vel_mean_abs = trunkAngVelMeanAbs,
                trunk_ang_vel_p95_abs = trunkAngVelP95Abs,
                cadence_diff = diffOfTwo(cadencePerStride),
                stride_time_diff = diffOfTwo(strideTimes),
                step_time_asymmetry_diff = diffOfTwo(stepTimeAsymPerStride),
                stride_length_norm_diff = diffOfTwo(strideLengthNormPerStride),
                stride_amp_norm_diff = diffOfTwo(strideAmpNormPerStride),
                step_length_asymmetry_diff = diffOfTwo(stepLengthAsymPerStride),
                knee_left_rom_diff = diffOfTwo(selectedStrides.map { it.kneeRomLeft }),
                knee_right_rom_diff = diffOfTwo(selectedStrides.map { it.kneeRomRight }),
                knee_left_max_diff = diffOfTwo(selectedStrides.map { it.kneeMaxLeft }),
                knee_right_max_diff = diffOfTwo(selectedStrides.map { it.kneeMaxRight }),
                ldj_knee_left_diff = diffOfTwo(ldjKneeLeftValues),
                ldj_knee_right_diff = diffOfTwo(ldjKneeRightValues),
                ldj_hip_diff = diffOfTwo(ldjHipValues),
                stride_length_relL_norm_diff = diffOfTwo(strideLengthRelLNormPerStride),
                stride_length_relR_norm_diff = diffOfTwo(strideLengthRelRNormPerStride),
                stride_length_rel_asym_diff = diffOfTwo(strideLengthRelAsymPerStride),
                ankle_ap_range_rel_norm_diff = diffOfTwo(ankleApRangeNormPerStride),
                midHip_ap_drift_norm_diff = diffOfTwo(midHipDriftNormPerStride),
                t_knee_left_peak_pct_diff = diffOfTwo(tKneeLeftPeak),
                t_knee_right_peak_pct_diff = diffOfTwo(tKneeRightPeak),
                t_trunk_peak_abs_pct_diff = diffOfTwo(tTrunkPeakAbs),
                t_toe_clearance_left_pct_diff = diffOfTwo(tToeClearanceLeft),
                t_toe_clearance_right_pct_diff = diffOfTwo(tToeClearanceRight),
                toe_clearance_left_max_diff = diffOfTwo(toeClearanceLeftMax),
                toe_clearance_right_max_diff = diffOfTwo(toeClearanceRightMax),
                toe_clearance_left_range_diff = diffOfTwo(toeClearanceLeftRange),
                toe_clearance_right_range_diff = diffOfTwo(toeClearanceRightRange),
                foot_pitch_left_mean_diff = diffOfTwo(footPitchLeftMean),
                foot_pitch_right_mean_diff = diffOfTwo(footPitchRightMean),
                foot_pitch_left_range_diff = diffOfTwo(footPitchLeftRange),
                foot_pitch_right_range_diff = diffOfTwo(footPitchRightRange),
                valid_stride_count = selectedStrides.size
            ),
            selectionReason,
            selectedIndices
        )
    }

    private fun asymmetryIndex(left: Float, right: Float): Float =
        if (left + right == 0f) 0f else (left - right) / (left + right)

    private fun computeBodyWidth(poseSeq: PoseSequence): Float {
        val widths = mutableListOf<Float>()
        for (frame in poseSeq.frames) {
            val sw = abs(frame.keypoints[MediaPipePoseBackend.LEFT_SHOULDER][0] - frame.keypoints[MediaPipePoseBackend.RIGHT_SHOULDER][0])
            val hw = abs(frame.keypoints[MediaPipePoseBackend.LEFT_HIP][0] - frame.keypoints[MediaPipePoseBackend.RIGHT_HIP][0])
            widths.add((sw + hw) / 2f)
        }
        return if (widths.isNotEmpty()) widths.average().toFloat() else 0.1f
    }

    private fun sliceStrideSignal(signal: FloatArray, stride: Stride): List<Float> =
        signal.slice(stride.startFrame until minOf(stride.endFrame, signal.size)).filter { !it.isNaN() }

    private fun computeLDJ(signal: List<Float>, fps: Float): Float {
        val valid = signal.filter { !it.isNaN() }
        if (valid.size < 3) return 0f
        val dt = if (fps > 0) 1f / fps else 0.033f
        val velocity = (1 until valid.size).map { (valid[it] - valid[it - 1]) / dt }
        if (velocity.size < 2) return 0f
        val accel = (1 until velocity.size).map { (velocity[it] - velocity[it - 1]) / dt }
        return if (accel.isEmpty()) 0f else sqrt(accel.map { it * it }.average().toFloat())
    }
}
