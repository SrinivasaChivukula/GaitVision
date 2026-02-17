package GaitVision.com.gait

import android.util.Log
import kotlin.math.*

internal object CycleDetector {
    private const val TAG = "GaitLogging"

    fun detectSteps(signal: FloatArray, fps: Float, minStepTimeS: Float, maxStepTimeS: Float, stepDistanceFactor: Float, stepProminenceFactor: Float): List<StepEvent> {
        val validPct = signal.count { !it.isNaN() }.toFloat() / signal.size
        if (validPct < 0.3f) return emptyList()

        val signalClean = signal.map { if (it.isNaN()) 0f else it }.toFloatArray()
        val stepFrames = estimateStepPeriod(signalClean, fps, minStepTimeS, maxStepTimeS)
        val minDistance = maxOf((stepFrames * stepDistanceFactor).toInt(), 5)
        val validVals = signalClean.filter { it != 0f }
        val minProminence = if (validVals.isNotEmpty()) validVals.std() * stepProminenceFactor else 0.01f

        val peaks = findPeaks(signalClean, minDistance, minProminence)
        val snappedPeaks = snapPeaksToLocalMax(peaks, signalClean, 2)
        if (peaks != snappedPeaks) Log.d(TAG, "Peak snap applied: $peaks -> $snappedPeaks")

        return snappedPeaks.map { StepEvent(frameIdx = it, timeS = it.toFloat() / fps) }
    }

    fun segmentStrides(steps: List<StepEvent>, signals: Signals, fps: Float): List<Stride> {
        if (steps.size < 3) return emptyList()
        val maxFrame = signals.timestamps.size - 1
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

    fun validateStrides(
        strides: List<Stride>,
        signals: Signals,
        fps: Float,
        minStepTimeS: Float,
        maxStepTimeS: Float,
        useRobustExtrema: Boolean,
        extremaPercentileLo: Float,
        extremaPercentileHi: Float
    ): List<Stride> {
        if (strides.isEmpty()) return strides
        val stepTimes = strides.map { it.step2TimeS - it.step1TimeS }
        val globalStepTime = stepTimes.median()
        val allInterAnkle = signals.interAnkleDist.filter { !it.isNaN() }
        val expectedRange = if (allInterAnkle.size >= 2) allInterAnkle.maxOrNull()!! - allInterAnkle.minOrNull()!! else 1f

        return strides.map { stride ->
            val start = stride.startFrame
            val end = minOf(stride.endFrame, signals.isValid.size - 1)
            if (end - start < 5) return@map stride.copy(isValid = false, invalidReason = "degenerate", qualityScore = 0f)

            val validFrames = (start until end).count { signals.isValid[it] }
            val totalFrames = end - start
            val validPct = if (totalFrames > 0) validFrames.toFloat() / totalFrames else 0f
            if (validPct == 0f) return@map stride.copy(isValid = false, invalidReason = "degenerate", qualityScore = 0f)

            val coverageScore = validPct
            val stepTimeDev = abs((stride.step2TimeS - stride.step1TimeS) - globalStepTime) / (globalStepTime + 1e-8f)
            val timingScore = maxOf(0f, 1f - stepTimeDev / 0.5f)

            val kneeLeftSlice = signals.kneeAngleLeft.slice(start until end).filter { !it.isNaN() }
            val kneeRightSlice = signals.kneeAngleRight.slice(start until end).filter { !it.isNaN() }
            val romLeft = if (kneeLeftSlice.size >= 5) computeRom(kneeLeftSlice, useRobustExtrema, extremaPercentileLo, extremaPercentileHi) else 0f
            val romRight = if (kneeRightSlice.size >= 5) computeRom(kneeRightSlice, useRobustExtrema, extremaPercentileLo, extremaPercentileHi) else 0f
            val romScore = sigmoidRomScore(maxOf(romLeft, romRight))

            val interAnkleSegment = signals.interAnkleDist.slice(start until end).filter { !it.isNaN() }
            val signalQuality = computeSignalQualityScore(interAnkleSegment)
            val segmentRange = if (interAnkleSegment.size >= 2) interAnkleSegment.maxOrNull()!! - interAnkleSegment.minOrNull()!! else 0f
            val motionEnergy = minOf(1f, segmentRange / (expectedRange + 1e-8f))

            val qualityScore = (0.20f * coverageScore + 0.20f * timingScore + 0.15f * romScore + 0.30f * signalQuality + 0.15f * motionEnergy).coerceIn(0f, 1f)

            Log.d(TAG, "  Stride [${start}..${end}] quality: cov=${String.format("%.3f", coverageScore)} tim=${String.format("%.3f", timingScore)} rom=${String.format("%.3f", romScore)} sig=${String.format("%.3f", signalQuality)} mot=${String.format("%.3f", motionEnergy)} -> ${String.format("%.4f", qualityScore)}")

            stride.copy(
                isValid = true,
                validFramePct = validPct,
                kneeRomLeft = romLeft,
                kneeRomRight = romRight,
                kneeMaxLeft = if (kneeLeftSlice.size >= 5) computeMaxAngle(kneeLeftSlice, useRobustExtrema, extremaPercentileHi) else 0f,
                kneeMaxRight = if (kneeRightSlice.size >= 5) computeMaxAngle(kneeRightSlice, useRobustExtrema, extremaPercentileHi) else 0f,
                qualityScore = qualityScore
            )
        }
    }

    fun select2InnerCycles(strides: List<Stride>): Triple<List<Stride>, String, List<Int>> {
        val valid = strides.withIndex().filter { it.value.isValid }
        if (valid.size < 2) return Triple(emptyList(), "", emptyList())

        var bestScore = -1f
        var bestA = -1
        var bestB = -1
        Log.d(TAG, "Pair selection")
        for (i in valid.indices) {
            for (j in i + 1 until valid.size) {
                val a = valid[i]; val b = valid[j]
                if (a.value.endFrame != b.value.startFrame) continue
                val score = a.value.qualityScore + b.value.qualityScore
                Log.d(TAG, "  Pair [${a.index}, ${b.index}]: score=${String.format("%.4f", score)}")
                if (score > bestScore) { bestScore = score; bestA = i; bestB = j }
            }
        }
        if (bestA < 0) return Triple(emptyList(), "", emptyList())
        val a = valid[bestA]; val b = valid[bestB]
        Log.d(TAG, "  Best: [${a.index}, ${b.index}] score=${String.format("%.4f", bestScore)}")
        return Triple(listOf(a.value, b.value), "best_pair", listOf(a.index, b.index))
    }

    private fun snapPeaksToLocalMax(peaks: List<Int>, signal: FloatArray, snapWindow: Int): List<Int> {
        if (peaks.isEmpty()) return peaks
        return peaks.map { peakIdx ->
            val start = maxOf(0, peakIdx - snapWindow)
            val end = minOf(signal.size - 1, peakIdx + snapWindow)
            var bestIdx = start
            var bestVal = signal[start]
            for (i in (start + 1)..end) {
                if (signal[i] > bestVal) { bestVal = signal[i]; bestIdx = i }
            }
            bestIdx
        }.distinct()
    }

    private fun estimateStepPeriod(signal: FloatArray, fps: Float, minStepTimeS: Float, maxStepTimeS: Float): Float {
        val mean = signal.average().toFloat()
        val centered = signal.map { it - mean }.toFloatArray()
        val std = sqrt(centered.map { it * it }.average().toFloat())
        if (std < 1e-6f) return fps * 0.5f

        val n = centered.size
        val acf = FloatArray(n)
        for (lag in 0 until n) {
            var sum = 0f
            for (i in 0 until n - lag) sum += centered[i] * centered[i + lag]
            acf[lag] = sum
        }
        val acf0 = acf[0] + 1e-8f
        for (i in acf.indices) acf[i] /= acf0

        val minLag = (fps * minStepTimeS).toInt()
        val maxLag = minOf((fps * maxStepTimeS).toInt(), n - 1)
        if (minLag >= maxLag) return fps * 0.5f
        val search = acf.slice(minLag..maxLag)
        val peaks = findPeaks(search.toFloatArray(), 1, 0.1f)
        return if (peaks.isNotEmpty()) (minLag + peaks[0]).toFloat() else (minLag + search.indices.maxByOrNull { search[it] }!!).toFloat()
    }

    private fun findPeaks(signal: FloatArray, minDistance: Int, minProminence: Float): List<Int> {
        val localMaxima = mutableListOf<Int>()
        for (i in 1 until signal.size - 1) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) localMaxima.add(i)
        }
        if (localMaxima.isEmpty()) return emptyList()

        val prominences = FloatArray(localMaxima.size)
        for ((idx, peakIdx) in localMaxima.withIndex()) {
            val peakHeight = signal[peakIdx]
            var leftBound = peakIdx - 1
            while (leftBound > 0 && signal[leftBound] < peakHeight) leftBound--
            var leftMin = signal[peakIdx]
            for (j in leftBound until peakIdx) if (signal[j] < leftMin) leftMin = signal[j]
            var rightBound = peakIdx + 1
            while (rightBound < signal.size - 1 && signal[rightBound] < peakHeight) rightBound++
            var rightMin = signal[peakIdx]
            for (j in (peakIdx + 1)..rightBound) if (signal[j] < rightMin) rightMin = signal[j]
            prominences[idx] = peakHeight - maxOf(leftMin, rightMin)
        }

        val filteredPeaks = mutableListOf<Int>()
        for ((idx, peakIdx) in localMaxima.withIndex()) {
            if (prominences[idx] >= minProminence) {
                if (filteredPeaks.isEmpty() || peakIdx - filteredPeaks.last() >= minDistance) {
                    filteredPeaks.add(peakIdx)
                } else if (prominences[idx] > prominences[localMaxima.indexOf(filteredPeaks.last())]) {
                    filteredPeaks[filteredPeaks.size - 1] = peakIdx
                }
            }
        }
        return filteredPeaks
    }

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

    private fun computeRom(values: List<Float>, useRobust: Boolean, pLo: Float, pHi: Float): Float {
        if (values.size < 2) return 0f
        return if (useRobust && values.size >= 10) {
            val sorted = values.sorted()
            percentile(sorted, pHi) - percentile(sorted, pLo)
        } else (values.maxOrNull() ?: 0f) - (values.minOrNull() ?: 0f)
    }

    private fun computeMaxAngle(values: List<Float>, useRobust: Boolean, pHi: Float): Float {
        if (values.isEmpty()) return 0f
        return if (useRobust && values.size >= 10) percentile(values.sorted(), pHi) else (values.maxOrNull() ?: 0f)
    }
}
