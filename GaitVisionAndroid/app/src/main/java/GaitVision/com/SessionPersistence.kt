package GaitVision.com

import android.content.Context
import android.util.Log
import GaitVision.com.data.AnalysisResult
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.SignalData
import GaitVision.com.data.repository.AnalysisResultRepository
import GaitVision.com.data.repository.SignalDataRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

// Saves the current AnalysisSession to DB: one AnalysisResult + N SignalData rows.
// Used by both the single-video flow and batch.
// Throws on DB error so the caller can decide what to do (toast vs mark row failed).
// Returns the new result id.
suspend fun persistCurrentSession(
    context: Context,
    db: AppDatabase,
    patientId: Int,
    videoFileName: String? = null,
): Long? = withContext(Dispatchers.IO) {
    val resolvedName = videoFileName ?: lookupDisplayName(context)
    val features = AnalysisSession.extractedFeatures
    val score = AnalysisSession.scoringResult
    val diagnostics = AnalysisSession.extractionDiagnostics
    val jitter = AnalysisSession.jitterComparison

    val result = AnalysisResult(
        patientId = patientId,
        videoFileName = resolvedName,
        videoLengthMicroseconds = AnalysisSession.videoLength,
        recordedAt = AnalysisSession.recordingDate,

        // scores
        overallScore = score?.getScoreForDatabase(),
        aeScore = score?.aeScore,
        ridgeScore = score?.ridgeScore,
        pcaScore = score?.pcaScore,

        // pipeline metadata
        stepSignalMode = AnalysisSession.stepSignalMode,
        validStrideCount = features?.valid_stride_count ?: 0,
        qualityFlag = diagnostics?.qualityFlag?.name,

        // diagnostics
        fpsDetected = diagnostics?.fpsDetected,
        numFramesTotal = diagnostics?.numFramesTotal ?: 0,
        numFramesValid = diagnostics?.numFramesValid ?: 0,
        validFrameRate = diagnostics?.validFrameRate,
        numStepsDetected = diagnostics?.numStepsDetected ?: 0,
        walkingDirection = diagnostics?.walkingDirection,
        wasFlipped = diagnostics?.wasFlipped ?: false,

        // pose-position jitter validation metrics
        rawPoseJitter = jitter?.raw?.jitterSecondDiffNorm?.finiteOrNull(),
        smoothedPoseJitter = jitter?.smoothed?.jitterSecondDiffNorm?.finiteOrNull(),
        poseJitterReductionPct = jitter?.jitterReductionPct?.finiteOrNull(),
        rawPoseVelocity = jitter?.raw?.meanVelocityNorm?.finiteOrNull(),
        smoothedPoseVelocity = jitter?.smoothed?.meanVelocityNorm?.finiteOrNull(),
        poseVelocityRetentionPct = jitter?.velocityRetentionPct?.finiteOrNull(),
        rawPoseSnapRate = jitter?.raw?.snapRate?.finiteOrNull(),
        smoothedPoseSnapRate = jitter?.smoothed?.snapRate?.finiteOrNull(),
        poseSnapReductionPct = jitter?.snapReductionPct?.finiteOrNull(),
        poseConfidenceCoverage = jitter?.smoothed?.confidenceCoverage?.finiteOrNull(),
        poseMedianBodyScale = jitter?.smoothed?.medianBodyScale?.finiteOrNull(),

        // 16 features
        cadenceSpm = features?.cadence_spm,
        strideTimeS = features?.stride_time_s,
        strideTimeCv = features?.stride_time_cv,
        stepTimeAsymmetry = features?.step_time_asymmetry,
        strideLengthNorm = features?.stride_length_norm,
        strideAmpNorm = features?.stride_amp_norm,
        stepLengthAsymmetry = features?.step_length_asymmetry,
        kneeLeftRom = features?.knee_left_rom,
        kneeRightRom = features?.knee_right_rom,
        kneeLeftMax = features?.knee_left_max,
        kneeRightMax = features?.knee_right_max,
        ldjKneeLeft = features?.ldj_knee_left,
        ldjKneeRight = features?.ldj_knee_right,
        ldjHip = features?.ldj_hip,
        trunkLeanStdDeg = features?.trunk_lean_std_deg,
        interAnkleCv = features?.inter_ankle_cv,

        stridesJson = AnalysisSession.extractedStrides?.let { strides ->
            JSONArray().apply {
                strides.forEach { s ->
                    put(JSONObject().apply {
                        put("sf", s.startFrame)
                        put("ef", s.endFrame)
                        put("st", s.startTimeS)
                        put("et", s.endTimeS)
                        put("s1f", s.step1Frame)
                        put("s2f", s.step2Frame)
                        put("s1t", s.step1TimeS)
                        put("s2t", s.step2TimeS)
                        put("v", s.isValid)
                        put("r", s.invalidReason ?: JSONObject.NULL)
                    })
                }
            }.toString()
        },
        selectedStrideIndicesJson = AnalysisSession.selectedStrideIndices?.let {
            JSONArray(it).toString()
        }
    )

    val resultRepo = AnalysisResultRepository(db.analysisResultDao())
    val signalRepo = SignalDataRepository(db.signalDataDao())

    val resultId = resultRepo.insertResult(result)
    AnalysisSession.currentResultId = resultId

    // signals are optional; not every quality flag produces them
    val signals = AnalysisSession.extractedSignals
    if (signals != null) {
        val maxFrames = signals.kneeAngleLeft.size
        val rows = ArrayList<SignalData>(maxFrames)
        for (frame in 0 until maxFrames) {
            rows.add(SignalData(
                resultId = resultId,
                frameNumber = frame,
                interAnkleDist = signals.interAnkleDist.getOrNull(frame)?.takeIf { !it.isNaN() },
                kneeAngleLeft = signals.kneeAngleLeft.getOrNull(frame)?.takeIf { !it.isNaN() },
                kneeAngleRight = signals.kneeAngleRight.getOrNull(frame)?.takeIf { !it.isNaN() },
                trunkAngle = signals.trunkAngle.getOrNull(frame)?.takeIf { !it.isNaN() },
                ankleLeftY = signals.ankleLeftY.getOrNull(frame)?.takeIf { !it.isNaN() },
                ankleRightY = signals.ankleRightY.getOrNull(frame)?.takeIf { !it.isNaN() },
                hipLeftY = signals.hipLeftY.getOrNull(frame)?.takeIf { !it.isNaN() },
                hipRightY = signals.hipRightY.getOrNull(frame)?.takeIf { !it.isNaN() },
                ankleLeftVy = signals.ankleLeftVy.getOrNull(frame)?.takeIf { !it.isNaN() },
                ankleRightVy = signals.ankleRightVy.getOrNull(frame)?.takeIf { !it.isNaN() },
                isValid = signals.isValid.getOrNull(frame) ?: true,
                timestamp = signals.timestamps.getOrNull(frame)?.takeIf { !it.isNaN() }
            ))
        }
        if (rows.isNotEmpty()) signalRepo.insertSignalDataList(rows)
        Log.d("SessionPersistence", "saved result $resultId with ${rows.size} signal rows")
    } else {
        Log.d("SessionPersistence", "saved result $resultId (no signals)")
    }

    resultId
}

// Thin wrapper so we can call the shared util with the session's URI.
private fun lookupDisplayName(context: Context): String {
    val uri = AnalysisSession.galleryUri ?: return ""
    return lookupDisplayName(context, uri)
}

private fun Float.finiteOrNull(): Float? = if (isFinite()) this else null
