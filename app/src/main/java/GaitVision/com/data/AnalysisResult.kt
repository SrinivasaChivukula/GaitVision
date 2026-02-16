package GaitVision.com.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "results",
    foreignKeys = [ForeignKey(
        entity = Patient::class,
        parentColumns = ["participantId"],
        childColumns = ["patientId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("patientId")]
)
data class AnalysisResult(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val patientId: Int,

    // Video info
    val videoFileName: String = "",
    val videoLengthMicroseconds: Long? = null,
    val recordedAt: Long = System.currentTimeMillis(),

    // Scores
    val overallScore: Double? = null,
    val aeScore: Float? = null,
    val ridgeScore: Float? = null,
    val pcaScore: Float? = null,

    // Pipeline metadata
    val stepSignalMode: String? = null,
    val validStrideCount: Int = 0,
    val qualityFlag: String? = null,

    // Diagnostic metadata (for accurate CSV export on reload)
    val fpsDetected: Float? = null,
    val numFramesTotal: Int = 0,
    val numFramesValid: Int = 0,
    val validFrameRate: Float? = null,
    val numStepsDetected: Int = 0,
    val walkingDirection: String? = null,
    val wasFlipped: Boolean = false,

    // 16 Gait Features (individual columns for queryability)
    val cadenceSpm: Float? = null,
    val strideTimeS: Float? = null,
    val strideTimeCv: Float? = null,
    val stepTimeAsymmetry: Float? = null,
    val strideLengthNorm: Float? = null,
    val strideAmpNorm: Float? = null,
    val stepLengthAsymmetry: Float? = null,
    val kneeLeftRom: Float? = null,
    val kneeRightRom: Float? = null,
    val kneeLeftMax: Float? = null,
    val kneeRightMax: Float? = null,
    val ldjKneeLeft: Float? = null,
    val ldjKneeRight: Float? = null,
    val ldjHip: Float? = null,
    val trunkLeanStdDeg: Float? = null,
    val interAnkleCv: Float? = null,

    // V2: Camera-robust distance features (5)
    val strideLengthRelLNorm: Float? = null,
    val strideLengthRelRNorm: Float? = null,
    val strideLengthRelAsym: Float? = null,
    val ankleApRangeRelNorm: Float? = null,
    val midHipApDriftNorm: Float? = null,

    // V2: Timing-of-extrema (5)
    val tKneeLeftPeakPct: Float? = null,
    val tKneeRightPeakPct: Float? = null,
    val tTrunkPeakAbsPct: Float? = null,
    val tToeClearanceLeftPct: Float? = null,
    val tToeClearanceRightPct: Float? = null,

    // V2: Foot clearance + pitch (8)
    val toeClearanceLeftMax: Float? = null,
    val toeClearanceRightMax: Float? = null,
    val toeClearanceLeftRange: Float? = null,
    val toeClearanceRightRange: Float? = null,
    val footPitchLeftMean: Float? = null,
    val footPitchRightMean: Float? = null,
    val footPitchLeftRange: Float? = null,
    val footPitchRightRange: Float? = null,

    // V2: Trunk enrichments (4)
    val trunkAbsMeanDeg: Float? = null,
    val trunkAbsP95Deg: Float? = null,
    val trunkAngVelMeanAbs: Float? = null,
    val trunkAngVelP95Abs: Float? = null,

    // V2: Diff features (31)
    val cadenceDiff: Float? = null,
    val strideTimeDiff: Float? = null,
    val stepTimeAsymmetryDiff: Float? = null,
    val strideLengthNormDiff: Float? = null,
    val strideAmpNormDiff: Float? = null,
    val stepLengthAsymmetryDiff: Float? = null,
    val kneeLeftRomDiff: Float? = null,
    val kneeRightRomDiff: Float? = null,
    val kneeLeftMaxDiff: Float? = null,
    val kneeRightMaxDiff: Float? = null,
    val ldjKneeLeftDiff: Float? = null,
    val ldjKneeRightDiff: Float? = null,
    val ldjHipDiff: Float? = null,
    val strideLengthRelLNormDiff: Float? = null,
    val strideLengthRelRNormDiff: Float? = null,
    val strideLengthRelAsymDiff: Float? = null,
    val ankleApRangeRelNormDiff: Float? = null,
    val midHipApDriftNormDiff: Float? = null,
    val tKneeLeftPeakPctDiff: Float? = null,
    val tKneeRightPeakPctDiff: Float? = null,
    val tTrunkPeakAbsPctDiff: Float? = null,
    val tToeClearanceLeftPctDiff: Float? = null,
    val tToeClearanceRightPctDiff: Float? = null,
    val toeClearanceLeftMaxDiff: Float? = null,
    val toeClearanceRightMaxDiff: Float? = null,
    val toeClearanceLeftRangeDiff: Float? = null,
    val toeClearanceRightRangeDiff: Float? = null,
    val footPitchLeftMeanDiff: Float? = null,
    val footPitchRightMeanDiff: Float? = null,
    val footPitchLeftRangeDiff: Float? = null,
    val footPitchRightRangeDiff: Float? = null,

    // Stride data (JSON for small list of stride boundaries)
    val stridesJson: String? = null,
    val selectedStrideIndicesJson: String? = null
)
