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

    // Pose-position jitter validation metrics (not used by scoring model)
    val rawPoseJitter: Float? = null,
    val smoothedPoseJitter: Float? = null,
    val poseJitterReductionPct: Float? = null,
    val rawPoseVelocity: Float? = null,
    val smoothedPoseVelocity: Float? = null,
    val poseVelocityRetentionPct: Float? = null,
    val rawPoseSnapRate: Float? = null,
    val smoothedPoseSnapRate: Float? = null,
    val poseSnapReductionPct: Float? = null,
    val poseConfidenceCoverage: Float? = null,
    val poseMedianBodyScale: Float? = null,

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

    // Stride data (JSON for small list of stride boundaries)
    val stridesJson: String? = null,
    val selectedStrideIndicesJson: String? = null
)
