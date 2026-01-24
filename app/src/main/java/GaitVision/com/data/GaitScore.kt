package GaitVision.com.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.PrimaryKey

@Entity(
    tableName = "gait_scores",
    foreignKeys = [
        ForeignKey(
            entity = Patient::class,
            parentColumns = ["participantId"],
            childColumns = ["patientId"],
            onDelete = ForeignKey.CASCADE
        ),
        ForeignKey(
            entity = Video::class,
            parentColumns = ["id"],
            childColumns = ["videoId"],
            onDelete = ForeignKey.CASCADE
        )
    ]
)
data class GaitScore(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val patientId: Int,
    val videoId: Long,
    val overallScore: Double,
    val recordedAt: Long = System.currentTimeMillis(),
    // Additional gait metrics that could be useful
    val leftKneeScore: Double? = null,
    val rightKneeScore: Double? = null,
    val leftHipScore: Double? = null,
    val rightHipScore: Double? = null,
    val torsoScore: Double? = null
)
