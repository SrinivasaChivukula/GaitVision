package GaitVision.com.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "signal_data",
    foreignKeys = [ForeignKey(
        entity = AnalysisResult::class,
        parentColumns = ["id"],
        childColumns = ["resultId"],
        onDelete = ForeignKey.CASCADE
    )],
    indices = [Index("resultId")]
)
data class SignalData(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val resultId: Long,
    val frameNumber: Int,

    // Inter-ankle distance (stride detection signal)
    val interAnkleDist: Float? = null,

    // Knee angles (L/R) processed (interpolated + smoothed)
    val kneeAngleLeft: Float? = null,
    val kneeAngleRight: Float? = null,

    // Trunk angle — processed (interpolated + smoothed)
    val trunkAngle: Float? = null,

    // Ankle Y positions (L/R) processed
    val ankleLeftY: Float? = null,
    val ankleRightY: Float? = null,

    // Hip Y positions (L/R) processed
    val hipLeftY: Float? = null,
    val hipRightY: Float? = null,

    // Foot positions (V2: heel/toe clearance, midHip AP)
    val heelLeftY: Float? = null,
    val heelRightY: Float? = null,
    val toeLeftY: Float? = null,
    val toeRightY: Float? = null,
    val midHipX: Float? = null,

    // Ankle velocities (L/R) derived from smoothed positions
    val ankleLeftVy: Float? = null,
    val ankleRightVy: Float? = null,

    // Frame metadata
    val isValid: Boolean = true,
    val timestamp: Float? = null
)
