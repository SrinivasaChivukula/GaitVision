package GaitVision.com.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "patients",
)
data class Patient(
    @PrimaryKey(autoGenerate = true)
    val participantId: Int? = null, // Auto-generated patient ID
    val firstName: String = "",
    val lastName: String = "",
    val age: Int? = null,
    val gender: String? = null,
    val height: Int, // Height in inches
    val createdAt: Long = System.currentTimeMillis()
) {
    // Full name property for convenience
    val fullName: String
        get() = if (firstName.isNotEmpty() || lastName.isNotEmpty()) {
            "$firstName $lastName".trim()
        } else {
            participantId?.toString() ?: "Unknown"
        }
}
