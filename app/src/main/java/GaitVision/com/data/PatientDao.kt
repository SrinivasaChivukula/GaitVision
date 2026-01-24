package GaitVision.com.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface PatientDao {

    // Create
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPatient(patient: Patient): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPatients(patients: List<Patient>): List<Long>

    // Read
    @Query("SELECT * FROM patients WHERE participantId = :patientId")
    suspend fun getPatientById(patientId: Int): Patient?

    @Query("SELECT * FROM patients")
    fun getAllPatients(): Flow<List<Patient>>

    @Query("SELECT * FROM patients WHERE firstName LIKE :search OR lastName LIKE :search")
    fun searchPatients(search: String): Flow<List<Patient>>

    @Query("SELECT * FROM patients WHERE participantId = :participantId LIMIT 1")
    suspend fun getPatientByParticipantId(participantId: Int): Patient?

    // Update
    @Update
    suspend fun updatePatient(patient: Patient): Int

    // Delete
    @Delete
    suspend fun deletePatient(patient: Patient): Int

    @Query("DELETE FROM patients WHERE participantId = :patientId")
    suspend fun deletePatientById(patientId: Int): Int

    // Utility
    @Query("SELECT COUNT(*) FROM patients")
    suspend fun getPatientCount(): Int

    @Query("SELECT * FROM patients ORDER BY lastName, firstName")
    fun getPatientsOrderedByName(): Flow<List<Patient>>
}
