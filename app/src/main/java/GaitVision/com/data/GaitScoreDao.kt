package GaitVision.com.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface GaitScoreDao {

    // Create
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertGaitScore(gaitScore: GaitScore): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertGaitScores(gaitScores: List<GaitScore>): List<Long>

    // Read
    @Query("SELECT * FROM gait_scores WHERE id = :scoreId")
    suspend fun getGaitScoreById(scoreId: Long): GaitScore?

    @Query("SELECT * FROM gait_scores WHERE patientId = :patientId")
    fun getGaitScoresByPatientId(patientId: Int): Flow<List<GaitScore>>

    @Query("SELECT * FROM gait_scores WHERE patientId = :patientId ORDER BY recordedAt DESC")
    fun getGaitScoresByPatientIdOrdered(patientId: Long): Flow<List<GaitScore>>

    @Query("SELECT * FROM gait_scores WHERE videoId = :videoId")
    suspend fun getGaitScoreByVideoId(videoId: Long): GaitScore?

    @Query("SELECT * FROM gait_scores WHERE patientId = :patientId ORDER BY overallScore DESC LIMIT 1")
    suspend fun getBestScoreForPatient(patientId: Int): GaitScore?

    @Query("SELECT * FROM gait_scores WHERE patientId = :patientId ORDER BY overallScore ASC LIMIT 1")
    suspend fun getWorstScoreForPatient(patientId: Int): GaitScore?

    @Query("SELECT AVG(overallScore) FROM gait_scores WHERE patientId = :patientId")
    suspend fun getAverageScoreForPatient(patientId: Int): Double?

    @Query("SELECT * FROM gait_scores")
    fun getAllGaitScores(): Flow<List<GaitScore>>

    // Update
    @Update
    suspend fun updateGaitScore(gaitScore: GaitScore): Int

    // Delete
    @Delete
    suspend fun deleteGaitScore(gaitScore: GaitScore): Int

    @Query("DELETE FROM gait_scores WHERE id = :scoreId")
    suspend fun deleteGaitScoreById(scoreId: Long): Int

    @Query("DELETE FROM gait_scores WHERE patientId = :patientId")
    suspend fun deleteGaitScoresByPatientId(patientId: Int): Int

    // Utility
    @Query("SELECT COUNT(*) FROM gait_scores")
    suspend fun getGaitScoreCount(): Int

    @Query("SELECT COUNT(*) FROM gait_scores WHERE patientId = :patientId")
    suspend fun getGaitScoreCountForPatient(patientId: Long): Int

    @Query("SELECT COUNT(*) FROM gait_scores WHERE videoId = :videoId")
    suspend fun getGaitScoreCountForVideo(videoId: Long): Int

    // Relations - Get patient with their scores
    @Query("""
        SELECT * FROM gait_scores
        WHERE patientId = :patientId
        ORDER BY recordedAt DESC
    """)
    fun getGaitScoresWithPatientInfo(patientId: Int): Flow<List<GaitScore>>

    // Search
    @Query("""
        SELECT gs.* FROM gait_scores gs
        INNER JOIN patients p ON gs.patientId = p.participantId
        WHERE p.firstName LIKE :search OR p.lastName LIKE :search
        ORDER BY gs.recordedAt DESC
    """)
    fun searchGaitScores(search: String): Flow<List<GaitScore>>
}
