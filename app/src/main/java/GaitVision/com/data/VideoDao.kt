package GaitVision.com.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface VideoDao {

    // Create
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertVideo(video: Video): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertVideos(videos: List<Video>): List<Long>

    // Read
    @Query("SELECT * FROM videos WHERE id = :videoId")
    suspend fun getVideoById(videoId: Long): Video?

    @Query("SELECT * FROM videos WHERE patientId = :patientId AND originalVideoPath = :originalPath LIMIT 1")
    suspend fun getVideoByPatientAndPath(patientId: Int, originalPath: String): Video?

    @Query("SELECT * FROM videos WHERE patientId = :patientId")
    fun getVideosByPatientId(patientId: Long): Flow<List<Video>>

    @Query("SELECT * FROM videos WHERE patientId = :patientId ORDER BY recordedAt DESC")
    fun getVideosByPatientIdOrdered(patientId: Int): Flow<List<Video>>

    @Query("SELECT * FROM videos")
    fun getAllVideos(): Flow<List<Video>>

    // Update
    @Update
    suspend fun updateVideo(video: Video): Int

    // Delete
    @Delete
    suspend fun deleteVideo(video: Video): Int

    @Query("DELETE FROM videos WHERE id = :videoId")
    suspend fun deleteVideoById(videoId: Long): Int

    @Query("DELETE FROM videos WHERE patientId = :patientId")
    suspend fun deleteVideosByPatientId(patientId: Int): Int

    // Utility
    @Query("SELECT COUNT(*) FROM videos")
    suspend fun getVideoCount(): Int

    @Query("SELECT COUNT(*) FROM videos WHERE patientId = :patientId")
    suspend fun getVideoCountForPatient(patientId: Int): Int

    // Relations
    @Query("""
        SELECT v.* FROM videos v
        INNER JOIN patients p ON v.patientId = p.participantId
        WHERE p.firstName LIKE :search OR p.lastName LIKE :search OR v.originalVideoPath LIKE :search
        ORDER BY v.recordedAt DESC
    """)
    fun searchVideos(search: String): Flow<List<Video>>
}
