package GaitVision.com.data.repository

import GaitVision.com.data.Video
import GaitVision.com.data.VideoDao
import kotlinx.coroutines.flow.Flow

class VideoRepository(private val videoDao: VideoDao) {

    // Create
    suspend fun insertVideo(video: Video): Long {
        return videoDao.insertVideo(video)
    }

    suspend fun insertVideos(videos: List<Video>): List<Long> {
        return videoDao.insertVideos(videos)
    }

    // Read
    suspend fun getVideoById(videoId: Long): Video? {
        return videoDao.getVideoById(videoId)
    }

    fun getVideosByPatientId(patientId: Int): Flow<List<Video>> {
        return videoDao.getVideosByPatientId(patientId)
    }

    fun getVideosByPatientIdOrdered(patientId: Int): Flow<List<Video>> {
        return videoDao.getVideosByPatientIdOrdered(patientId)
    }

    fun getAllVideos(): Flow<List<Video>> {
        return videoDao.getAllVideos()
    }

    // Update
    suspend fun updateVideo(video: Video): Boolean {
        return videoDao.updateVideo(video) > 0
    }

    // Delete
    suspend fun deleteVideo(video: Video): Boolean {
        return videoDao.deleteVideo(video) > 0
    }

    suspend fun deleteVideoById(videoId: Long): Boolean {
        return videoDao.deleteVideoById(videoId) > 0
    }

    suspend fun deleteVideosByPatientId(patientId: Int): Boolean {
        return videoDao.deleteVideosByPatientId(patientId) > 0
    }

    // Utility
    suspend fun getVideoCount(): Int {
        return videoDao.getVideoCount()
    }

    suspend fun getVideoCountForPatient(patientId: Int): Int {
        return videoDao.getVideoCountForPatient(patientId)
    }

    // Search
    fun searchVideos(searchQuery: String): Flow<List<Video>> {
        val query = "%$searchQuery%"
        return videoDao.searchVideos(query)
    }

    // logic
    suspend fun videoExists(videoId: Long): Boolean {
        return getVideoById(videoId) != null
    }

    suspend fun hasVideosForPatient(patientId: Int): Boolean {
        return getVideoCountForPatient(patientId) > 0
    }

    // Get the most recent video for a patient
    suspend fun getLatestVideoForPatient(patientId: Int): Video? {
        return videoDao.getVideosByPatientIdOrdered(patientId).let { flow ->
            val videos = mutableListOf<Video>()
            flow.collect { videos.addAll(it) }
            return videos.firstOrNull()
        }
    }
}
