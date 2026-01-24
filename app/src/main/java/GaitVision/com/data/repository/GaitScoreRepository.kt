package GaitVision.com.data.repository

import GaitVision.com.data.GaitScore
import GaitVision.com.data.GaitScoreDao
import kotlinx.coroutines.flow.Flow

class GaitScoreRepository(private val gaitScoreDao: GaitScoreDao) {

    // Create operations
    suspend fun insertGaitScore(gaitScore: GaitScore): Long {
        return gaitScoreDao.insertGaitScore(gaitScore)
    }

    suspend fun insertGaitScores(gaitScores: List<GaitScore>): List<Long> {
        return gaitScoreDao.insertGaitScores(gaitScores)
    }

    // Read operations
    suspend fun getGaitScoreById(scoreId: Long): GaitScore? {
        return gaitScoreDao.getGaitScoreById(scoreId)
    }

    fun getGaitScoresByPatientId(patientId: Int): Flow<List<GaitScore>> {
        return gaitScoreDao.getGaitScoresByPatientId(patientId)
    }

    fun getGaitScoresByPatientIdOrdered(patientId: Int): Flow<List<GaitScore>> {
        return gaitScoreDao.getGaitScoresByPatientIdOrdered(patientId)
    }

    suspend fun getGaitScoreByVideoId(videoId: Long): GaitScore? {
        return gaitScoreDao.getGaitScoreByVideoId(videoId)
    }

    fun getAllGaitScores(): Flow<List<GaitScore>> {
        return gaitScoreDao.getAllGaitScores()
    }

    // Best/Worst score operations
    suspend fun getBestScoreForPatient(patientId: Int): GaitScore? {
        return gaitScoreDao.getBestScoreForPatient(patientId)
    }

    suspend fun getWorstScoreForPatient(patientId: Int): GaitScore? {
        return gaitScoreDao.getWorstScoreForPatient(patientId)
    }

    suspend fun getAverageScoreForPatient(patientId: Int): Double? {
        return gaitScoreDao.getAverageScoreForPatient(patientId)
    }

    // Update operations
    suspend fun updateGaitScore(gaitScore: GaitScore): Boolean {
        return gaitScoreDao.updateGaitScore(gaitScore) > 0
    }

    // Delete operations
    suspend fun deleteGaitScore(gaitScore: GaitScore): Boolean {
        return gaitScoreDao.deleteGaitScore(gaitScore) > 0
    }

    suspend fun deleteGaitScoreById(scoreId: Long): Boolean {
        return gaitScoreDao.deleteGaitScoreById(scoreId) > 0
    }

    suspend fun deleteGaitScoresByPatientId(patientId: Int): Boolean {
        return gaitScoreDao.deleteGaitScoresByPatientId(patientId) > 0
    }

    // Utility operations
    suspend fun getGaitScoreCount(): Int {
        return gaitScoreDao.getGaitScoreCount()
    }

    suspend fun getGaitScoreCountForPatient(patientId: Int): Int {
        return gaitScoreDao.getGaitScoreCountForPatient(patientId)
    }

    suspend fun getGaitScoreCountForVideo(videoId: Long): Int {
        return gaitScoreDao.getGaitScoreCountForVideo(videoId)
    }

    // Search operations
    fun searchGaitScores(searchQuery: String): Flow<List<GaitScore>> {
        val query = "%$searchQuery%"
        return gaitScoreDao.searchGaitScores(query)
    }

    // Business logic operations
    suspend fun gaitScoreExists(scoreId: Long): Boolean {
        return getGaitScoreById(scoreId) != null
    }

    suspend fun hasScoresForPatient(patientId: Int): Boolean {
        return getGaitScoreCountForPatient(patientId) > 0
    }

    suspend fun hasScoreForVideo(videoId: Long): Boolean {
        return getGaitScoreCountForVideo(videoId) > 0
    }
}
