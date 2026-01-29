package GaitVision.com.mediapipe

import android.util.Log
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Converts MediaPipe PoseLandmarkerResult to our internal PoseFrame format.
 */
object MediaPipeResultConverter {
    
    private const val TAG = "MediaPipeResultConverter"
    
    /**
     * Convert MediaPipe result to PoseFrame.
     * 
     * MediaPipe API structure:
     * - result.landmarks() returns List<List<NormalizedLandmark>>
     * - First list = poses (one per person)
     * - Second list = landmarks for that pose (33 landmarks)
     * - Each landmark has x(), y(), z(), visibility() methods
     * 
     * @param result MediaPipe PoseLandmarkerResult
     * @param frameIdx Frame index in video
     * @param timestampS Frame timestamp in seconds
     * @return PoseFrame or null if no pose detected
     */
    fun toPoseFrame(
        result: PoseLandmarkerResult?,
        frameIdx: Int,
        timestampS: Float
    ): PoseFrame? {
        if (result == null) {
            return null
        }
        
        val landmarksList = result.landmarks()
        if (landmarksList.isEmpty()) {
            return null
        }
        
        // MediaPipe returns list of poses (multi-person support)
        // For gait analysis, we use the first (most confident) pose
        val landmarks = landmarksList.firstOrNull() ?: return null
        
        if (landmarks.size < 33) {
            Log.w(TAG, "Expected 33 landmarks but got ${landmarks.size}")
        }
        
        // MediaPipe landmarks are already normalized (0-1)
        // Extract 33 keypoints (MediaPipe Pose has 33 landmarks)
        val keypoints = Array(33) { FloatArray(2) }
        val confidences = FloatArray(33)
        
        for (i in landmarks.indices) {
            if (i >= 33) break // Safety check
            
            val landmark = landmarks[i]
            keypoints[i][0] = landmark.x()  // Normalized x (0-1)
            keypoints[i][1] = landmark.y()  // Normalized y (0-1)
            // visibility() returns Optional<Float>, use orElse to provide default
            confidences[i] = landmark.visibility().orElse(0f)
        }
        
        // Fill remaining slots with zeros if we got fewer than 33 landmarks
        for (i in landmarks.size until 33) {
            keypoints[i][0] = 0f
            keypoints[i][1] = 0f
            confidences[i] = 0f
        }
        
        // Get detection confidence from average visibility of core keypoints
        val coreIndices = listOf(
            MediaPipePoseBackend.LEFT_SHOULDER,
            MediaPipePoseBackend.RIGHT_SHOULDER,
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.RIGHT_ANKLE
        )
        val detectionConfidence = coreIndices
            .filter { it < landmarks.size }
            .map { confidences[it] }
            .average()
            .toFloat()
            .coerceIn(0f, 1f)
        
        return PoseFrame(
            frameIdx = frameIdx,
            timestampS = timestampS,
            keypoints = keypoints,
            confidences = confidences,
            detectionConfidence = detectionConfidence,
            isValid = true
        )
    }
}
