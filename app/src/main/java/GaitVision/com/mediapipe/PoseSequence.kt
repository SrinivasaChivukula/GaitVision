package GaitVision.com.mediapipe

/**
 * Data model for pose sequence from a video.
 * Mirrors PC PoseSequence dataclass for feature parity.
 */
data class PoseSequence(
    val videoId: String,
    val fps: Float,
    val frameWidth: Int,
    val frameHeight: Int,
    val numFramesTotal: Int,
    val frames: List<PoseFrame> = emptyList(),
    val walkingDirection: String = "left_to_right",  // Before any flip
    val wasFlipped: Boolean = false
) {
    val detectionRate: Float
        get() = if (numFramesTotal == 0) 0.0f else frames.size.toFloat() / numFramesTotal
    
    val durationS: Float
        get() = frames.lastOrNull()?.timestampS ?: 0.0f
}

/**
 * Single frame of pose data.
 * Mirrors PC PoseFrame dataclass.
 */
data class PoseFrame(
    val frameIdx: Int,
    val timestampS: Float,
    val keypoints: Array<FloatArray>,  // Shape (33, 2) - x_norm, y_norm
    val confidences: FloatArray,      // Shape (33,) - visibility scores
    val detectionConfidence: Float = 1.0f,
    val isValid: Boolean = true
) {
    // Override equals/hashCode to properly compare arrays
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        
        other as PoseFrame
        
        if (frameIdx != other.frameIdx) return false
        if (timestampS != other.timestampS) return false
        if (!keypoints.contentDeepEquals(other.keypoints)) return false
        if (!confidences.contentEquals(other.confidences)) return false
        if (detectionConfidence != other.detectionConfidence) return false
        if (isValid != other.isValid) return false
        
        return true
    }
    
    override fun hashCode(): Int {
        var result = frameIdx
        result = 31 * result + timestampS.hashCode()
        result = 31 * result + keypoints.contentDeepHashCode()
        result = 31 * result + confidences.contentHashCode()
        result = 31 * result + detectionConfidence.hashCode()
        result = 31 * result + isValid.hashCode()
        return result
    }
}
