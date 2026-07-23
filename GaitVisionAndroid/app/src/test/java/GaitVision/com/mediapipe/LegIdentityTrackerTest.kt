package GaitVision.com.mediapipe

import org.junit.Assert.assertTrue
import org.junit.Test

class LegIdentityTrackerTest {
    @Test
    fun persistentSwapIsConfirmedAgainstLastAcceptedIdentity() {
        val tracker = LegIdentityTracker(
            swapVotesRequired = 3,
            jumpDistanceThreshold = 10f,
        )

        tracker.correct(frame(0, leftX = 0.25f, rightX = 0.75f))

        tracker.correct(frame(1, leftX = 0.26f, rightX = 0.76f, mediaPipeLabelsSwapped = true))
        tracker.correct(frame(2, leftX = 0.27f, rightX = 0.77f, mediaPipeLabelsSwapped = true))
        val corrected = tracker.correct(frame(3, leftX = 0.28f, rightX = 0.78f, mediaPipeLabelsSwapped = true))
        val nextCorrected = tracker.correct(frame(4, leftX = 0.29f, rightX = 0.79f, mediaPipeLabelsSwapped = true))

        assertLegsInOriginalOrder(corrected)
        assertLegsInOriginalOrder(nextCorrected)
    }

    @Test
    fun mutatingReturnedSwappedFrameDoesNotCorruptTrackerPrevious() {
        val tracker = LegIdentityTracker(
            swapVotesRequired = 1,
            jumpDistanceThreshold = 10f,
        )

        tracker.correct(frame(0, leftX = 0.25f, rightX = 0.75f))
        val corrected = tracker.correct(frame(1, leftX = 0.26f, rightX = 0.76f, mediaPipeLabelsSwapped = true))
        assertLegsInOriginalOrder(corrected)

        for (idx in legIndices) {
            corrected.keypoints[idx][0] = 0.99f
            corrected.keypoints[idx][1] = 0.99f
            corrected.confidences[idx] = 0f
        }

        val nextCorrected = tracker.correct(frame(2, leftX = 0.27f, rightX = 0.77f, mediaPipeLabelsSwapped = true))
        assertLegsInOriginalOrder(nextCorrected)
    }

    private fun frame(
        frameIdx: Int,
        leftX: Float,
        rightX: Float,
        mediaPipeLabelsSwapped: Boolean = false,
    ): PoseFrame {
        val keypoints = Array(33) { floatArrayOf(0f, 0f) }
        val confidences = FloatArray(33)
        val labeledLeftX = if (mediaPipeLabelsSwapped) rightX else leftX
        val labeledRightX = if (mediaPipeLabelsSwapped) leftX else rightX

        leftLegIndices.forEachIndexed { offset, idx ->
            keypoints[idx] = floatArrayOf(labeledLeftX, 0.4f + offset * 0.01f)
            confidences[idx] = 1f
        }
        rightLegIndices.forEachIndexed { offset, idx ->
            keypoints[idx] = floatArrayOf(labeledRightX, 0.4f + offset * 0.01f)
            confidences[idx] = 1f
        }

        return PoseFrame(
            frameIdx = frameIdx,
            timestampS = frameIdx / 30f,
            keypoints = keypoints,
            confidences = confidences,
        )
    }

    private fun assertLegsInOriginalOrder(frame: PoseFrame) {
        assertTrue(
            "expected left leg slot to contain the lower-x leg after correction",
            frame.keypoints[MediaPipePoseBackend.LEFT_ANKLE][0] <
                frame.keypoints[MediaPipePoseBackend.RIGHT_ANKLE][0],
        )
    }

    private companion object {
        val leftLegIndices = intArrayOf(
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.LEFT_HEEL,
            MediaPipePoseBackend.LEFT_FOOT_INDEX,
        )
        val rightLegIndices = intArrayOf(
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.RIGHT_ANKLE,
            MediaPipePoseBackend.RIGHT_HEEL,
            MediaPipePoseBackend.RIGHT_FOOT_INDEX,
        )
        val legIndices = leftLegIndices + rightLegIndices
    }
}
