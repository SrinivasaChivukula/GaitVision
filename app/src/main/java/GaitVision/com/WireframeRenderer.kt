package GaitVision.com

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.util.Log

import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame

/**
 * Renders skeleton wireframe overlay on video frames and calculates joint angles.
 * 
 * Extracted from imageProcessing.kt for better organization.
 * Mutates global angle lists (leftAnkleAngles, rightAnkleAngles, etc.) as a side effect.
 */

/**
 * Draw skeleton overlay and calculate angles from MediaPipe pose frame.
 * 
 * MediaPipe returns normalized coordinates (0-1), which we convert to pixel
 * coordinates for drawing. Angle calculations use the same formulas as before.
 * 
 * SIDE EFFECTS: Adds calculated angles to global angle lists in globalVariables.kt
 * 
 * @param bitmap Frame to draw on (modified in place)
 * @param poseFrame Pose detection result with normalized coordinates
 * @return The modified bitmap with skeleton overlay
 */
fun drawOnBitmapMediaPipe(bitmap: Bitmap, poseFrame: PoseFrame?): Bitmap {
    if (poseFrame == null) {
        return bitmap
    }
    
    val width = bitmap.width.toFloat()
    val height = bitmap.height.toFloat()
    val keypoints = poseFrame.keypoints
    val confidences = poseFrame.confidences
    
    // Helper to get pixel coordinates from normalized keypoint
    fun getPixelCoords(landmarkIdx: Int): Pair<Float, Float> {
        val x = keypoints[landmarkIdx][0] * width
        val y = keypoints[landmarkIdx][1] * height
        return Pair(x, y)
    }
    
    // Get coordinates for all landmarks we need
    val (leftShoulderX, leftShoulderY) = getPixelCoords(MediaPipePoseBackend.LEFT_SHOULDER)
    val (rightShoulderX, rightShoulderY) = getPixelCoords(MediaPipePoseBackend.RIGHT_SHOULDER)
    val (leftHipX, leftHipY) = getPixelCoords(MediaPipePoseBackend.LEFT_HIP)
    val (rightHipX, rightHipY) = getPixelCoords(MediaPipePoseBackend.RIGHT_HIP)
    val (leftKneeX, leftKneeY) = getPixelCoords(MediaPipePoseBackend.LEFT_KNEE)
    val (rightKneeX, rightKneeY) = getPixelCoords(MediaPipePoseBackend.RIGHT_KNEE)
    val (leftAnkleX, leftAnkleY) = getPixelCoords(MediaPipePoseBackend.LEFT_ANKLE)
    val (rightAnkleX, rightAnkleY) = getPixelCoords(MediaPipePoseBackend.RIGHT_ANKLE)
    val (leftHeelX, leftHeelY) = getPixelCoords(MediaPipePoseBackend.LEFT_HEEL)
    val (rightHeelX, rightHeelY) = getPixelCoords(MediaPipePoseBackend.RIGHT_HEEL)
    val (leftFootIndexX, leftFootIndexY) = getPixelCoords(MediaPipePoseBackend.LEFT_FOOT_INDEX)
    val (rightFootIndexX, rightFootIndexY) = getPixelCoords(MediaPipePoseBackend.RIGHT_FOOT_INDEX)
    
    // Angle Calculations (same formulas as before)
    // Ankle Angles
    val leftAnkleAngle = GetAnglesA(leftFootIndexX, leftFootIndexY, leftAnkleX, leftAnkleY, leftKneeX, leftKneeY)
    if (!leftAnkleAngle.isNaN() && leftAnkleAngle < 70 && leftAnkleAngle > -25) {
        leftAnkleAngles.add(leftAnkleAngle)
    } else {
        count++
        Log.d("ErrorCheck", "Left Ankle: $leftAnkleAngle")
    }
    
    val rightAnkleAngle = GetAnglesA(rightFootIndexX, rightFootIndexY, rightAnkleX, rightAnkleY, rightKneeX, rightKneeY)
    if (!rightAnkleAngle.isNaN() && rightAnkleAngle < 70 && rightAnkleAngle > -25) {
        rightAnkleAngles.add(rightAnkleAngle)
    } else {
        count++
        Log.d("ErrorCheck", "Right Ankle: $rightAnkleAngle")
    }

    // Knee Angles
    val leftKneeAngle = GetAngles(leftAnkleX, leftAnkleY, leftKneeX, leftKneeY, leftHipX, leftHipY)
    if (!leftKneeAngle.isNaN()) {
        leftKneeAngles.add(leftKneeAngle)
    } else {
        count++
    }
    
    val rightKneeAngle = GetAngles(rightAnkleX, rightAnkleY, rightKneeX, rightKneeY, rightHipX, rightHipY)
    if (!rightKneeAngle.isNaN()) {
        rightKneeAngles.add(rightKneeAngle)
    } else {
        count++
    }

    // Hip Angles
    val leftHipAngle = GetAngles(leftKneeX, leftKneeY, leftHipX, leftHipY, leftShoulderX, leftShoulderY)
    if (!leftHipAngle.isNaN()) {
        leftHipAngles.add(leftHipAngle)
    } else {
        count++
    }
    
    val rightHipAngle = GetAngles(rightKneeX, rightKneeY, rightHipX, rightHipY, rightShoulderX, rightShoulderY)
    if (!rightHipAngle.isNaN()) {
        rightHipAngles.add(rightHipAngle)
    } else {
        count++
    }

    // Torso Angle
    val torsoAngle = calcTorso(
        (leftHipX + rightHipX) / 2, (leftHipY + rightHipY) / 2,
        (rightShoulderX + leftShoulderX) / 2, (rightShoulderY + leftShoulderY) / 2
    )
    if (!torsoAngle.isNaN() && torsoAngle > -20 && torsoAngle < 20) {
        torsoAngles.add(torsoAngle)
    } else {
        count++
        Log.d("ErrorCheck", "TorsoAngle: $torsoAngle")
    }

    // Stride angle
    val strideAngle = calcStrideAngle(
        leftHeelX, leftHeelY,
        (leftHipX + rightHipX) / 2f, (leftHipY + rightHipY) / 2,
        rightHeelX, rightHeelY
    )
    strideAngles.add(strideAngle)

    // Draw skeleton overlay
    val canvas = Canvas(bitmap)

    val paintCircleRight = Paint().apply { setARGB(255, 255, 0, 0) }
    val paintCircleLeft = Paint().apply { setARGB(255, 0, 0, 255) }
    val paintLine = Paint().apply {
        setARGB(255, 255, 255, 255)
        strokeWidth = 4f
    }

    // Draw connections
    canvas.drawLine(rightHipX, rightHipY, rightKneeX, rightKneeY, paintLine)
    canvas.drawLine(leftHipX, leftHipY, leftKneeX, leftKneeY, paintLine)
    canvas.drawLine(rightKneeX, rightKneeY, rightAnkleX, rightAnkleY, paintLine)
    canvas.drawLine(leftKneeX, leftKneeY, leftAnkleX, leftAnkleY, paintLine)
    canvas.drawLine(rightAnkleX, rightAnkleY, rightFootIndexX, rightFootIndexY, paintLine)
    canvas.drawLine(leftAnkleX, leftAnkleY, leftFootIndexX, leftFootIndexY, paintLine)
    canvas.drawLine(rightAnkleX, rightAnkleY, rightHeelX, rightHeelY, paintLine)
    canvas.drawLine(leftAnkleX, leftAnkleY, leftHeelX, leftHeelY, paintLine)
    canvas.drawLine(rightHeelX, rightHeelY, rightFootIndexX, rightFootIndexY, paintLine)
    canvas.drawLine(leftHeelX, leftHeelY, leftFootIndexX, leftFootIndexY, paintLine)

    // Draw points
    canvas.drawCircle(rightHipX, rightHipY, 4f, paintCircleRight)
    canvas.drawCircle(leftHipX, leftHipY, 4f, paintCircleLeft)
    canvas.drawCircle(rightKneeX, rightKneeY, 4f, paintCircleRight)
    canvas.drawCircle(leftKneeX, leftKneeY, 4f, paintCircleLeft)
    canvas.drawCircle(rightAnkleX, rightAnkleY, 4f, paintCircleRight)
    canvas.drawCircle(leftAnkleX, leftAnkleY, 4f, paintCircleLeft)
    canvas.drawCircle(rightHeelX, rightHeelY, 4f, paintCircleRight)
    canvas.drawCircle(leftHeelX, leftHeelY, 4f, paintCircleLeft)
    canvas.drawCircle(rightFootIndexX, rightFootIndexY, 4f, paintCircleRight)
    canvas.drawCircle(leftFootIndexX, leftFootIndexY, 4f, paintCircleLeft)

    return bitmap
}
