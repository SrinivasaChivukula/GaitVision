package GaitVision.com

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint

import GaitVision.com.gait.GaitConfig
import GaitVision.com.mediapipe.MediaPipePoseBackend
import GaitVision.com.mediapipe.PoseFrame

import android.graphics.Color
import android.graphics.RectF
import android.graphics.Typeface

/**
 * Renders skeleton wireframe overlay on video frames.
 * 
 * Only draws the skeleton - angle calculations are now handled by FeatureExtractor
 * and stored in Signals object for single source of truth.
 */

/**
 * Draw skeleton overlay from MediaPipe pose frame.
 * 
 * MediaPipe returns normalized coordinates (0-1), which we convert to pixel
 * coordinates for drawing.
 * 
 * @param bitmap Frame to draw on (modified in place)
 * @param poseFrame Pose detection result with normalized coordinates
 * @return The modified bitmap with skeleton overlay
 */
fun drawOnBitmapMediaPipe(bitmap: Bitmap, poseFrame: PoseFrame?, showLabels: Boolean = true): Bitmap {
    if (poseFrame == null) {
        return bitmap
    }
    
    val width = bitmap.width.toFloat()
    val height = bitmap.height.toFloat()
    val keypoints = poseFrame.keypoints

    val minDimension = minOf(width, height)
    val confidences = poseFrame.confidences
    val minConfidence = GaitConfig.MIN_CONFIDENCE

    val leftColor = Color.rgb(0, 122, 255)
    val rightColor = Color.rgb(255, 59, 48)

    val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(190, 0, 0, 0)
        style = Paint.Style.FILL
    }

    val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = (minDimension * 0.028f).coerceIn(18f, 34f)
        typeface = Typeface.DEFAULT_BOLD
    }
    fun hasPoint(landmarkIdx: Int): Boolean {
        val point = keypoints.getOrNull(landmarkIdx) ?: return false
        if (point.size < 2 || point[0].isNaN() || point[1].isNaN()) return false
        return confidences.getOrNull(landmarkIdx)?.let { it >= minConfidence } == true
    }

    fun getPixelCoords(landmarkIdx: Int): Pair<Float, Float> {
        val point = keypoints[landmarkIdx]
        return Pair(point[0] * width, point[1] * height)
    }

    // Draw skeleton overlay
    val canvas = Canvas(bitmap)

    val paintCircleRight = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = rightColor }
    val paintCircleLeft = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = leftColor }
    val paintLineRight = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = rightColor
        strokeWidth = 4f
        strokeCap = Paint.Cap.ROUND
    }
    val paintLineLeft = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = leftColor
        strokeWidth = 4f
        strokeCap = Paint.Cap.ROUND
    }

    fun drawBone(startIdx: Int, endIdx: Int, paint: Paint) {
        if (!hasPoint(startIdx) || !hasPoint(endIdx)) return
        val (startX, startY) = getPixelCoords(startIdx)
        val (endX, endY) = getPixelCoords(endIdx)
        canvas.drawLine(startX, startY, endX, endY, paint)
    }

    fun drawPoint(landmarkIdx: Int, paint: Paint) {
        if (!hasPoint(landmarkIdx)) return
        val (x, y) = getPixelCoords(landmarkIdx)
        canvas.drawCircle(x, y, 4f, paint)
    }

    drawBone(MediaPipePoseBackend.LEFT_HIP, MediaPipePoseBackend.LEFT_KNEE, paintLineLeft)
    drawBone(MediaPipePoseBackend.LEFT_KNEE, MediaPipePoseBackend.LEFT_ANKLE, paintLineLeft)
    drawBone(MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.LEFT_FOOT_INDEX, paintLineLeft)
    drawBone(MediaPipePoseBackend.LEFT_ANKLE, MediaPipePoseBackend.LEFT_HEEL, paintLineLeft)
    drawBone(MediaPipePoseBackend.LEFT_HEEL, MediaPipePoseBackend.LEFT_FOOT_INDEX, paintLineLeft)

    drawBone(MediaPipePoseBackend.RIGHT_HIP, MediaPipePoseBackend.RIGHT_KNEE, paintLineRight)
    drawBone(MediaPipePoseBackend.RIGHT_KNEE, MediaPipePoseBackend.RIGHT_ANKLE, paintLineRight)
    drawBone(MediaPipePoseBackend.RIGHT_ANKLE, MediaPipePoseBackend.RIGHT_FOOT_INDEX, paintLineRight)
    drawBone(MediaPipePoseBackend.RIGHT_ANKLE, MediaPipePoseBackend.RIGHT_HEEL, paintLineRight)
    drawBone(MediaPipePoseBackend.RIGHT_HEEL, MediaPipePoseBackend.RIGHT_FOOT_INDEX, paintLineRight)

    drawPoint(MediaPipePoseBackend.LEFT_HIP, paintCircleLeft)
    drawPoint(MediaPipePoseBackend.LEFT_KNEE, paintCircleLeft)
    drawPoint(MediaPipePoseBackend.LEFT_ANKLE, paintCircleLeft)
    drawPoint(MediaPipePoseBackend.LEFT_HEEL, paintCircleLeft)
    drawPoint(MediaPipePoseBackend.LEFT_FOOT_INDEX, paintCircleLeft)

    drawPoint(MediaPipePoseBackend.RIGHT_HIP, paintCircleRight)
    drawPoint(MediaPipePoseBackend.RIGHT_KNEE, paintCircleRight)
    drawPoint(MediaPipePoseBackend.RIGHT_ANKLE, paintCircleRight)
    drawPoint(MediaPipePoseBackend.RIGHT_HEEL, paintCircleRight)
    drawPoint(MediaPipePoseBackend.RIGHT_FOOT_INDEX, paintCircleRight)

    fun drawLabel(text: String, x: Float, y: Float, color: Int) {
        labelTextPaint.color = color

        val paddingX = 8f
        val paddingY = 5f
        val textWidth = labelTextPaint.measureText(text)

        val labelX = x.coerceIn(4f, width - textWidth - 12f)
        val baseline = y.coerceIn(20f, height - 8f)

        val rect = RectF(
            labelX - paddingX,
            baseline + labelTextPaint.ascent() - paddingY,
            labelX + textWidth + paddingX,
            baseline + labelTextPaint.descent() + paddingY
        )

        canvas.drawRoundRect(rect, 6f, 6f, labelBackgroundPaint)
        canvas.drawText(text, labelX, baseline, labelTextPaint)
    }

    if (showLabels) {
        if (hasPoint(MediaPipePoseBackend.LEFT_KNEE)) {
            val (leftKneeX, leftKneeY) = getPixelCoords(MediaPipePoseBackend.LEFT_KNEE)
            drawLabel("LK", leftKneeX - 30f, leftKneeY - 10f, leftColor)
        }
        if (hasPoint(MediaPipePoseBackend.RIGHT_KNEE)) {
            val (rightKneeX, rightKneeY) = getPixelCoords(MediaPipePoseBackend.RIGHT_KNEE)
            drawLabel("RK", rightKneeX + 10f, rightKneeY - 10f, rightColor)
        }
        if (
            hasPoint(MediaPipePoseBackend.LEFT_HIP) &&
            hasPoint(MediaPipePoseBackend.LEFT_KNEE) &&
            hasPoint(MediaPipePoseBackend.LEFT_ANKLE)
        ) {
            val (leftHipX, leftHipY) = getPixelCoords(MediaPipePoseBackend.LEFT_HIP)
            val (leftAnkleX, leftAnkleY) = getPixelCoords(MediaPipePoseBackend.LEFT_ANKLE)
            drawLabel("L Leg", (leftHipX + leftAnkleX) / 2f, (leftHipY + leftAnkleY) / 2f, leftColor)
        }
        if (
            hasPoint(MediaPipePoseBackend.RIGHT_HIP) &&
            hasPoint(MediaPipePoseBackend.RIGHT_KNEE) &&
            hasPoint(MediaPipePoseBackend.RIGHT_ANKLE)
        ) {
            val (rightHipX, rightHipY) = getPixelCoords(MediaPipePoseBackend.RIGHT_HIP)
            val (rightAnkleX, rightAnkleY) = getPixelCoords(MediaPipePoseBackend.RIGHT_ANKLE)
            drawLabel("R Leg", (rightHipX + rightAnkleX) / 2f, (rightHipY + rightAnkleY) / 2f, rightColor)
        }
    }

    return bitmap
}
