package GaitVision.com.video

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import java.io.FileDescriptor

private const val TAG = "GaitMedia"

/**
 * Video metadata extracted during decode setup.
 */
data class VideoMetadata(
    val width: Int,
    val height: Int,
    val fps: Float,
    val durationUs: Long,
    val totalFrames: Int,
    val videoMime: String
)

/**
 * Convert YUV_420_888 Image (from MediaCodec) to ARGB Bitmap.
 * LOSSLESS direct YUV→RGB conversion. Uses reusable buffers to avoid allocations per frame.
 */
private var yBytesCache: ByteArray? = null
private var uBytesCache: ByteArray? = null
private var vBytesCache: ByteArray? = null
private var pixelsCache: IntArray? = null
private var bitmapCache: Bitmap? = null

internal fun imageToBitmap(image: Image): Bitmap {
    val width = image.width
    val height = image.height
    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]
    val yRowStride = yPlane.rowStride
    val uvRowStride = uPlane.rowStride
    val uvPixelStride = uPlane.pixelStride
    val yBuffer = yPlane.buffer.duplicate().apply { position(0) }
    val uBuffer = uPlane.buffer.duplicate().apply { position(0) }
    val vBuffer = vPlane.buffer.duplicate().apply { position(0) }
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()
    val pixelCount = width * height
    val yBytes = yBytesCache?.takeIf { it.size == ySize } ?: ByteArray(ySize).also { yBytesCache = it }
    val uBytes = uBytesCache?.takeIf { it.size == uSize } ?: ByteArray(uSize).also { uBytesCache = it }
    val vBytes = vBytesCache?.takeIf { it.size == vSize } ?: ByteArray(vSize).also { vBytesCache = it }
    val pixels = pixelsCache?.takeIf { it.size == pixelCount } ?: IntArray(pixelCount).also { pixelsCache = it }
    yBuffer.get(yBytes, 0, ySize)
    uBuffer.get(uBytes, 0, uSize)
    vBuffer.get(vBytes, 0, vSize)
    for (row in 0 until height) {
        val yRowOffset = row * yRowStride
        val uvRowOffset = (row shr 1) * uvRowStride
        val pixelRowOffset = row * width
        for (col in 0 until width) {
            val y = (yBytes[yRowOffset + col].toInt() and 0xFF)
            val uvIndex = uvRowOffset + (col shr 1) * uvPixelStride
            val u = (uBytes[uvIndex].toInt() and 0xFF) - 128
            val v = (vBytes[uvIndex].toInt() and 0xFF) - 128
            var r = (y + 1.370705 * v).toInt()
            var g = (y - 0.698001 * v - 0.337633 * u).toInt()
            var b = (y + 1.732446 * u).toInt()
            r = r.coerceIn(0, 255)
            g = g.coerceIn(0, 255)
            b = b.coerceIn(0, 255)
            pixels[pixelRowOffset + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    val bitmap = bitmapCache?.takeIf { it.width == width && it.height == height && !it.isRecycled }
        ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmapCache = it }
    bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
    return bitmap
}

/** Open a video URI via PFD, falling back to content resolver. */
fun setDataSourceSafe(context: Context, uri: Uri, vararg targets: Any) {
    try {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
        if (pfd != null) {
            for (t in targets) {
                when (t) {
                    is MediaExtractor -> t.setDataSource(pfd.fileDescriptor)
                    is MediaMetadataRetriever -> t.setDataSource(pfd.fileDescriptor)
                }
            }
            pfd.close()
            return
        }
    } catch (_: Exception) {}
    for (t in targets) {
        when (t) {
            is MediaExtractor -> t.setDataSource(context, uri, null)
            is MediaMetadataRetriever -> t.setDataSource(context, uri)
        }
    }
}

/** Release reusable YUV conversion buffers. */
fun releaseDecoderCaches() {
    bitmapCache?.recycle()
    bitmapCache = null
    yBytesCache = null
    uBytesCache = null
    vBytesCache = null
    pixelsCache = null
}

/**
 * Decode video frames using MediaCodec. Calls onMetadataReady before first frame, then onFrame for each frame.
 * Optional onProgress(totalFrames, currentIndex) for UI updates.
 * Returns VideoMetadata on success, null on failure.
 */
fun decodeFramesMediaCodec(
    context: Context,
    uri: Uri,
    onMetadataReady: (VideoMetadata) -> Unit,
    onFrame: (Bitmap, Int) -> Unit,
    onProgress: ((totalFrames: Int, currentIndex: Int) -> Unit)? = null
): VideoMetadata? {
    val extractor = MediaExtractor()
    val retriever = MediaMetadataRetriever()
    try {
        setDataSourceSafe(context, uri, extractor, retriever)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to open video: ${e.message}")
        return null
    }
    var videoTrackIndex = -1
    var videoFormat: MediaFormat? = null
    for (i in 0 until extractor.trackCount) {
        val format = extractor.getTrackFormat(i)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: ""
        if (mime.startsWith("video/")) {
            videoTrackIndex = i
            videoFormat = format
            break
        }
    }
    if (videoTrackIndex < 0 || videoFormat == null) {
        Log.e(TAG, "No video track found")
        extractor.release()
        retriever.release()
        return null
    }
    extractor.selectTrack(videoTrackIndex)
    val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
    val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
    val videoMime = videoFormat.getString(MediaFormat.KEY_MIME) ?: "video/avc"
    val durationUs = videoFormat.getLong(MediaFormat.KEY_DURATION)
    var fps = 30f
    if (videoFormat.containsKey(MediaFormat.KEY_FRAME_RATE)) {
        fps = videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE).toFloat()
    } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val videoLengthMs = durationUs / 1000
        val frameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
        if (frameCount != null && videoLengthMs > 0) {
            val calculatedFps = (frameCount * 1000f) / videoLengthMs
            if (calculatedFps in 15f..120f) fps = calculatedFps
        }
    }
    retriever.release()
    val totalFrames = ((durationUs * fps) / 1_000_000).toInt()
    val metadata = VideoMetadata(width, height, fps, durationUs, totalFrames, videoMime)

    val decoder: MediaCodec
    try {
        decoder = MediaCodec.createDecoderByType(videoMime)
        decoder.configure(videoFormat, null, null, 0)
        decoder.start()
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize decoder: ${e.message}")
        extractor.release()
        return null
    }

    onMetadataReady(metadata)
    val bufferInfo = MediaCodec.BufferInfo()
    var frameIndex = 0
    var inputDone = false
    var outputDone = false
    try {
        while (!outputDone) {
            if (!inputDone) {
                val inputBufferId = decoder.dequeueInputBuffer(10000)
                if (inputBufferId >= 0) {
                    val inputBuffer = decoder.getInputBuffer(inputBufferId)
                    if (inputBuffer != null) {
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            decoder.queueInputBuffer(inputBufferId, 0, sampleSize, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
            }
            val outputBufferId = decoder.dequeueOutputBuffer(bufferInfo, 10000)
            when {
                outputBufferId == MediaCodec.INFO_TRY_AGAIN_LATER -> {}
                outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {}
                outputBufferId >= 0 -> {
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        outputDone = true
                        decoder.releaseOutputBuffer(outputBufferId, false)
                    } else {
                        val image = decoder.getOutputImage(outputBufferId)
                        if (image != null) {
                            try {
                                val frame = imageToBitmap(image)
                                onFrame(frame, frameIndex)
                                onProgress?.invoke(totalFrames, frameIndex + 1)
                            } finally {
                                image.close()
                            }
                            decoder.releaseOutputBuffer(outputBufferId, false)
                            frameIndex++
                        } else {
                            decoder.releaseOutputBuffer(outputBufferId, false)
                        }
                    }
                }
            }
        }
    } finally {
        decoder.stop()
        decoder.release()
        extractor.release()
    }
    Log.d(TAG, "Decoded $frameIndex frames")
    return metadata
}

/**
 * Fallback: decode frames using getFrameAtTime (slower).
 */
fun decodeFramesGetFrameAtTime(
    context: Context,
    uri: Uri,
    onMetadataReady: (VideoMetadata) -> Unit,
    onFrame: (Bitmap, Int) -> Unit
): VideoMetadata? {
    val retriever = MediaMetadataRetriever()
    setDataSourceSafe(context, uri, retriever)
    val videoLengthMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
    val videoLengthUs = videoLengthMs * 1000L
    var fps = 30f
    val fpsStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
    fpsStr?.toFloatOrNull()?.let { if (it in 15f..120f) fps = it }
    if (fps == 30f && Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val frameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
        if (frameCount != null && videoLengthMs > 0) {
            val calculatedFps = (frameCount * 1000f) / videoLengthMs
            if (calculatedFps in 15f..120f) fps = calculatedFps
        }
    }
    val frameIntervalUs = (1000000L / fps).toLong()
    val totalFrames = (videoLengthUs / frameIntervalUs).toInt()
    val firstFrame = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)
    if (firstFrame == null) {
        retriever.release()
        return null
    }
    val width = firstFrame.width
    val height = firstFrame.height
    val metadata = VideoMetadata(width, height, fps, videoLengthUs, totalFrames, "video/avc")
    onMetadataReady(metadata)
    var frameIndex = 0
    var currTimeUs = 0L
    while (currTimeUs <= videoLengthUs) {
        val frame = retriever.getFrameAtTime(currTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        if (frame != null) {
            onFrame(frame, frameIndex)
            frameIndex++
        }
        currTimeUs += frameIntervalUs
    }
    retriever.release()
    return metadata
}

/**
 * Detect actual FPS from video metadata. Falls back to 30 FPS if detection fails.
 */
fun detectVideoFps(context: Context, uri: Uri?): Float {
    if (uri == null) return 30f
    val retriever = MediaMetadataRetriever()
    return try {
        setDataSourceSafe(context, uri, retriever)
        val frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
        val fps = frameRateStr?.toFloatOrNull()
        if (fps != null && fps > 0) {
            Log.d(TAG, "Detected FPS from metadata: $fps")
            fps
        } else {
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val videoFrameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            if (videoFrameCount != null && videoFrameCount > 0 && durationMs > 0) {
                val estimatedFps = (videoFrameCount * 1000f) / durationMs
                Log.d(TAG, "Estimated FPS from frame count: $estimatedFps")
                estimatedFps.coerceIn(15f, 120f)
            } else {
                Log.d(TAG, "Could not detect FPS, using default 30")
                30f
            }
        }
    } catch (e: Exception) {
        Log.w(TAG, "Error detecting FPS: ${e.message}, using default 30")
        30f
    } finally {
        retriever.release()
    }
}
