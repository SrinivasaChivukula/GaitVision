package GaitVision.com.video

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.view.Surface

class EncoderState(
    val encoder: MediaCodec,
    val mediaMuxer: MediaMuxer,
    val inputSurface: Surface,
    val bufferInfo: MediaCodec.BufferInfo = MediaCodec.BufferInfo(),
    val frameDurationUs: Long
) {
    var trackIndex: Int = -1
    var muxerStarted: Boolean = false

    fun encodeFrame(bitmap: Bitmap, frameIndex: Int) {
        val canvas = inputSurface.lockCanvas(null)
        canvas.drawBitmap(bitmap, 0f, 0f, null)
        inputSurface.unlockCanvasAndPost(canvas)
        drainEncoder(frameIndex)
    }

    fun drainEncoder(frameIndex: Int, timeout: Long = 1000) {
        while (true) {
            val outputId = encoder.dequeueOutputBuffer(bufferInfo, timeout)
            when {
                outputId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!muxerStarted) {
                        trackIndex = mediaMuxer.addTrack(encoder.outputFormat)
                        mediaMuxer.setOrientationHint(0)
                        mediaMuxer.start()
                        muxerStarted = true
                    }
                }
                outputId >= 0 -> {
                    val outputBuffer = encoder.getOutputBuffer(outputId) ?: break
                    if (muxerStarted) {
                        bufferInfo.presentationTimeUs = frameIndex * frameDurationUs
                        mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                    }
                    encoder.releaseOutputBuffer(outputId, false)
                }
                else -> break
            }
        }
    }

    fun finishEncoding() {
        encoder.signalEndOfInputStream()
        while (true) {
            val outputId = encoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputId >= 0) {
                val outputBuffer = encoder.getOutputBuffer(outputId) ?: break
                if (muxerStarted) {
                    mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                }
                encoder.releaseOutputBuffer(outputId, false)
            } else {
                break
            }
        }
    }

    fun release() {
        encoder.stop()
        encoder.release()
        mediaMuxer.stop()
        mediaMuxer.release()
    }
}

/** Create encoder + muxer. */
fun createEncoderState(outputPath: String, width: Int, height: Int, fps: Float): EncoderState {
    val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    val format = MediaFormat.createVideoFormat("video/avc", width, height).apply {
        setInteger(MediaFormat.KEY_BIT_RATE, 1000000)
        setInteger(MediaFormat.KEY_FRAME_RATE, fps.toInt())
        setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
    }
    val encoder = MediaCodec.createEncoderByType("video/avc")
    encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    val inputSurface = encoder.createInputSurface()
    encoder.start()
    val frameDurationUs = (1000000.0 / fps).toLong()
    return EncoderState(encoder, mediaMuxer, inputSurface, frameDurationUs = frameDurationUs)
}
