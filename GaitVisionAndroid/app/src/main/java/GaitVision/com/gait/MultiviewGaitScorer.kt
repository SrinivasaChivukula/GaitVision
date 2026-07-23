package GaitVision.com.gait

import android.content.Context
import android.util.Log
import GaitVision.com.BuildConfig
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Multi-view (side + back) autoencoder scorer.
 *
 * NOTE: Preliminary model trained on N=9 paired subjects (see
 * multiview_encoder.json "note" field). Score calibration (p1/p99 range)
 * is a rough placeholder until more paired data is collected - retrain
 * and recalibrate as the cohort dataset grows.
 */
class MultiviewGaitScorer(private val context: Context) {
    companion object {
        private const val TAG = "GaitDebug"
        private const val MODEL_FILE = "multiview_encoder.tflite"
        private const val CONFIG_FILE = "multiview_encoder.json"
        private const val NUM_FEATURES = 26

        // Rough placeholder calibration - based on observed training-set
        // distances (-0.39 to -2.27). Needs recalibration with more data.
        private const val SCORE_P1 = -3f
        private const val SCORE_P99 = 0f
    }

    private var interpreter: Interpreter? = null
    private var scalerMean: FloatArray? = null
    private var scalerScale: FloatArray? = null
    private var normalCentroid: FloatArray? = null
    private var invCovariance: Array<FloatArray>? = null
    private var latentDim: Int = 2
    private var isAvailable = false

    init {
        try {
            val modelBuffer = loadModelFile(MODEL_FILE)
            val configJson = loadJsonAsset(CONFIG_FILE)

            if (modelBuffer != null && configJson != null) {
                interpreter = Interpreter(modelBuffer)
                val config = JSONObject(configJson)

                val scaler = config.getJSONObject("scaler")
                scalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
                scalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
                normalCentroid = jsonArrayToFloatArray(config.getJSONArray("normal_centroid"))
                latentDim = config.getInt("latent_dim")

                val invCovArray = config.getJSONArray("inv_covariance")
                invCovariance = Array(invCovArray.length()) { i ->
                    jsonArrayToFloatArray(invCovArray.getJSONArray(i))
                }

                isAvailable = true
                Log.d(TAG, "MultiviewGaitScorer initialized (preliminary N=9 model)")
            } else {
                Log.w(TAG, "MultiviewGaitScorer: model or config not found in assets")
            }
        } catch (e: Exception) {
            Log.e(TAG, "MultiviewGaitScorer init failed: ${e.message}")
            isAvailable = false
        }
    }

    /**
     * Score a combined feature vector (16 side + 10 back = 26 features, in
     * BackFeatureExtractor.BACK_FEATURE_COLUMNS order appended after
     * GaitFeatures.FEATURE_COLUMNS order).
     * Returns 0-100 where 100 = healthy, 0 = severely impaired, NaN if unavailable.
     */
    fun score(sideFeatures: GaitFeatures, backFeatures: BackGaitFeatures): Float {
        if (!isAvailable) {
            Log.w(TAG, "MultiviewGaitScorer not available")
            return Float.NaN
        }

        val combined = sideFeatures.toFeatureArray() + backFeatures.toFeatureArray()
        if (combined.size != NUM_FEATURES) {
            Log.e(TAG, "Feature count mismatch: expected $NUM_FEATURES, got ${combined.size}")
            return Float.NaN
        }

        val mean = scalerMean ?: return Float.NaN
        val scale = scalerScale ?: return Float.NaN
        val centroid = normalCentroid ?: return Float.NaN
        val invCov = invCovariance ?: return Float.NaN
        val model = interpreter ?: return Float.NaN

        return try {
            val normalized = scaleFeatures(combined, mean, scale)
            val inputBuffer = Array(1) { normalized }
            val outputBuffer = Array(1) { FloatArray(latentDim) }
            model.run(inputBuffer, outputBuffer)

            val z = outputBuffer[0]
            val dist = mahalanobisDistance(z, centroid, invCov)
            val rawScore = -dist

            val span = SCORE_P99 - SCORE_P1
            val healthScore = if (span != 0f) {
                ((rawScore - SCORE_P1) / span * 100f).coerceIn(0f, 100f)
            } else 50f

            if (BuildConfig.DEBUG) {
                Log.d(TAG, "Multiview: latent=${z.joinToString()}, dist=$dist, raw=$rawScore -> health=$healthScore")
            }
            healthScore
        } catch (e: Exception) {
            Log.e(TAG, "Multiview scoring exception: ${e.message}")
            Float.NaN
        }
    }

    private fun mahalanobisDistance(z: FloatArray, mu: FloatArray, invCov: Array<FloatArray>): Float {
        val d = z.size
        val diff = FloatArray(d) { z[it] - mu[it] }
        val invCovDiff = FloatArray(d)
        for (i in 0 until d) {
            var sum = 0f
            for (j in 0 until d) {
                sum += invCov[i][j] * diff[j]
            }
            invCovDiff[i] = sum
        }
        var quad = 0f
        for (i in 0 until d) {
            quad += diff[i] * invCovDiff[i]
        }
        return sqrt(quad.coerceAtLeast(0f))
    }

    private fun scaleFeatures(features: FloatArray, mean: FloatArray, scale: FloatArray): FloatArray {
        val n = minOf(features.size, mean.size, scale.size)
        val scaled = FloatArray(features.size)
        for (i in 0 until n) {
            val feat = if (features[i].isNaN()) 0f else features[i]
            scaled[i] = if (scale[i] > 1e-8f) (feat - mean[i]) / scale[i] else feat - mean[i]
        }
        return scaled
    }

    private fun loadJsonAsset(filename: String): String? {
        return try {
            val raw = context.assets.open(filename).bufferedReader().use { it.readText() }
            raw.replace("-Infinity", "-999999")
                .replace("Infinity", "999999")
                .replace(": NaN", ": null")
        } catch (e: Exception) {
            Log.d(TAG, "Could not load $filename")
            null
        }
    }

    private fun jsonArrayToFloatArray(jsonArray: org.json.JSONArray): FloatArray {
        return FloatArray(jsonArray.length()) { i -> jsonArray.getDouble(i).toFloat() }
    }

    private fun loadModelFile(filename: String): MappedByteBuffer? {
        return try {
            val assetFileDescriptor = context.assets.openFd(filename)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            Log.d(TAG, "Could not load model $filename")
            null
        }
    }

    fun release() {
        interpreter?.close()
        interpreter = null
    }
}