package GaitVision.com.gait

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Gait scoring using 3 trained models:
 * 1. AE-4D-normal - Autoencoder (reconstruction error) - THIS IS USED FOR DB
 * 2. Ridge - Linear regression (severity prediction)
 * 3. PCA-4 - PCA reconstruction error
 * 
 * Each model has its own scaler embedded in its JSON config.
 * Only AE score is used for the patient database.
 */
class GaitScorer(private val context: Context) {
    
    companion object {
        private const val TAG = "GaitDebug"
        
        // Model files
        private const val AE_CONFIG = "AE-4D-normal.json"
        private const val AE_TFLITE = "AE-4D-normal.tflite"
        private const val RIDGE_CONFIG = "Ridge.json"
        private const val PCA_CONFIG = "PCA-4.json"
    }
    
    // AE model
    private var aeInterpreter: Interpreter? = null
    private var aeScalerMean: FloatArray? = null
    private var aeScalerScale: FloatArray? = null
    private var aeScoreP1: Float = 0f
    private var aeScoreP99: Float = 1f
    private var aeAvailable = false
    
    // Ridge model
    private var ridgeCoef: FloatArray? = null
    private var ridgeIntercept: Float = 0f
    private var ridgeScalerMean: FloatArray? = null
    private var ridgeScalerScale: FloatArray? = null
    private var ridgeScoreMin: Float = 0f      // From score_range
    private var ridgeScoreMax: Float = 100f    // From score_range
    private var ridgeThreshold: Float = 75f    // Classification threshold
    private var ridgeAvailable = false
    
    // PCA model
    private var pcaComponents: Array<FloatArray>? = null  // 4x16 matrix
    private var pcaScalerMean: FloatArray? = null
    private var pcaScalerScale: FloatArray? = null
    private var pcaScoreP1: Float = 0f
    private var pcaScoreP99: Float = 1f
    private var pcaAvailable = false
    
    private var isInitialized = false
    
    /**
     * Initialize all models from JSON configs.
     */
    fun initialize(): Boolean {
        try {
            // Load AE model
            loadAEModel()
            
            // Load Ridge model
            loadRidgeModel()
            
            // Load PCA model
            loadPCAModel()
            
            isInitialized = aeAvailable || ridgeAvailable || pcaAvailable
            Log.d(TAG, "Scorer initialized. Available: AE=$aeAvailable, Ridge=$ridgeAvailable, PCA=$pcaAvailable")
            return isInitialized
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize scorer", e)
            return false
        }
    }
    
    private fun loadAEModel() {
        try {
            Log.d(TAG, "Loading AE model from $AE_CONFIG...")
            val configJson = loadJsonAsset(AE_CONFIG)
            if (configJson == null) {
                Log.e(TAG, "AE: Could not load config JSON")
                return
            }
            val config = JSONObject(configJson)
            
            // Load scaler
            val scaler = config.getJSONObject("scaler")
            aeScalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
            aeScalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
            Log.d(TAG, "AE: Loaded scaler mean=${aeScalerMean?.size}, scale=${aeScalerScale?.size}")
            
            // Load score mapping
            val scoreMapping = config.getJSONObject("score_mapping")
            aeScoreP1 = scoreMapping.getDouble("p1").toFloat()
            aeScoreP99 = scoreMapping.getDouble("p99").toFloat()
            Log.d(TAG, "AE: score_mapping p1=$aeScoreP1, p99=$aeScoreP99")
            
            // Load TFLite model
            Log.d(TAG, "Loading AE TFLite from $AE_TFLITE...")
            val modelBuffer = loadModelFile(AE_TFLITE)
            if (modelBuffer != null) {
                aeInterpreter = Interpreter(modelBuffer)
                aeAvailable = true
                
                // Log model shape info
                val interp = aeInterpreter!!
                Log.d(TAG, "AE TFLite loaded. Input tensors: ${interp.inputTensorCount}, Output tensors: ${interp.outputTensorCount}")
                Log.d(TAG, "AE input[0] shape: ${interp.getInputTensor(0).shape().contentToString()}")
                Log.d(TAG, "AE output[0] shape: ${interp.getOutputTensor(0).shape().contentToString()}")
                Log.d(TAG, "Loaded AE model: ${config.getString("model_name")}")
            } else {
                Log.e(TAG, "AE: Could not load TFLite model file")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load AE model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    private fun loadRidgeModel() {
        try {
            val configJson = loadJsonAsset(RIDGE_CONFIG) ?: return
            val config = JSONObject(configJson)
            
            // Load scaler
            val scaler = config.getJSONObject("scaler")
            ridgeScalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
            ridgeScalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
            
            // Load coefficients
            ridgeCoef = jsonArrayToFloatArray(config.getJSONArray("coefficients"))
            ridgeIntercept = config.getDouble("intercept").toFloat()
            
            // Load score range for proper 0-100 mapping
            val scoreRange = config.getJSONObject("score_range")
            ridgeScoreMin = scoreRange.getDouble("min").toFloat()
            ridgeScoreMax = scoreRange.getDouble("max").toFloat()
            ridgeThreshold = config.getDouble("threshold").toFloat()
            
            Log.d(TAG, "Ridge: range=[$ridgeScoreMin, $ridgeScoreMax], threshold=$ridgeThreshold")
            
            ridgeAvailable = true
            Log.d(TAG, "Loaded Ridge model: ${config.getString("model_name")}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load Ridge model: ${e.message}")
        }
    }
    
    private fun loadPCAModel() {
        try {
            Log.d(TAG, "Loading PCA model from $PCA_CONFIG...")
            val configJson = loadJsonAsset(PCA_CONFIG)
            if (configJson == null) {
                Log.e(TAG, "PCA: Could not load config JSON")
                return
            }
            val config = JSONObject(configJson)
            
            // Load scaler
            val scaler = config.getJSONObject("scaler")
            pcaScalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
            pcaScalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
            Log.d(TAG, "PCA: Loaded scaler mean=${pcaScalerMean?.size}, scale=${pcaScalerScale?.size}")
            
            // Load PCA components (4x16 matrix)
            val pca = config.getJSONObject("pca")
            val componentsJson = pca.getJSONArray("components")
            pcaComponents = Array(componentsJson.length()) { i ->
                jsonArrayToFloatArray(componentsJson.getJSONArray(i))
            }
            Log.d(TAG, "PCA: Loaded components ${pcaComponents?.size}x${pcaComponents?.get(0)?.size}")
            
            // Load score mapping
            val scoreMapping = config.getJSONObject("score_mapping")
            pcaScoreP1 = scoreMapping.getDouble("p1").toFloat()
            pcaScoreP99 = scoreMapping.getDouble("p99").toFloat()
            Log.d(TAG, "PCA: score_mapping p1=$pcaScoreP1, p99=$pcaScoreP99")
            
            pcaAvailable = true
            Log.d(TAG, "Loaded PCA model: ${config.getString("model_name")}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load PCA model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    /**
     * Score gait features using all available models.
     * Returns ScoringResult with individual model scores.
     * 
     * NOTE: For patient DB, use aeScore (the primary score).
     */
    fun score(features: GaitFeatures): ScoringResult {
        if (!isInitialized) {
            Log.w(TAG, "Scorer not initialized")
            return ScoringResult.default()
        }
        
        val featureArray = features.toFeatureArray()
        
        // Log raw features for debugging
        Log.d(TAG, "=== SCORING DEBUG ===")
        Log.d(TAG, "Raw features (${featureArray.size}):")
        GaitFeatures.FEATURE_COLUMNS.forEachIndexed { i, name ->
            val value = if (i < featureArray.size) featureArray[i] else Float.NaN
            Log.d(TAG, "  [$i] $name = $value")
        }
        
        // Check for NaN in input
        val nanCount = featureArray.count { it.isNaN() }
        if (nanCount > 0) {
            Log.w(TAG, "WARNING: $nanCount NaN values in input features!")
        }
        
        // Compute each model's score (each uses its own scaler)
        val aeScore = if (aeAvailable) computeAEScore(featureArray) else Float.NaN
        val ridgeScore = if (ridgeAvailable) computeRidgeScore(featureArray) else Float.NaN
        val pcaScore = if (pcaAvailable) computePCAScore(featureArray) else Float.NaN
        
        Log.d(TAG, "=== FINAL SCORES ===")
        Log.d(TAG, "  AE: $aeScore (available=$aeAvailable)")
        Log.d(TAG, "  Ridge: $ridgeScore (available=$ridgeAvailable)")
        Log.d(TAG, "  PCA: $pcaScore (available=$pcaAvailable)")
        
        return ScoringResult(
            aeScore = aeScore,           // THIS IS THE PRIMARY SCORE FOR DB
            ridgeScore = ridgeScore,
            pcaScore = pcaScore
        )
    }
    
    /**
     * Compute AE score (reconstruction error → 0-100).
     * 0 = severe impairment, 100 = healthy
     */
    private fun computeAEScore(features: FloatArray): Float {
        Log.d(TAG, "--- AE SCORING ---")
        
        val model = aeInterpreter
        if (model == null) {
            Log.e(TAG, "AE FAIL: interpreter is null")
            return Float.NaN
        }
        
        val mean = aeScalerMean
        if (mean == null) {
            Log.e(TAG, "AE FAIL: scaler mean is null")
            return Float.NaN
        }
        
        val scale = aeScalerScale
        if (scale == null) {
            Log.e(TAG, "AE FAIL: scaler scale is null")
            return Float.NaN
        }
        
        Log.d(TAG, "AE: interpreter OK, mean size=${mean.size}, scale size=${scale.size}")
        
        try {
            // Log TFLite model info
            val inputTensor = model.getInputTensor(0)
            val outputTensor = model.getOutputTensor(0)
            Log.d(TAG, "AE TFLite input shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "AE TFLite output shape: ${outputTensor.shape().contentToString()}")
            
            // AE model: 16 inputs → 16 reconstructed outputs
            val numFeatures = 16
            
            // Normalize with AE's scaler
            val normalized = scaleFeatures(features, mean, scale)
            Log.d(TAG, "AE: normalized ${normalized.size} features")
            Log.d(TAG, "AE: first 4 normalized: [${normalized.take(4).joinToString()}]")
            
            // Prepare input/output buffers with explicit size
            val inputBuffer = Array(1) { FloatArray(numFeatures) }
            val outputBuffer = Array(1) { FloatArray(numFeatures) }
            
            // Copy normalized features to input buffer
            for (i in 0 until minOf(normalized.size, numFeatures)) {
                inputBuffer[0][i] = normalized[i]
            }
            
            Log.d(TAG, "AE: running inference...")
            
            // Run TFLite inference
            model.run(inputBuffer, outputBuffer)
            
            Log.d(TAG, "AE: inference complete")
            Log.d(TAG, "AE: first 4 output: [${outputBuffer[0].take(4).joinToString()}]")
            
            // Compute MSE (reconstruction error)
            var mse = 0f
            for (i in 0 until numFeatures) {
                val diff = inputBuffer[0][i] - outputBuffer[0][i]
                mse += diff * diff
            }
            mse /= numFeatures
            
            Log.d(TAG, "AE: MSE = $mse, p1=${aeScoreP1}, p99=${aeScoreP99}")
            
            // Map to 0-100 using the model's score_mapping
            // Original mapping (compressed range):
            // val mapped = ((mse - aeScoreP1) / (aeScoreP99 - aeScoreP1) * 100f).coerceIn(0f, 100f)
            
            // Alternative: Use clinical thresholds for better differentiation
            // Normal MSE: ~0.5, Mild: ~2.3, Moderate: ~2.3, Severe: ~4.8
            // Map using median values for better spread:
            // MSE ≤ 0.8 → 85-100 (Normal)
            // MSE 0.8-2.0 → 65-85 (Mild)
            // MSE 2.0-4.0 → 40-65 (Moderate)
            // MSE > 4.0 → 0-40 (Severe)
            val healthScore = when {
                mse <= 0.8f -> 100f - (mse / 0.8f * 15f)  // 85-100
                mse <= 2.0f -> 85f - ((mse - 0.8f) / 1.2f * 20f)  // 65-85
                mse <= 4.0f -> 65f - ((mse - 2.0f) / 2.0f * 25f)  // 40-65
                else -> maxOf(0f, 40f - ((mse - 4.0f) / 12.14f * 40f))  // 0-40
            }
            
            Log.d(TAG, "AE: mapped (clinical scale), healthScore=$healthScore")
            return healthScore
            
        } catch (e: Exception) {
            Log.e(TAG, "AE EXCEPTION: ${e.javaClass.simpleName}: ${e.message}")
            e.printStackTrace()
            return Float.NaN
        }
    }
    
    /**
     * Compute Ridge score (severity prediction → 0-100).
     * Ridge outputs a raw score where higher = healthier.
     * We map to 0-100 using the score_range from training.
     */
    private fun computeRidgeScore(features: FloatArray): Float {
        Log.d(TAG, "--- RIDGE SCORING ---")
        
        val coef = ridgeCoef ?: return Float.NaN
        val mean = ridgeScalerMean ?: return Float.NaN
        val scale = ridgeScalerScale ?: return Float.NaN
        
        try {
            // Normalize with Ridge's scaler
            val normalized = scaleFeatures(features, mean, scale)
            Log.d(TAG, "Ridge: normalized ${normalized.size} features")
            Log.d(TAG, "Ridge: first 4 normalized: [${normalized.take(4).joinToString()}]")
            
            // Linear prediction: raw = dot(x, coef) + intercept
            var rawScore = ridgeIntercept
            for (i in normalized.indices.take(coef.size)) {
                rawScore += normalized[i] * coef[i]
            }
            
            Log.d(TAG, "Ridge: raw=$rawScore, range=[$ridgeScoreMin, $ridgeScoreMax], threshold=$ridgeThreshold")
            Log.d(TAG, "Ridge: ${if (rawScore > ridgeThreshold) "HEALTHY" else "IMPAIRED"} (raw ${if (rawScore > ridgeThreshold) ">" else "<"} threshold)")
            
            // Map to 0-100 using clinical thresholds (consistent with AE/PCA)
            // Ridge predicts health-like score where higher = healthier
            // Threshold ~75: above = healthy, below = impaired
            // Raw > 85 → Health 85-100 (Normal)
            // Raw 70-85 → Health 65-85 (Mild)  
            // Raw 50-70 → Health 40-65 (Moderate)
            // Raw < 50 → Health 0-40 (Severe)
            val healthScore = when {
                rawScore >= 85f -> 85f + ((rawScore - 85f) / 30f * 15f).coerceAtMost(15f)  // 85-100
                rawScore >= 70f -> 65f + ((rawScore - 70f) / 15f * 20f)  // 65-85
                rawScore >= 50f -> 40f + ((rawScore - 50f) / 20f * 25f)  // 40-65
                else -> maxOf(0f, (rawScore / 50f * 40f))  // 0-40
            }
            
            Log.d(TAG, "Ridge: healthScore=$healthScore (clinical scale)")
            return healthScore
            
        } catch (e: Exception) {
            Log.e(TAG, "Ridge prediction error", e)
            return Float.NaN
        }
    }
    
    /**
     * Compute PCA score (reconstruction error → 0-100).
     */
    private fun computePCAScore(features: FloatArray): Float {
        Log.d(TAG, "--- PCA SCORING ---")
        
        val components = pcaComponents
        if (components == null) {
            Log.e(TAG, "PCA FAIL: components is null")
            return Float.NaN
        }
        
        val mean = pcaScalerMean
        if (mean == null) {
            Log.e(TAG, "PCA FAIL: scaler mean is null")
            return Float.NaN
        }
        
        val scale = pcaScalerScale
        if (scale == null) {
            Log.e(TAG, "PCA FAIL: scaler scale is null")
            return Float.NaN
        }
        
        Log.d(TAG, "PCA: components=${components.size}x${components[0].size}, mean=${mean.size}, scale=${scale.size}")
        
        try {
            // Normalize with PCA's scaler
            val normalized = scaleFeatures(features, mean, scale)
            Log.d(TAG, "PCA: normalized ${normalized.size} features")
            Log.d(TAG, "PCA: first 4 normalized: [${normalized.take(4).joinToString()}]")
            
            // Project onto PC space: pc = x @ components.T
            val nComponents = components.size
            val projected = FloatArray(nComponents)
            for (i in 0 until nComponents) {
                for (j in normalized.indices.take(components[i].size)) {
                    projected[i] += normalized[j] * components[i][j]
                }
            }
            Log.d(TAG, "PCA: projected (${nComponents}D): [${projected.joinToString()}]")
            
            // Reconstruct: x_recon = pc @ components
            val reconstructed = FloatArray(normalized.size)
            for (i in reconstructed.indices) {
                for (j in 0 until nComponents) {
                    if (i < components[j].size) {
                        reconstructed[i] += projected[j] * components[j][i]
                    }
                }
            }
            Log.d(TAG, "PCA: first 4 reconstructed: [${reconstructed.take(4).joinToString()}]")
            
            // Compute MSE
            var mse = 0f
            for (i in normalized.indices) {
                val diff = normalized[i] - reconstructed[i]
                mse += diff * diff
            }
            mse /= normalized.size
            
            Log.d(TAG, "PCA: MSE = $mse, p1=${pcaScoreP1}, p99=${pcaScoreP99}")
            
            // Map to 0-100 using clinical thresholds (same as AE)
            // Normal MSE: ~0.5, Mild: ~2.0, Moderate: ~2.3, Severe: ~5.4
            // MSE ≤ 0.8 → 85-100 (Normal)
            // MSE 0.8-2.0 → 65-85 (Mild)
            // MSE 2.0-4.0 → 40-65 (Moderate)
            // MSE > 4.0 → 0-40 (Severe)
            val healthScore = when {
                mse <= 0.8f -> 100f - (mse / 0.8f * 15f)  // 85-100
                mse <= 2.0f -> 85f - ((mse - 0.8f) / 1.2f * 20f)  // 65-85
                mse <= 4.0f -> 65f - ((mse - 2.0f) / 2.0f * 25f)  // 40-65
                else -> maxOf(0f, 40f - ((mse - 4.0f) / 10f * 40f))  // 0-40
            }
            
            Log.d(TAG, "PCA: healthScore=$healthScore (clinical scale)")
            return healthScore
            
        } catch (e: Exception) {
            Log.e(TAG, "PCA EXCEPTION: ${e.javaClass.simpleName}: ${e.message}")
            e.printStackTrace()
            return Float.NaN
        }
    }
    
    /**
     * Apply StandardScaler: (x - mean) / scale
     */
    private fun scaleFeatures(features: FloatArray, mean: FloatArray, scale: FloatArray): FloatArray {
        val n = minOf(features.size, mean.size, scale.size)
        val scaled = FloatArray(features.size)
        
        for (i in 0 until n) {
            val feat = if (features[i].isNaN()) 0f else features[i]
            scaled[i] = if (scale[i] > 1e-8f) {
                (feat - mean[i]) / scale[i]
            } else {
                feat - mean[i]
            }
        }
        
        return scaled
    }
    
    // Helper functions
    
    private fun loadJsonAsset(filename: String): String? {
        return try {
            val raw = context.assets.open(filename).bufferedReader().use { it.readText() }
            // Sanitize JSON: Android's JSONObject can't parse Infinity/-Infinity/NaN
            raw.replace("-Infinity", "-999999")
               .replace("Infinity", "999999")
               .replace(": NaN", ": null")
        } catch (e: Exception) {
            Log.d(TAG, "Could not load $filename")
            null
        }
    }
    
    private fun jsonArrayToFloatArray(jsonArray: org.json.JSONArray): FloatArray {
        return FloatArray(jsonArray.length()) { i ->
            jsonArray.getDouble(i).toFloat()
        }
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
    
    /**
     * Release resources.
     */
    fun release() {
        aeInterpreter?.close()
        aeInterpreter = null
        isInitialized = false
    }
}

/**
 * Scoring results from all 3 models.
 * 
 * All scores are 0-100 where 100 = healthy, 0 = severely impaired.
 * 
 * For patient database: USE aeScore
 */
data class ScoringResult(
    val aeScore: Float,      // Autoencoder - PRIMARY SCORE FOR DB
    val ridgeScore: Float,   // Ridge regression
    val pcaScore: Float      // PCA reconstruction error
) {
    /**
     * Get the score to save to patient database.
     * This is the AE score.
     */
    fun getScoreForDatabase(): Double {
        return if (aeScore.isNaN()) 0.0 else aeScore.toDouble()
    }
    
    /**
     * Get average of available scores (for display only).
     */
    fun getAverageScore(): Float {
        val scores = listOf(aeScore, ridgeScore, pcaScore).filter { !it.isNaN() }
        return if (scores.isNotEmpty()) scores.average().toFloat() else 50f
    }
    
    companion object {
        fun default() = ScoringResult(
            aeScore = Float.NaN,
            ridgeScore = Float.NaN,
            pcaScore = Float.NaN
        )
    }
    
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as ScoringResult
        return aeScore == other.aeScore
    }

    override fun hashCode(): Int = aeScore.hashCode()
}
