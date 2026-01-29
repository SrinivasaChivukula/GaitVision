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
        private const val TAG = "GaitScorer"
        
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
            val configJson = loadJsonAsset(AE_CONFIG) ?: return
            val config = JSONObject(configJson)
            
            // Load scaler
            val scaler = config.getJSONObject("scaler")
            aeScalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
            aeScalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
            
            // Load score mapping
            val scoreMapping = config.getJSONObject("score_mapping")
            aeScoreP1 = scoreMapping.getDouble("p1").toFloat()
            aeScoreP99 = scoreMapping.getDouble("p99").toFloat()
            
            // Load TFLite model
            val modelBuffer = loadModelFile(AE_TFLITE)
            if (modelBuffer != null) {
                aeInterpreter = Interpreter(modelBuffer)
                aeAvailable = true
                Log.d(TAG, "Loaded AE model: ${config.getString("model_name")}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load AE model: ${e.message}")
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
            
            ridgeAvailable = true
            Log.d(TAG, "Loaded Ridge model: ${config.getString("model_name")}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load Ridge model: ${e.message}")
        }
    }
    
    private fun loadPCAModel() {
        try {
            val configJson = loadJsonAsset(PCA_CONFIG) ?: return
            val config = JSONObject(configJson)
            
            // Load scaler
            val scaler = config.getJSONObject("scaler")
            pcaScalerMean = jsonArrayToFloatArray(scaler.getJSONArray("mean"))
            pcaScalerScale = jsonArrayToFloatArray(scaler.getJSONArray("scale"))
            
            // Load PCA components (4x16 matrix)
            val pca = config.getJSONObject("pca")
            val componentsJson = pca.getJSONArray("components")
            pcaComponents = Array(componentsJson.length()) { i ->
                jsonArrayToFloatArray(componentsJson.getJSONArray(i))
            }
            
            // Load score mapping
            val scoreMapping = config.getJSONObject("score_mapping")
            pcaScoreP1 = scoreMapping.getDouble("p1").toFloat()
            pcaScoreP99 = scoreMapping.getDouble("p99").toFloat()
            
            pcaAvailable = true
            Log.d(TAG, "Loaded PCA model: ${config.getString("model_name")}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load PCA model: ${e.message}")
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
        
        // Compute each model's score (each uses its own scaler)
        val aeScore = if (aeAvailable) computeAEScore(featureArray) else Float.NaN
        val ridgeScore = if (ridgeAvailable) computeRidgeScore(featureArray) else Float.NaN
        val pcaScore = if (pcaAvailable) computePCAScore(featureArray) else Float.NaN
        
        Log.d(TAG, "Scores - AE: $aeScore, Ridge: $ridgeScore, PCA: $pcaScore")
        
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
        val model = aeInterpreter ?: run {
            Log.w(TAG, "AE: interpreter is null")
            return Float.NaN
        }
        val mean = aeScalerMean ?: run {
            Log.w(TAG, "AE: scaler mean is null")
            return Float.NaN
        }
        val scale = aeScalerScale ?: run {
            Log.w(TAG, "AE: scaler scale is null")
            return Float.NaN
        }
        
        // AE model: 16 inputs → 16 reconstructed outputs
        val numFeatures = 16
        
        // Normalize with AE's scaler
        val normalized = scaleFeatures(features, mean, scale)
        Log.d(TAG, "AE: normalized ${normalized.size} features")
        
        // Prepare input/output buffers with explicit size
        val inputBuffer = Array(1) { FloatArray(numFeatures) }
        val outputBuffer = Array(1) { FloatArray(numFeatures) }
        
        // Copy normalized features to input buffer
        for (i in 0 until minOf(normalized.size, numFeatures)) {
            inputBuffer[0][i] = normalized[i]
        }
        
        // Run TFLite inference
        model.run(inputBuffer, outputBuffer)
        
        // Compute MSE (reconstruction error)
        var mse = 0f
        for (i in 0 until numFeatures) {
            val diff = inputBuffer[0][i] - outputBuffer[0][i]
            mse += diff * diff
        }
        mse /= numFeatures
        
        Log.d(TAG, "AE: MSE = $mse, p1=${aeScoreP1}, p99=${aeScoreP99}")
        
        // Map to 0-100 using the model's score_mapping
        // Raw error: low = healthy, high = impaired
        // Formula: clip((score - p1) / (p99 - p1) * 100, 0, 100)
        // This gives 0-100 where HIGH = impaired
        // We want HIGH = healthy, so invert: 100 - mapped
        val mapped = ((mse - aeScoreP1) / (aeScoreP99 - aeScoreP1) * 100f).coerceIn(0f, 100f)
        val healthScore = 100f - mapped  // Invert so 100 = healthy
        
        Log.d(TAG, "AE: mapped=$mapped, healthScore=$healthScore")
        return healthScore
    }
    
    /**
     * Compute Ridge score (severity prediction → 0-100).
     */
    private fun computeRidgeScore(features: FloatArray): Float {
        val coef = ridgeCoef ?: return Float.NaN
        val mean = ridgeScalerMean ?: return Float.NaN
        val scale = ridgeScalerScale ?: return Float.NaN
        
        try {
            // Normalize with Ridge's scaler
            val normalized = scaleFeatures(features, mean, scale)
            
            // Linear prediction: severity = dot(x, coef) + intercept
            var severity = ridgeIntercept
            for (i in normalized.indices.take(coef.size)) {
                severity += normalized[i] * coef[i]
            }
            
            // Ridge predicts severity (0-3 scale approx)
            // Convert to health score: (1 - severity/3) * 100
            // But the actual range from JSON is ~0 to ~115, threshold ~75
            // Let's use a simpler mapping based on observed range
            val healthScore = ((1f - severity / 100f) * 100f).coerceIn(0f, 100f)
            
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
        val components = pcaComponents ?: run {
            Log.w(TAG, "PCA: components is null")
            return Float.NaN
        }
        val mean = pcaScalerMean ?: run {
            Log.w(TAG, "PCA: scaler mean is null")
            return Float.NaN
        }
        val scale = pcaScalerScale ?: run {
            Log.w(TAG, "PCA: scaler scale is null")
            return Float.NaN
        }
        
        // Normalize with PCA's scaler
        val normalized = scaleFeatures(features, mean, scale)
        Log.d(TAG, "PCA: normalized ${normalized.size} features, ${components.size} components")
        
        // Project onto PC space: pc = x @ components.T
        val nComponents = components.size
        val projected = FloatArray(nComponents)
        for (i in 0 until nComponents) {
            for (j in normalized.indices.take(components[i].size)) {
                projected[i] += normalized[j] * components[i][j]
            }
        }
        
        // Reconstruct: x_recon = pc @ components
        val reconstructed = FloatArray(normalized.size)
        for (i in reconstructed.indices) {
            for (j in 0 until nComponents) {
                if (i < components[j].size) {
                    reconstructed[i] += projected[j] * components[j][i]
                }
            }
        }
        
        // Compute MSE
        var mse = 0f
        for (i in normalized.indices) {
            val diff = normalized[i] - reconstructed[i]
            mse += diff * diff
        }
        mse /= normalized.size
        
        Log.d(TAG, "PCA: MSE = $mse, p1=${pcaScoreP1}, p99=${pcaScoreP99}")
        
        // Map to 0-100 (invert so 100 = healthy)
        val mapped = ((mse - pcaScoreP1) / (pcaScoreP99 - pcaScoreP1) * 100f).coerceIn(0f, 100f)
        val healthScore = 100f - mapped
        
        Log.d(TAG, "PCA: mapped=$mapped, healthScore=$healthScore")
        return healthScore
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
            context.assets.open(filename).bufferedReader().use { it.readText() }
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
