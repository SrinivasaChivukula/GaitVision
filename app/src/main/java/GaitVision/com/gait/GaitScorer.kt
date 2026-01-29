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
    
    // Clinical mapping config (loaded from JSON)
    private data class ClinicalMapping(
        val lowerIsBetter: Boolean,
        val breakpoints: FloatArray,
        val healthScores: FloatArray
    )
    
    // AE model
    private var aeInterpreter: Interpreter? = null
    private var aeScalerMean: FloatArray? = null
    private var aeScalerScale: FloatArray? = null
    private var aeScoreP1: Float = 0f
    private var aeScoreP99: Float = 1f
    private var aeClinicalMapping: ClinicalMapping? = null
    private var aeAvailable = false
    
    // Ridge model
    private var ridgeCoef: FloatArray? = null
    private var ridgeIntercept: Float = 0f
    private var ridgeScalerMean: FloatArray? = null
    private var ridgeScalerScale: FloatArray? = null
    private var ridgeScoreMin: Float = 0f      // From score_range
    private var ridgeScoreMax: Float = 100f    // From score_range
    private var ridgeThreshold: Float = 75f    // Classification threshold
    private var ridgeClinicalMapping: ClinicalMapping? = null
    private var ridgeAvailable = false
    
    // PCA model
    private var pcaComponents: Array<FloatArray>? = null  // 4x16 matrix
    private var pcaScalerMean: FloatArray? = null
    private var pcaScalerScale: FloatArray? = null
    private var pcaScoreP1: Float = 0f
    private var pcaScoreP99: Float = 1f
    private var pcaClinicalMapping: ClinicalMapping? = null
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
            
            // Load clinical mapping
            aeClinicalMapping = loadClinicalMapping(config)
            Log.d(TAG, "AE: clinical mapping loaded=${aeClinicalMapping != null}")
            
            // Load TFLite model
            Log.d(TAG, "Loading AE TFLite from $AE_TFLITE...")
            val modelBuffer = loadModelFile(AE_TFLITE)
            if (modelBuffer != null && aeClinicalMapping != null) {
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
            
            // Load clinical mapping
            ridgeClinicalMapping = loadClinicalMapping(config)
            Log.d(TAG, "Ridge: clinical mapping loaded=${ridgeClinicalMapping != null}")
            
            Log.d(TAG, "Ridge: range=[$ridgeScoreMin, $ridgeScoreMax], threshold=$ridgeThreshold")
            
            ridgeAvailable = ridgeClinicalMapping != null
            Log.d(TAG, "Loaded Ridge model: ${config.getString("model_name")}, available=$ridgeAvailable")
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
            
            // Load clinical mapping
            pcaClinicalMapping = loadClinicalMapping(config)
            Log.d(TAG, "PCA: clinical mapping loaded=${pcaClinicalMapping != null}")
            
            pcaAvailable = pcaClinicalMapping != null
            Log.d(TAG, "Loaded PCA model: ${config.getString("model_name")}, available=$pcaAvailable")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load PCA model: ${e.message}")
            e.printStackTrace()
        }
    }
    
    /**
     * Load clinical mapping from JSON config.
     * Returns null if not present (falls back to hardcoded defaults).
     */
    private fun loadClinicalMapping(config: JSONObject): ClinicalMapping? {
        return try {
            if (!config.has("clinical_mapping")) return null
            
            val mapping = config.getJSONObject("clinical_mapping")
            val lowerIsBetter = mapping.getBoolean("lower_is_better")
            val breakpointsJson = mapping.getJSONArray("breakpoints")
            val healthScoresJson = mapping.getJSONArray("health_scores")
            
            val breakpoints = FloatArray(breakpointsJson.length()) { 
                breakpointsJson.getDouble(it).toFloat() 
            }
            val healthScores = FloatArray(healthScoresJson.length()) { 
                healthScoresJson.getDouble(it).toFloat() 
            }
            
            ClinicalMapping(lowerIsBetter, breakpoints, healthScores)
        } catch (e: Exception) {
            Log.w(TAG, "Could not load clinical_mapping: ${e.message}")
            null
        }
    }
    
    /**
     * Apply clinical mapping to convert raw score to 0-100 health score.
     * Uses piecewise linear interpolation between breakpoints.
     */
    private fun applyClinicalMapping(rawScore: Float, mapping: ClinicalMapping): Float {
        val breakpoints = mapping.breakpoints
        val healthScores = mapping.healthScores
        
        // Find which segment the score falls into
        if (mapping.lowerIsBetter) {
            // Lower raw = healthier (MSE-based: AE, PCA)
            // healthScores[0] = score at raw=0, healthScores[n] = score at raw=infinity
            for (i in breakpoints.indices) {
                if (rawScore <= breakpoints[i]) {
                    val prevBreak = if (i == 0) 0f else breakpoints[i - 1]
                    val prevHealth = healthScores[i]
                    val nextHealth = healthScores[i + 1]
                    val t = (rawScore - prevBreak) / (breakpoints[i] - prevBreak)
                    return prevHealth - t * (prevHealth - nextHealth)
                }
            }
            // Beyond last breakpoint - extrapolate to minimum
            val lastBreak = breakpoints.last()
            val lastHealth = healthScores[healthScores.size - 2]
            val minHealth = healthScores.last()
            // Extrapolate: assume same span as last segment for gradual decline
            val lastSpan = if (breakpoints.size >= 2) breakpoints.last() - breakpoints[breakpoints.size - 2] else lastBreak
            val t = ((rawScore - lastBreak) / (lastSpan * 3)).coerceAtMost(1f)  // 3x span to reach minimum
            return maxOf(minHealth, lastHealth - t * (lastHealth - minHealth))
        } else {
            // Higher raw = healthier (Ridge regression)
            // healthScores[0] = score at raw=0, healthScores[n] = score at raw=infinity
            for (i in breakpoints.indices) {
                if (rawScore <= breakpoints[i]) {
                    val prevBreak = if (i == 0) 0f else breakpoints[i - 1]
                    val prevHealth = if (i == 0) healthScores[0] else healthScores[i]
                    val nextHealth = healthScores[i + 1]
                    val t = (rawScore - prevBreak) / (breakpoints[i] - prevBreak)
                    return prevHealth + t * (nextHealth - prevHealth)
                }
            }
            // Beyond last breakpoint - extrapolate to maximum
            val lastBreak = breakpoints.last()
            val lastHealth = healthScores[healthScores.size - 2]
            val maxHealth = healthScores.last()
            val lastSpan = if (breakpoints.size >= 2) breakpoints.last() - breakpoints[breakpoints.size - 2] else 15f
            val t = ((rawScore - lastBreak) / lastSpan).coerceAtMost(1f)
            return minOf(maxHealth, lastHealth + t * (maxHealth - lastHealth))
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
            
            // Map MSE to 0-100 health score using clinical thresholds from config
            val healthScore = applyClinicalMapping(mse, aeClinicalMapping!!)
            Log.d(TAG, "AE: MSE=$mse -> healthScore=$healthScore")
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
            
            // Map raw score to 0-100 health score using clinical thresholds from config
            val healthScore = applyClinicalMapping(rawScore, ridgeClinicalMapping!!)
            Log.d(TAG, "Ridge: raw=$rawScore -> healthScore=$healthScore")
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
            
            // Map MSE to 0-100 health score using clinical thresholds from config
            val healthScore = applyClinicalMapping(mse, pcaClinicalMapping!!)
            Log.d(TAG, "PCA: MSE=$mse -> healthScore=$healthScore")
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
