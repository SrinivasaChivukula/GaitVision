@file:JvmName("GaitMath")
package GaitVision.com.gait

import kotlin.math.sqrt

/** average() returning 0f if empty. */
internal fun List<Float>.averageOrZero(): Float = if (isNotEmpty()) average().toFloat() else 0f

internal fun List<Float>.std(): Float {
    if (size < 2) return 0f
    val mean = average().toFloat()
    val variance = sumOf { ((it - mean) * (it - mean)).toDouble() } / size
    return sqrt(variance).toFloat()
}

internal fun List<Float>.median(): Float {
    if (isEmpty()) return 0f
    val sorted = sorted()
    return if (size % 2 == 0) (sorted[size / 2 - 1] + sorted[size / 2]) / 2f else sorted[size / 2]
}

internal fun FloatArray.std(): Float = this.toList().filter { !it.isNaN() }.std()

/** Linear-interp percentile. */
internal fun percentile(sortedValues: List<Float>, p: Float): Float {
    if (sortedValues.isEmpty()) return 0f
    if (sortedValues.size == 1) return sortedValues[0]
    val n = sortedValues.size
    val idx = (n - 1) * p / 100f
    val lower = idx.toInt().coerceIn(0, n - 2)
    val upper = lower + 1
    val fraction = idx - lower
    return sortedValues[lower] + fraction * (sortedValues[upper] - sortedValues[lower])
}
