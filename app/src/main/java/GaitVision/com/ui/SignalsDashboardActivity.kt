package GaitVision.com.ui

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.LimitLine
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.SignalData
import GaitVision.com.extractedSignals
import GaitVision.com.extractedStrides
import GaitVision.com.selectedStrideIndices
import GaitVision.com.stepSignalMode
import GaitVision.com.gait.Signals
import GaitVision.com.gait.Stride
import org.json.JSONArray
import org.json.JSONObject

/**
 * Dashboard showing all computed signals for debugging and analysis.
 * Mirrors the PC pipeline's expanded_dashboard.py visualization.
 * at some point the goal is to have the app process the traiing batch, so pc parity will be abandonded
 */
class SignalsDashboardActivity : BaseActivity() {

    companion object {
        private const val TAG = "GaitUI"
    }

    private lateinit var tvStepMode: TextView
    private lateinit var tvValidStrides: TextView
    private lateinit var tvFrameValidity: TextView
    private lateinit var tvLegend: TextView
    private lateinit var btnSelectSignal: Button

    private val chartMap = linkedMapOf<String, LineChart>()

    private var currentSignal = "INTER_ANKLE"
    private var cachedSignals: Signals? = null
    private var cachedStrides: List<Stride>? = null
    private val populatedCharts = mutableSetOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_signals_dashboard)

        setupCommonHeader("Signal Dashboard")
        initializeViews()
        setupButtons()
        loadData()
    }

    private fun initializeViews() {
        tvStepMode = findViewById(R.id.tvStepMode)
        tvValidStrides = findViewById(R.id.tvValidStrides)
        tvFrameValidity = findViewById(R.id.tvFrameValidity)
        tvLegend = findViewById(R.id.tvLegend)
        btnSelectSignal = findViewById(R.id.btnSelectSignal)

        chartMap["INTER_ANKLE"] = findViewById(R.id.chartInterAnkle)
        chartMap["KNEE_ANGLES"] = findViewById(R.id.chartKneeAngles)
        chartMap["ANKLE_Y"] = findViewById(R.id.chartAnkleY)
        chartMap["ANKLE_VY"] = findViewById(R.id.chartAnkleVy)
        chartMap["HIP_Y"] = findViewById(R.id.chartHipY)
        chartMap["HEEL_Y"] = findViewById(R.id.chartHeelY)
        chartMap["TOE_Y"] = findViewById(R.id.chartToeY)
        chartMap["MIDHIP_X"] = findViewById(R.id.chartMidHipX)
        chartMap["TRUNK"] = findViewById(R.id.chartTrunk)

        chartMap.values.forEach { configureChart(it) }
    }

    private fun configureChart(chart: LineChart) {
        chart.description.isEnabled = false
        chart.setBackgroundColor(Color.TRANSPARENT)
        chart.setDrawGridBackground(false)
        chart.legend.textColor = Color.WHITE
        chart.legend.textSize = 10f

        chart.xAxis.textColor = Color.WHITE
        chart.xAxis.gridColor = Color.parseColor("#333333")
        chart.xAxis.setDrawAxisLine(true)

        chart.axisLeft.textColor = Color.WHITE
        chart.axisLeft.gridColor = Color.parseColor("#333333")
        chart.axisRight.isEnabled = false

        chart.setTouchEnabled(true)
        chart.isDragEnabled = true
        chart.setScaleEnabled(true)
        chart.setPinchZoom(true)
    }

    private fun setupButtons() {
        btnSelectSignal.setOnClickListener {
            showSignalPopup()
        }
    }

    private fun showSignalPopup() {
        val popup = PopupMenu(this, btnSelectSignal)
        popup.menu.add(0, 1, 0, "Inter-Ankle Distance")
        popup.menu.add(0, 2, 1, "Knee Angles")
        popup.menu.add(0, 3, 2, "Ankle Y Positions")
        popup.menu.add(0, 4, 3, "Ankle Velocities")
        popup.menu.add(0, 5, 4, "Hip Y Positions")
        popup.menu.add(0, 7, 5, "Heel Y Positions")
        popup.menu.add(0, 8, 6, "Toe Y Positions")
        popup.menu.add(0, 9, 7, "MidHip X Position")
        popup.menu.add(0, 6, 8, "Trunk Angle")

        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                1 -> showChart("INTER_ANKLE", "Inter-Ankle Distance")
                2 -> showChart("KNEE_ANGLES", "Knee Angles")
                3 -> showChart("ANKLE_Y", "Ankle Y Positions")
                4 -> showChart("ANKLE_VY", "Ankle Velocities")
                5 -> showChart("HIP_Y", "Hip Y Positions")
                7 -> showChart("HEEL_Y", "Heel Y Positions")
                8 -> showChart("TOE_Y", "Toe Y Positions")
                9 -> showChart("MIDHIP_X", "MidHip X Position")
                6 -> showChart("TRUNK", "Trunk Angle")
            }
            btnSelectSignal.text = item.title
            updateLegend(item.itemId)
            true
        }
        popup.show()
    }

    private fun showChart(signalType: String, title: String) {
        currentSignal = signalType

        chartMap.values.forEach { it.visibility = View.INVISIBLE }

        // Lazy-populate on first show
        val signals = cachedSignals
        if (signals != null && signalType !in populatedCharts) {
            populateChart(signalType, signals, cachedStrides)
            populatedCharts.add(signalType)
        }

        chartMap[signalType]?.visibility = View.VISIBLE
    }

    private fun updateLegend(signalId: Int) {
        val strideLegend = "Gold = SELECTED, Green = Valid, Gray = Invalid"
        tvLegend.text = when (signalId) {
            1 -> "Horizontal distance between ankles | $strideLegend"
            2 -> "Blue = Left Knee, Red = Right Knee | $strideLegend"
            3 -> "Blue = Left Ankle Y, Red = Right Ankle Y (inverted) | $strideLegend"
            4 -> "Blue = Left Ankle Vy, Red = Right Ankle Vy | $strideLegend"
            5 -> "Blue = Left Hip Y, Red = Right Hip Y (inverted) | $strideLegend"
            7 -> "Blue = Left Heel Y, Red = Right Heel Y (inverted) | $strideLegend"
            8 -> "Blue = Left Toe Y, Red = Right Toe Y (inverted) | $strideLegend"
            9 -> "MidHip X (anteroposterior) | $strideLegend"
            6 -> "Trunk lean angle (degrees) | $strideLegend"
            else -> "Blue = Left, Red = Right | $strideLegend"
        }
    }

    private fun loadData() {
        val resultId = intent.getLongExtra(ResultsActivity.EXTRA_RESULT_ID, -1L)

        if (resultId > 0) {
            loadFromDatabase(resultId)
        } else {
            displaySignals()
        }
    }

    /**
     * Load from DB into globals, then call displaySignals() as normal.
     *
     * WARNING: This overwrites shared globals (extractedSignals, extractedStrides, etc.).
     * Safe today because navigation is linear, but a ViewModel/StateFlow refactor
     * should replace this if we ever need concurrent or comparative analysis views.
     */
    private fun loadFromDatabase(resultId: Long) {
        lifecycleScope.launch {
            val db = AppDatabase.getDatabase(this@SignalsDashboardActivity)
            val result = withContext(Dispatchers.IO) { db.analysisResultDao().getResultById(resultId) }
            val signalRows = withContext(Dispatchers.IO) { db.signalDataDao().getSignalDataByResultId(resultId) }

            if (signalRows.isEmpty()) {
                Toast.makeText(this@SignalsDashboardActivity, "No signal data for this analysis", Toast.LENGTH_SHORT).show()
                return@launch
            }

            // Set globals so all existing code works
            val signals = buildSignalsFromDb(signalRows)
            extractedSignals = signals
            stepSignalMode = result?.stepSignalMode
            extractedStrides = parseStridesJson(result?.stridesJson)
            selectedStrideIndices = parseSelectedIndicesJson(result?.selectedStrideIndicesJson)

            Log.d(TAG, "Loaded signals: ${signals.timestamps.size} frames, heelY non-NaN=${signals.heelLeftY.count { !it.isNaN() }}, toeY=${signals.toeLeftY.count { !it.isNaN() }}, midHipX=${signals.midHipX.count { !it.isNaN() }}")

            displaySignals()
        }
    }

    private fun displaySignals() {
        val signals = extractedSignals
        val strides = extractedStrides

        if (signals == null) {
            tvStepMode.text = "No data"
            tvValidStrides.text = "--"
            tvFrameValidity.text = "--"
            Toast.makeText(this, "No signal data available", Toast.LENGTH_SHORT).show()
            return
        }

        // Cache for lazy chart population
        cachedSignals = signals
        cachedStrides = strides
        populatedCharts.clear()

        tvStepMode.text = stepSignalMode ?: "UNKNOWN"
        tvValidStrides.text = (strides?.count { it.isValid } ?: 0).toString()

        val validFrames = signals.isValid.count { it }
        val totalFrames = signals.isValid.size
        val validPct = if (totalFrames > 0) (validFrames * 100 / totalFrames) else 0
        tvFrameValidity.text = "$validPct%"

        // Only populate the default visible chart; others populated on demand
        showChart("INTER_ANKLE", "Inter-Ankle Distance")

        Log.d(TAG, "Loaded ${signals.timestamps.size} frames, ${strides?.size ?: 0} strides")
    }

    private fun buildSignalsFromDb(rows: List<SignalData>): Signals {
        val n = rows.size
        val timestamps = FloatArray(n) { rows[it].timestamp ?: (it / 30f) }
        val frameIndices = IntArray(n) { rows[it].frameNumber }
        val isValid = BooleanArray(n) { rows[it].isValid }
        val interAnkleDist = FloatArray(n) { rows[it].interAnkleDist ?: Float.NaN }
        val kneeAngleLeft = FloatArray(n) { rows[it].kneeAngleLeft ?: Float.NaN }
        val kneeAngleRight = FloatArray(n) { rows[it].kneeAngleRight ?: Float.NaN }
        val trunkAngle = FloatArray(n) { rows[it].trunkAngle ?: Float.NaN }
        val ankleLeftY = FloatArray(n) { rows[it].ankleLeftY ?: Float.NaN }
        val ankleRightY = FloatArray(n) { rows[it].ankleRightY ?: Float.NaN }
        val hipLeftY = FloatArray(n) { rows[it].hipLeftY ?: Float.NaN }
        val hipRightY = FloatArray(n) { rows[it].hipRightY ?: Float.NaN }
        val ankleLeftVy = FloatArray(n) { rows[it].ankleLeftVy ?: Float.NaN }
        val ankleRightVy = FloatArray(n) { rows[it].ankleRightVy ?: Float.NaN }
        val heelLeftY = FloatArray(n) { rows[it].heelLeftY ?: Float.NaN }
        val heelRightY = FloatArray(n) { rows[it].heelRightY ?: Float.NaN }
        val toeLeftY = FloatArray(n) { rows[it].toeLeftY ?: Float.NaN }
        val toeRightY = FloatArray(n) { rows[it].toeRightY ?: Float.NaN }
        val midHipX = FloatArray(n) { rows[it].midHipX ?: Float.NaN }
        // Fields not stored in signal_data — fill with NaN
        val empty = FloatArray(n) { Float.NaN }

        return Signals(
            timestamps = timestamps,
            frameIndices = frameIndices,
            isValid = isValid,
            interAnkleDist = interAnkleDist,
            kneeAngleLeft = kneeAngleLeft,
            kneeAngleRight = kneeAngleRight,
            trunkAngle = trunkAngle,
            ankleAngleLeft = empty,
            ankleAngleRight = empty,
            hipAngleLeft = empty,
            hipAngleRight = empty,
            strideAngle = empty,
            ankleLeftX = empty,
            ankleRightX = empty,
            ankleLeftY = ankleLeftY,
            ankleRightY = ankleRightY,
            hipLeftY = hipLeftY,
            hipRightY = hipRightY,
            heelLeftY = heelLeftY,
            heelRightY = heelRightY,
            toeLeftY = toeLeftY,
            toeRightY = toeRightY,
            midHipX = midHipX,
            ankleLeftVy = ankleLeftVy,
            ankleRightVy = ankleRightVy,
            hipAvgVy = empty
        )
    }

    private fun parseStridesJson(json: String?): List<Stride>? {
        if (json.isNullOrBlank()) return null
        return try {
            val arr = JSONArray(json)
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                Stride(
                    startFrame = o.getInt("sf"),
                    endFrame = o.getInt("ef"),
                    startTimeS = o.getDouble("st").toFloat(),
                    endTimeS = o.getDouble("et").toFloat(),
                    step1Frame = o.getInt("s1f"),
                    step2Frame = o.getInt("s2f"),
                    step1TimeS = o.getDouble("s1t").toFloat(),
                    step2TimeS = o.getDouble("s2t").toFloat(),
                    isValid = o.getBoolean("v"),
                    invalidReason = if (o.isNull("r")) null else o.getString("r")
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse strides JSON", e)
            null
        }
    }

    private fun parseSelectedIndicesJson(json: String?): List<Int>? {
        if (json.isNullOrBlank()) return null
        return try {
            val arr = JSONArray(json)
            (0 until arr.length()).map { arr.getInt(it) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse selected indices JSON", e)
            null
        }
    }

    /** Signal series config: (signal array, label, color hex, invertY?) */
    private data class SeriesConfig(val signal: FloatArray, val label: String, val color: String, val invertY: Boolean = false)

    private fun populateChart(signalType: String, signals: Signals, strides: List<Stride>?) {
        val chart = chartMap[signalType] ?: return
        val addZeroLine = signalType == "ANKLE_VY" || signalType == "TRUNK"

        val seriesList: List<SeriesConfig> = when (signalType) {
            "INTER_ANKLE" -> listOf(SeriesConfig(signals.interAnkleDist, "Inter-Ankle Distance", "#2196F3"))
            "KNEE_ANGLES" -> listOf(
                SeriesConfig(signals.kneeAngleLeft, "Left Knee", "#3498DB"),
                SeriesConfig(signals.kneeAngleRight, "Right Knee", "#E74C3C"))
            "ANKLE_Y" -> listOf(
                SeriesConfig(signals.ankleLeftY, "Left Ankle Y", "#3498DB", invertY = true),
                SeriesConfig(signals.ankleRightY, "Right Ankle Y", "#E74C3C", invertY = true))
            "ANKLE_VY" -> listOf(
                SeriesConfig(signals.ankleLeftVy, "Left Ankle Vy", "#3498DB"),
                SeriesConfig(signals.ankleRightVy, "Right Ankle Vy", "#E74C3C"))
            "HIP_Y" -> listOf(
                SeriesConfig(signals.hipLeftY, "Left Hip Y", "#3498DB", invertY = true),
                SeriesConfig(signals.hipRightY, "Right Hip Y", "#E74C3C", invertY = true))
            "HEEL_Y" -> listOf(
                SeriesConfig(signals.heelLeftY, "Left Heel Y", "#3498DB", invertY = true),
                SeriesConfig(signals.heelRightY, "Right Heel Y", "#E74C3C", invertY = true))
            "TOE_Y" -> listOf(
                SeriesConfig(signals.toeLeftY, "Left Toe Y", "#3498DB", invertY = true),
                SeriesConfig(signals.toeRightY, "Right Toe Y", "#E74C3C", invertY = true))
            "MIDHIP_X" -> listOf(SeriesConfig(signals.midHipX, "MidHip X", "#9B59B6"))
            "TRUNK" -> listOf(SeriesConfig(signals.trunkAngle, "Trunk Angle", "#9B59B6"))
            else -> return
        }

        val dataSets = seriesList.map { cfg ->
            val entries = mutableListOf<Entry>()
            for (i in signals.timestamps.indices) {
                val t = signals.timestamps[i]
                val v = cfg.signal[i]
                if (!t.isNaN() && !v.isNaN()) {
                    entries.add(Entry(t, if (cfg.invertY) -v else v))
                }
            }
            LineDataSet(entries, cfg.label).apply {
                color = Color.parseColor(cfg.color)
                setDrawCircles(false)
                setDrawValues(false)
                lineWidth = 1.5f
            }
        }

        chart.data = LineData(dataSets)
        if (addZeroLine) {
            val zeroLine = LimitLine(0f, "").apply { lineColor = Color.GRAY; lineWidth = 0.8f }
            chart.axisLeft.addLimitLine(zeroLine)
        }
        addStrideHighlights(chart, signals, strides)
        chart.invalidate()
    }

    /**
     * Add highlight bands for strides.
     * - Gold/thick = SELECTED strides (used for feature computation)
     * - Green/dashed = Other valid strides
     * - Gray/thin = Invalid strides
     */
    private fun addStrideHighlights(chart: LineChart, signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        if (strides == null) return

        // Clear previous limit lines
        chart.xAxis.removeAllLimitLines()
        
        val selectedIndices = selectedStrideIndices ?: emptyList()

        strides.forEachIndexed { idx, stride ->
            val startTime = signals.timestamps.getOrNull(stride.startFrame) ?: return@forEachIndexed
            val endTime = signals.timestamps.getOrNull(minOf(stride.endFrame, signals.timestamps.size - 1)) ?: return@forEachIndexed

            if (startTime.isNaN() || endTime.isNaN()) return@forEachIndexed

            val isSelected = selectedIndices.contains(idx)
            val lineColor = when {
                isSelected -> Color.parseColor("#FFC107")  // Gold = SELECTED
                stride.isValid -> Color.parseColor("#4CAF50")  // Green = valid but not selected
                else -> Color.parseColor("#9E9E9E")  // Gray = invalid
            }
            val lineWidth = if (isSelected) 2.5f else 1f
            
            // Add start line
            val startLine = LimitLine(startTime, if (isSelected) "★" else "")
            startLine.lineColor = lineColor
            startLine.lineWidth = lineWidth
            if (!isSelected) startLine.enableDashedLine(10f, 5f, 0f)
            chart.xAxis.addLimitLine(startLine)

            // Add end line
            val endLine = LimitLine(endTime, "")
            endLine.lineColor = lineColor
            endLine.lineWidth = lineWidth
            if (!isSelected) endLine.enableDashedLine(10f, 5f, 0f)
            chart.xAxis.addLimitLine(endLine)
        }
    }

}
