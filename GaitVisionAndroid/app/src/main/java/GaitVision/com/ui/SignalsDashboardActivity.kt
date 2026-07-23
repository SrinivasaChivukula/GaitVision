package GaitVision.com.ui

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
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
import GaitVision.com.AnalysisSession
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
        private const val TAG = "SignalsDashboard"
    }

    private lateinit var tvStepMode: TextView
    private lateinit var tvValidStrides: TextView
    private lateinit var tvFrameValidity: TextView
    private lateinit var tvLegend: TextView
    private lateinit var btnSelectSignal: LinearLayout

    private lateinit var chartInterAnkle: LineChart
    private lateinit var chartKneeAngles: LineChart
    private lateinit var chartAnkleY: LineChart
    private lateinit var chartAnkleVy: LineChart
    private lateinit var chartHipY: LineChart
    private lateinit var chartTrunk: LineChart

    private var currentSignal = "INTER_ANKLE"
    private var cachedSignals: Signals? = null
    private var cachedStrides: List<Stride>? = null
    private val populatedCharts = mutableSetOf<String>()

    // Local state to avoid race conditions with singleton AnalysisSession
    private var localSignals: Signals? = null
    private var localStrides: List<Stride>? = null
    private var localStepMode: String? = null
    private var localSelectedIndices: List<Int>? = null

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

        chartInterAnkle = findViewById(R.id.chartInterAnkle)
        chartKneeAngles = findViewById(R.id.chartKneeAngles)
        chartAnkleY = findViewById(R.id.chartAnkleY)
        chartAnkleVy = findViewById(R.id.chartAnkleVy)
        chartHipY = findViewById(R.id.chartHipY)
        chartTrunk = findViewById(R.id.chartTrunk)

        // Configure all charts with dark theme
        listOf(chartInterAnkle, chartKneeAngles, chartAnkleY, chartAnkleVy, chartHipY, chartTrunk).forEach { chart ->
            configureChart(chart)
        }
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
        popup.menu.add(0, 6, 5, "Trunk Angle")

        popup.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                1 -> showChart("INTER_ANKLE", "Inter-Ankle Distance")
                2 -> showChart("KNEE_ANGLES", "Knee Angles")
                3 -> showChart("ANKLE_Y", "Ankle Y Positions")
                4 -> showChart("ANKLE_VY", "Ankle Velocities")
                5 -> showChart("HIP_Y", "Hip Y Positions")
                6 -> showChart("TRUNK", "Trunk Angle")
            }
            // Update the signal selector label in the row
            (btnSelectSignal.getChildAt(1) as? TextView)?.text = item.title
            updateLegend(item.itemId)
            true
        }
        popup.show()
    }

    private fun showChart(signalType: String, title: String) {
        currentSignal = signalType

        chartInterAnkle.visibility = View.INVISIBLE
        chartKneeAngles.visibility = View.INVISIBLE
        chartAnkleY.visibility = View.INVISIBLE
        chartAnkleVy.visibility = View.INVISIBLE
        chartHipY.visibility = View.INVISIBLE
        chartTrunk.visibility = View.INVISIBLE

        // Lazy-populate: only build chart data on first show
        val signals = cachedSignals
        if (signals != null && signalType !in populatedCharts) {
            when (signalType) {
                "INTER_ANKLE" -> populateInterAnkleChart(signals, cachedStrides)
                "KNEE_ANGLES" -> populateKneeAnglesChart(signals, cachedStrides)
                "ANKLE_Y" -> populateAnkleYChart(signals, cachedStrides)
                "ANKLE_VY" -> populateAnkleVyChart(signals, cachedStrides)
                "HIP_Y" -> populateHipYChart(signals, cachedStrides)
                "TRUNK" -> populateTrunkChart(signals, cachedStrides)
            }
            populatedCharts.add(signalType)
        }

        when (signalType) {
            "INTER_ANKLE" -> chartInterAnkle.visibility = View.VISIBLE
            "KNEE_ANGLES" -> chartKneeAngles.visibility = View.VISIBLE
            "ANKLE_Y" -> chartAnkleY.visibility = View.VISIBLE
            "ANKLE_VY" -> chartAnkleVy.visibility = View.VISIBLE
            "HIP_Y" -> chartHipY.visibility = View.VISIBLE
            "TRUNK" -> chartTrunk.visibility = View.VISIBLE
        }
    }

    private fun updateLegend(signalId: Int) {
        val strideLegend = "Gold = SELECTED, Green = Valid, Gray = Invalid"
        tvLegend.text = when (signalId) {
            1 -> "Horizontal distance between ankles | $strideLegend"
            2 -> "Blue = Left Knee, Red = Right Knee | $strideLegend"
            3 -> "Blue = Left Ankle Y, Red = Right Ankle Y (inverted) | $strideLegend"
            4 -> "Blue = Left Ankle Vy, Red = Right Ankle Vy | $strideLegend"
            5 -> "Blue = Left Hip Y, Red = Right Hip Y (inverted) | $strideLegend"
            6 -> "Trunk lean angle (degrees) | $strideLegend"
            else -> "Blue = Left, Red = Right | $strideLegend"
        }
    }

    private fun loadData() {
        val resultId = intent.getLongExtra(ResultsActivity.EXTRA_RESULT_ID, -1L)

        if (resultId > 0) {
            loadFromDatabase(resultId)
        } else {
            // Copy from global session to local state (Live analysis path)
            localSignals = AnalysisSession.extractedSignals
            localStrides = AnalysisSession.extractedStrides
            localStepMode = AnalysisSession.stepSignalMode
            localSelectedIndices = AnalysisSession.selectedStrideIndices
            
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

            // Populate local state
            localSignals = buildSignalsFromDb(signalRows)
            localStepMode = result?.stepSignalMode
            localStrides = parseStridesJson(result?.stridesJson)
            localSelectedIndices = parseSelectedIndicesJson(result?.selectedStrideIndicesJson)

            displaySignals()
        }
    }

    private fun displaySignals() {
        val signals = localSignals
        val strides = localStrides

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

        tvStepMode.text = localStepMode ?: "UNKNOWN"
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
            frontalKneeOffsetLeft = empty,
            frontalKneeOffsetRight = empty,
            ankleLeftX = empty,
            ankleRightX = empty,
            ankleLeftY = ankleLeftY,
            ankleRightY = ankleRightY,
            kneeLeftX = empty,
            kneeRightX = empty,
            kneeLeftY = empty,
            kneeRightY = empty,
            hipLeftY = hipLeftY,
            hipRightY = hipRightY,
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

    private fun populateInterAnkleChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val entries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            val v = signals.interAnkleDist[i]
            if (!t.isNaN() && !v.isNaN()) {
                entries.add(Entry(t, v))
            }
        }

        val dataSet = LineDataSet(entries, "Inter-Ankle Distance")
        dataSet.color = Color.parseColor("#2196F3")
        dataSet.setDrawCircles(false)
        dataSet.setDrawValues(false)
        dataSet.lineWidth = 1.5f

        chartInterAnkle.data = LineData(dataSet)
        addStrideHighlights(chartInterAnkle, signals, strides)
        chartInterAnkle.invalidate()
    }

    private fun populateKneeAnglesChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val leftEntries = mutableListOf<Entry>()
        val rightEntries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            if (!t.isNaN()) {
                if (!signals.kneeAngleLeft[i].isNaN()) {
                    leftEntries.add(Entry(t, signals.kneeAngleLeft[i]))
                }
                if (!signals.kneeAngleRight[i].isNaN()) {
                    rightEntries.add(Entry(t, signals.kneeAngleRight[i]))
                }
            }
        }

        val leftDataSet = LineDataSet(leftEntries, "Left Knee")
        leftDataSet.color = Color.parseColor("#3498DB")
        leftDataSet.setDrawCircles(false)
        leftDataSet.setDrawValues(false)
        leftDataSet.lineWidth = 1.5f

        val rightDataSet = LineDataSet(rightEntries, "Right Knee")
        rightDataSet.color = Color.parseColor("#E74C3C")
        rightDataSet.setDrawCircles(false)
        rightDataSet.setDrawValues(false)
        rightDataSet.lineWidth = 1.5f

        chartKneeAngles.data = LineData(leftDataSet, rightDataSet)
        addStrideHighlights(chartKneeAngles, signals, strides)
        chartKneeAngles.invalidate()
    }

    private fun populateAnkleYChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val leftEntries = mutableListOf<Entry>()
        val rightEntries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            if (!t.isNaN()) {
                // Invert Y so down is down on screen
                if (!signals.ankleLeftY[i].isNaN()) {
                    leftEntries.add(Entry(t, -signals.ankleLeftY[i]))
                }
                if (!signals.ankleRightY[i].isNaN()) {
                    rightEntries.add(Entry(t, -signals.ankleRightY[i]))
                }
            }
        }

        val leftDataSet = LineDataSet(leftEntries, "Left Ankle Y")
        leftDataSet.color = Color.parseColor("#3498DB")
        leftDataSet.setDrawCircles(false)
        leftDataSet.setDrawValues(false)
        leftDataSet.lineWidth = 1.5f

        val rightDataSet = LineDataSet(rightEntries, "Right Ankle Y")
        rightDataSet.color = Color.parseColor("#E74C3C")
        rightDataSet.setDrawCircles(false)
        rightDataSet.setDrawValues(false)
        rightDataSet.lineWidth = 1.5f

        chartAnkleY.data = LineData(leftDataSet, rightDataSet)
        addStrideHighlights(chartAnkleY, signals, strides)
        chartAnkleY.invalidate()
    }

    private fun populateAnkleVyChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val leftEntries = mutableListOf<Entry>()
        val rightEntries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            if (!t.isNaN()) {
                if (!signals.ankleLeftVy[i].isNaN()) {
                    leftEntries.add(Entry(t, signals.ankleLeftVy[i]))
                }
                if (!signals.ankleRightVy[i].isNaN()) {
                    rightEntries.add(Entry(t, signals.ankleRightVy[i]))
                }
            }
        }

        val leftDataSet = LineDataSet(leftEntries, "Left Ankle Vy")
        leftDataSet.color = Color.parseColor("#3498DB")
        leftDataSet.setDrawCircles(false)
        leftDataSet.setDrawValues(false)
        leftDataSet.lineWidth = 1.5f

        val rightDataSet = LineDataSet(rightEntries, "Right Ankle Vy")
        rightDataSet.color = Color.parseColor("#E74C3C")
        rightDataSet.setDrawCircles(false)
        rightDataSet.setDrawValues(false)
        rightDataSet.lineWidth = 1.5f

        chartAnkleVy.data = LineData(leftDataSet, rightDataSet)
        
        // Add zero line for velocity
        val zeroLine = LimitLine(0f, "")
        zeroLine.lineColor = Color.GRAY
        zeroLine.lineWidth = 0.8f
        chartAnkleVy.axisLeft.addLimitLine(zeroLine)
        
        addStrideHighlights(chartAnkleVy, signals, strides)
        chartAnkleVy.invalidate()
    }

    private fun populateHipYChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val leftEntries = mutableListOf<Entry>()
        val rightEntries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            if (!t.isNaN()) {
                // Invert Y so down is down on screen
                if (!signals.hipLeftY[i].isNaN()) {
                    leftEntries.add(Entry(t, -signals.hipLeftY[i]))
                }
                if (!signals.hipRightY[i].isNaN()) {
                    rightEntries.add(Entry(t, -signals.hipRightY[i]))
                }
            }
        }

        val leftDataSet = LineDataSet(leftEntries, "Left Hip Y")
        leftDataSet.color = Color.parseColor("#3498DB")
        leftDataSet.setDrawCircles(false)
        leftDataSet.setDrawValues(false)
        leftDataSet.lineWidth = 1.5f

        val rightDataSet = LineDataSet(rightEntries, "Right Hip Y")
        rightDataSet.color = Color.parseColor("#E74C3C")
        rightDataSet.setDrawCircles(false)
        rightDataSet.setDrawValues(false)
        rightDataSet.lineWidth = 1.5f

        chartHipY.data = LineData(leftDataSet, rightDataSet)
        addStrideHighlights(chartHipY, signals, strides)
        chartHipY.invalidate()
    }

    private fun populateTrunkChart(signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        val entries = mutableListOf<Entry>()

        for (i in signals.timestamps.indices) {
            val t = signals.timestamps[i]
            val v = signals.trunkAngle[i]
            if (!t.isNaN() && !v.isNaN()) {
                entries.add(Entry(t, v))
            }
        }

        val dataSet = LineDataSet(entries, "Trunk Angle")
        dataSet.color = Color.parseColor("#9B59B6")
        dataSet.setDrawCircles(false)
        dataSet.setDrawValues(false)
        dataSet.lineWidth = 1.5f

        chartTrunk.data = LineData(dataSet)
        
        // Add zero line
        val zeroLine = LimitLine(0f, "")
        zeroLine.lineColor = Color.GRAY
        zeroLine.lineWidth = 0.8f
        chartTrunk.axisLeft.addLimitLine(zeroLine)
        
        addStrideHighlights(chartTrunk, signals, strides)
        chartTrunk.invalidate()
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
        
        val selectedIndices = localSelectedIndices ?: emptyList()

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
