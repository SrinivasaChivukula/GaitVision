package GaitVision.com.ui

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.LimitLine
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import GaitVision.com.R
import GaitVision.com.extractedSignals
import GaitVision.com.extractedStrides
import GaitVision.com.extractionDiagnostics
import GaitVision.com.stepSignalMode
import GaitVision.com.participantId
import GaitVision.com.gait.GaitCsvExporter
import GaitVision.com.gait.Signals

/**
 * Dashboard showing all computed signals for debugging and analysis.
 * Mirrors the PC pipeline's expanded_dashboard.py visualization.
 */
class SignalsDashboardActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SignalsDashboard"
    }

    private lateinit var tvStepMode: TextView
    private lateinit var tvValidStrides: TextView
    private lateinit var tvFrameValidity: TextView
    private lateinit var tvLegend: TextView
    private lateinit var btnSelectSignal: Button

    private lateinit var chartInterAnkle: LineChart
    private lateinit var chartKneeAngles: LineChart
    private lateinit var chartAnkleY: LineChart
    private lateinit var chartAnkleVy: LineChart
    private lateinit var chartHipY: LineChart
    private lateinit var chartTrunk: LineChart

    private var currentSignal = "INTER_ANKLE"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_signals_dashboard)

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
        findViewById<Button>(R.id.btnBack).setOnClickListener {
            finish()
        }

        btnSelectSignal.setOnClickListener {
            showSignalPopup()
        }

        findViewById<Button>(R.id.btnExportSignals).setOnClickListener {
            exportSignals()
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
            btnSelectSignal.text = item.title
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
        tvLegend.text = when (signalId) {
            1 -> "Horizontal distance between ankles | Green bands = Valid strides"
            2 -> "Blue = Left Knee, Red = Right Knee | Green bands = Valid strides"
            3 -> "Blue = Left Ankle Y, Red = Right Ankle Y (inverted) | Green bands = Valid strides"
            4 -> "Blue = Left Ankle Vy, Red = Right Ankle Vy | Green bands = Valid strides"
            5 -> "Blue = Left Hip Y, Red = Right Hip Y (inverted) | Green bands = Valid strides"
            6 -> "Trunk lean angle (degrees) | Green bands = Valid strides"
            else -> "Blue = Left, Red = Right | Green bands = Valid strides"
        }
    }

    private fun loadData() {
        val signals = extractedSignals
        val strides = extractedStrides
        val diagnostics = extractionDiagnostics

        if (signals == null) {
            tvStepMode.text = "No data"
            tvValidStrides.text = "--"
            tvFrameValidity.text = "--"
            Toast.makeText(this, "No signal data available", Toast.LENGTH_SHORT).show()
            return
        }

        // Update info card
        tvStepMode.text = stepSignalMode ?: "UNKNOWN"
        tvValidStrides.text = (strides?.count { it.isValid } ?: 0).toString()

        val validFrames = signals.isValid.count { it }
        val totalFrames = signals.isValid.size
        val validPct = if (totalFrames > 0) (validFrames * 100 / totalFrames) else 0
        tvFrameValidity.text = "$validPct%"

        // Populate all charts
        populateInterAnkleChart(signals, strides)
        populateKneeAnglesChart(signals, strides)
        populateAnkleYChart(signals, strides)
        populateAnkleVyChart(signals, strides)
        populateHipYChart(signals, strides)
        populateTrunkChart(signals, strides)

        Log.d(TAG, "Loaded ${signals.timestamps.size} frames, ${strides?.size ?: 0} strides")
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
     * Add green highlight bands for valid strides.
     * MPAndroidChart doesn't support axvspan directly, so we use LimitLines.
     */
    private fun addStrideHighlights(chart: LineChart, signals: Signals, strides: List<GaitVision.com.gait.Stride>?) {
        if (strides == null) return

        // Clear previous limit lines
        chart.xAxis.removeAllLimitLines()

        for (stride in strides) {
            if (!stride.isValid) continue

            val startTime = signals.timestamps.getOrNull(stride.startFrame) ?: continue
            val endTime = signals.timestamps.getOrNull(minOf(stride.endFrame, signals.timestamps.size - 1)) ?: continue

            if (startTime.isNaN() || endTime.isNaN()) continue

            // Add start line (green)
            val startLine = LimitLine(startTime, "")
            startLine.lineColor = Color.parseColor("#4CAF50")
            startLine.lineWidth = 1f
            startLine.enableDashedLine(10f, 5f, 0f)
            chart.xAxis.addLimitLine(startLine)

            // Add end line (green)
            val endLine = LimitLine(endTime, "")
            endLine.lineColor = Color.parseColor("#4CAF50")
            endLine.lineWidth = 1f
            endLine.enableDashedLine(10f, 5f, 0f)
            chart.xAxis.addLimitLine(endLine)
        }
    }

    private fun exportSignals() {
        val signals = extractedSignals
        if (signals == null) {
            Toast.makeText(this, "No signals to export", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val path = GaitCsvExporter.exportSignalsToCSV(
                context = this,
                signals = signals,
                participantId = participantId.toString()
            )

            if (path != null) {
                Toast.makeText(this, "Signals exported to Documents", Toast.LENGTH_SHORT).show()
                Log.d(TAG, "Exported signals to: $path")
            } else {
                Toast.makeText(this, "Failed to export signals", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting signals", e)
            Toast.makeText(this, "Export error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}
