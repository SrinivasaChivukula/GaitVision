package GaitVision.com

import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.PopupMenu
import android.widget.TextView
import androidx.activity.ComponentActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToLong
import kotlin.math.sqrt

class LastActivity : ComponentActivity()
{
    fun loadFloatBinFile(context: Context, filename: String): FloatArray {
        val inputStream = context.assets.open(filename)
        val bytes = inputStream.readBytes()
        inputStream.close()

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val numFloats = bytes.size / 4

        val result = FloatArray(numFloats)
        for (i in 0 until numFloats) {
            result[i] = buffer.float
        }

        return result
    }

    fun loadNpyFloatArray(assetStream: InputStream): FloatArray {
        val header = ByteArray(128)
        assetStream.read(header)

        val data = assetStream.readBytes()
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

        val floatList = mutableListOf<Float>()
        while (buffer.hasRemaining()) {
            floatList.add(buffer.float)
        }

        return floatList.toFloatArray()
    }

    fun euclideanDistance(a: FloatArray, b: FloatArray): Float {
        var sum = 0f

        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }

        return sqrt(sum)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_last)

        val inputData = floatArrayOf(
            leftKneeMinAngles.average().toFloat(),
            leftKneeMaxAngles.average().toFloat(),
            rightKneeMinAngles.average().toFloat(),
            rightKneeMaxAngles.average().toFloat(),
            torsoMinAngles.average().toFloat(),
            torsoMaxAngles.average().toFloat(),
            calcStrideLengthAvg(participantHeight.toFloat() * 39.37F),
            leftKneeMaxAngles.average().toFloat() - leftKneeMinAngles.average().toFloat(),
            rightKneeMaxAngles.average().toFloat() - rightKneeMinAngles.average().toFloat()
        )

        val tfliteModel = FileUtil.loadMappedFile(this, "encoder_model.tflite")
        val interpreter = Interpreter(tfliteModel)
        val scalerMean = loadFloatBinFile(this, "scaler_mean.bin")
        val scalerScale = loadFloatBinFile(this, "scaler_scale.bin")
        val cleanCentroidStream = assets.open("clean_centroid.npy")
        val impairedCentroidStream = assets.open("impaired_centroid.npy")
        val cleanCentroid = loadNpyFloatArray(cleanCentroidStream)
        val impairedCentroid = loadNpyFloatArray(impairedCentroidStream)

        val minScaleValue = 1e-15f
        val safeScalerScale = scalerScale.map {
            if (it < minScaleValue) minScaleValue else it
        }.toFloatArray()

        val scaledInput = FloatArray(inputData.size) { i ->
            (inputData[i] - scalerMean[i]) / safeScalerScale[i]
        }

        val output = Array(1) { FloatArray(2) }
        val input = arrayOf(scaledInput)

        interpreter.run(input, output)

        Log.d("ErrorCheck", "ScalerMean: ${scalerMean.contentToString()}")
        Log.d("ErrorCheck", "ScalerScale: ${scalerScale.contentToString()}")
        Log.d("ErrorCheck", "InputData: ${inputData.contentToString()}")
        Log.d("ErrorCheck", "ScaledInputData: ${input[0].joinToString(", ")}")
        Log.d("ErrorCheck", "Output: ${output[0].contentToString()} Length: ${output[0].size}")
        Log.d("ErrorCheck", "Clean Centroid: ${cleanCentroid.contentToString()} Length: ${cleanCentroid.size}")
        Log.d("ErrorCheck", "Impaired Centroid: ${impairedCentroid.contentToString()} Length: ${impairedCentroid.size}")

        val distClean = euclideanDistance(output[0], cleanCentroid)
        val distImpaired = euclideanDistance(output[0], impairedCentroid)
        Log.d("ErrorCheck", "DistClean: $distClean")
        Log.d("ErrorCheck", "DistImpaired: $distImpaired")

        val gaitIndexUnscaled = 1 - (distClean / (distClean + distImpaired))
        val gaitIndexScaled = gaitIndexUnscaled * 100

        Log.d("ErrorCheck", "Gait Index (Unscaled): $gaitIndexUnscaled")
        Log.d("ErrorCheck", "Gait Index (Scaled): $gaitIndexScaled")

        println("Gait Index (Unscaled): $gaitIndexUnscaled")
        println("Gait Index (Scaled): $gaitIndexScaled")

        val scoreTextView = findViewById<TextView>(R.id.score_textview)
        scoreTextView.text = gaitIndexScaled.roundToLong().toString()

        val reviewCheckBox = findViewById<CheckBox>(R.id.professional_review_checkbox)
        val exportButton = findViewById<Button>(R.id.submit_id_btn)

        exportButton.isEnabled = false

        reviewCheckBox.setOnCheckedChangeListener { _, isChecked ->
            exportButton.isEnabled = isChecked
        }

        val chooseGraphBtn = findViewById<Button>(R.id.select_graph_btn)
        val popupMenu = PopupMenu(this, chooseGraphBtn)
        popupMenu.menuInflater.inflate(R.menu.popup_menu_2, popupMenu.menu)

        val hipGraph = findViewById<com.github.mikephil.charting.charts.LineChart>(R.id.lineChartHip)
        val kneeGraph = findViewById<com.github.mikephil.charting.charts.LineChart>(R.id.lineChartKnee)
        val ankleGraph = findViewById<com.github.mikephil.charting.charts.LineChart>(R.id.lineChartAnkle)
        val torsoGraph = findViewById<com.github.mikephil.charting.charts.LineChart>(R.id.lineChartTorso)

        plotLineGraph(kneeGraph, leftKneeAngles, rightKneeAngles, "Left Knee Angles", "Right Knee Angles")
        plotLineGraph(ankleGraph, leftAnkleAngles, rightAnkleAngles, "Left Ankle Angles", "Right Ankle Angles")
        plotLineGraph(hipGraph, leftHipAngles, rightHipAngles, "Left Hip Angles", "Right Hip Angles")
        plotLineGraph(torsoGraph, torsoAngles, torsoAngles, "Torso Angles", "Torso Angles")

        popupMenu.setOnMenuItemClickListener { menuItem ->
            val id = menuItem.itemId

            if (id == R.id.menu_hip) {
                chooseGraphBtn.text = "HIP GRAPH"

                hipGraph.visibility = View.VISIBLE
                kneeGraph.visibility = View.INVISIBLE
                ankleGraph.visibility = View.INVISIBLE
                torsoGraph.visibility = View.INVISIBLE
            }
            else if (id == R.id.menu_knee) {
                chooseGraphBtn.text = "KNEE GRAPH"

                hipGraph.visibility = View.INVISIBLE
                kneeGraph.visibility = View.VISIBLE
                ankleGraph.visibility = View.INVISIBLE
                torsoGraph.visibility = View.INVISIBLE
            }
            else if (id == R.id.menu_ankle) {
                chooseGraphBtn.text = "ANKLE GRAPH"

                hipGraph.visibility = View.INVISIBLE
                kneeGraph.visibility = View.INVISIBLE
                ankleGraph.visibility = View.VISIBLE
                torsoGraph.visibility = View.INVISIBLE
            }
            else if (id == R.id.menu_torso) {
                chooseGraphBtn.text = "TORSO GRAPH"

                hipGraph.visibility = View.INVISIBLE
                kneeGraph.visibility = View.INVISIBLE
                ankleGraph.visibility = View.INVISIBLE
                torsoGraph.visibility = View.VISIBLE
            }

            false
        }

        chooseGraphBtn.setOnClickListener {
            popupMenu.show()
        }

        val fileData: List<MutableList<Float>> = mutableListOf(
            leftHipAngles,
            rightHipAngles,
            leftKneeAngles,
            rightKneeAngles,
            leftAnkleAngles,
            rightAnkleAngles,
            torsoAngles
        )

        val angleNames = listOf(
            "LeftHip",
            "RightHip",
            "LeftKnee",
            "RightKnee",
            "LeftAnkle",
            "RightAnkle",
            "Torso"
        )

        exportButton.setOnClickListener {
            for (i in fileData.indices) {
                val fileName = buildString {
                    append(participantId)
                    append("_")
                    append(angleNames[i])
                    append(".csv")
                }

                writeToFile(fileName, fileData[i])
                renameTo(participantId)
            }

            val builder: AlertDialog.Builder = AlertDialog.Builder(this)
            builder
                .setMessage("CSV Files saved to Documents as ParticipantID_GraphName.csv.\n\nUpdated video saved to Videos as ParticipantID_video.mp4")
                .setTitle("Successfully Exported")

            val dialog: AlertDialog = builder.create()
            dialog.show()
        }

        val mainMenuBtn = findViewById<Button>(R.id.main_mnu_btn)
        mainMenuBtn.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        val sharedPref = getSharedPreferences("HelpPrefs", Context.MODE_PRIVATE)
        val isHelpShown = sharedPref.getBoolean("Help03Shown", false)

        if (!isHelpShown) {
            showHelpDialog()

            val editor = sharedPref.edit()
            editor.putBoolean("Help03Shown", true)
            editor.apply()
        }

        val help03Btn = findViewById<Button>(R.id.help03_btn)
        help03Btn.setOnClickListener {
            showHelpDialog()
        }
    }

    private fun showHelpDialog() {
        val dialogBinding = layoutInflater.inflate(R.layout.help03_dialog, null)

        val myDialog = Dialog(this)
        myDialog.setContentView(dialogBinding)

        myDialog.setCancelable(false)
        myDialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        myDialog.show()

        val yes03Btn = dialogBinding.findViewById<Button>(R.id.help03_yes)
        yes03Btn.setOnClickListener {
            myDialog.dismiss()
        }
    }

    private fun writeToFile(fileName: String, fileData: MutableList<Float>) {
        val fileDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val outputFile = File(fileDirectory, fileName)

        FileOutputStream(outputFile).use { output ->
            val identifiersText = "Frame #,Angle\n"
            output.write(identifiersText.toByteArray())
            for (i in 0 until fileData.size) {
                val floatData = fileData[i].toString()
                val index = i.toString()
                output.write(index.toByteArray())
                output.write(",".toByteArray())
                output.write(floatData.toByteArray())
                output.write("\n".toByteArray())
            }
        }
    }

    private fun renameTo(participantId: String) {
        val vidName = buildString {
            append(participantId)
            append("_video.mp4")
        }

        val oldFilePath = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "edited_video.mp4")
        val newFilePath = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), vidName)

        editedUri = Uri.fromFile(newFilePath)

        oldFilePath.renameTo(newFilePath)
    }
}