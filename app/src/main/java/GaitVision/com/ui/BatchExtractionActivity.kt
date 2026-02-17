package GaitVision.com.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import GaitVision.com.R
import GaitVision.com.batch.MetadataRow
import GaitVision.com.batch.listFolderContents
import GaitVision.com.batch.parseMetadataCsv
import GaitVision.com.batch.validateFolderMatchesMetadata
import GaitVision.com.gait.GaitCsvExporter
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.runVideoPipelineHeadless
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets

class BatchExtractionActivity : BaseActivity() {

    companion object {
        private const val TAG = "BatchExtract"
    }

    private lateinit var tvStatus: TextView
    private lateinit var tvBatchOverall: TextView
    private lateinit var tvBatchCurrentVideo: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var progressBarCurrent: ProgressBar
    private lateinit var btnPickFolder: Button
    private lateinit var btnPickOutput: Button
    private lateinit var btnRun: Button
    private lateinit var btnCancel: Button

    private var folderTreeUri: Uri? = null
    private var outputUri: Uri? = null
    private var metadataRows: List<MetadataRow> = emptyList()
    private var videoUriByName: Map<String, Uri> = emptyMap()
    private var extractionJob: Job? = null

    private val folderPicker = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        uri?.let { onFolderPicked(it) }
    }

    private val outputPicker = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        uri?.let { onOutputPicked(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_batch_extraction)

        setupCommonHeader("Batch Extract")
        NavigationHelper.setupBottomNav(this)

        tvStatus = findViewById(R.id.tvBatchStatus)
        tvBatchOverall = findViewById(R.id.tvBatchOverall)
        tvBatchCurrentVideo = findViewById(R.id.tvBatchCurrentVideo)
        progressBar = findViewById(R.id.progressBatch)
        progressBarCurrent = findViewById(R.id.progressBatchCurrent)
        btnPickFolder = findViewById(R.id.btnBatchPickFolder)
        btnPickOutput = findViewById(R.id.btnBatchPickOutput)
        btnRun = findViewById(R.id.btnBatchRun)
        btnCancel = findViewById(R.id.btnBatchCancel)

        progressBar.visibility = android.view.View.GONE
        progressBarCurrent.visibility = android.view.View.GONE
        tvBatchOverall.visibility = android.view.View.GONE
        tvBatchCurrentVideo.visibility = android.view.View.GONE
        btnCancel.visibility = android.view.View.GONE

        btnPickFolder.setOnClickListener { folderPicker.launch(null) }
        btnPickOutput.setOnClickListener {
            outputPicker.launch("extraction_results_${System.currentTimeMillis()}.csv")
        }
        btnRun.setOnClickListener { startExtraction() }
        btnCancel.setOnClickListener { extractionJob?.cancel() }
    }

    private fun onFolderPicked(uri: Uri) {
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {}
        folderTreeUri = uri
        val contents = listFolderContents(this, uri)
        when {
            contents.error != null -> {
                tvStatus.text = "Error: ${contents.error}"
                metadataRows = emptyList()
                videoUriByName = emptyMap()
            }
            contents.metadataUri == null -> {
                tvStatus.text = "metadata.csv not found in folder"
                metadataRows = emptyList()
                videoUriByName = emptyMap()
            }
            else -> {
                contentResolver.openInputStream(contents.metadataUri!!)?.use { stream ->
                    metadataRows = parseMetadataCsv(stream)
                } ?: run {
                    metadataRows = emptyList()
                }
                videoUriByName = contents.videoUris.toMap()
                val (ok, err) = validateFolderMatchesMetadata(
                    videoUriByName.keys,
                    metadataRows
                )
                if (!ok) {
                    tvStatus.text = "Validation failed: $err"
                } else {
                    tvStatus.text = "Folder OK: ${metadataRows.size} videos"
                }
            }
        }
    }

    private fun onOutputPicked(uri: Uri) {
        try {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            )
        } catch (_: SecurityException) {}
        outputUri = uri
        tvStatus.text = (tvStatus.text.toString() + "\nOutput: ${uri.lastPathSegment?.takeLast(40) ?: "selected"}")
    }

    private fun startExtraction() {
        val folder = folderTreeUri
        val output = outputUri
        if (folder == null || output == null) {
            Toast.makeText(this, "Pick folder and output file first", Toast.LENGTH_SHORT).show()
            return
        }
        val (ok, err) = validateFolderMatchesMetadata(
            videoUriByName.keys,
            metadataRows
        )
        if (!ok) {
            Toast.makeText(this, err, Toast.LENGTH_LONG).show()
            return
        }
        btnPickFolder.isEnabled = false
        btnPickOutput.isEnabled = false
        btnRun.isEnabled = false
        btnCancel.visibility = android.view.View.VISIBLE
        progressBar.visibility = android.view.View.VISIBLE
        progressBarCurrent.visibility = android.view.View.VISIBLE
        tvBatchOverall.visibility = android.view.View.VISIBLE
        tvBatchCurrentVideo.visibility = android.view.View.VISIBLE
        progressBar.max = metadataRows.size
        progressBar.progress = 0
        tvBatchOverall.text = "Overall: 0/${metadataRows.size} videos"

        extractionJob = lifecycleScope.launch {
            try {
                runExtraction(folder, output)
            } catch (e: Exception) {
                Log.e(TAG, "Batch extraction failed", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@BatchExtractionActivity, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                withContext(Dispatchers.Main) {
                    btnPickFolder.isEnabled = true
                    btnPickOutput.isEnabled = true
                    btnRun.isEnabled = true
                    btnCancel.visibility = android.view.View.GONE
                    progressBar.visibility = android.view.View.GONE
                    progressBarCurrent.visibility = android.view.View.GONE
                    tvBatchOverall.visibility = android.view.View.GONE
                    tvBatchCurrentVideo.visibility = android.view.View.GONE
                    tvStatus.text = "Done"
                }
            }
        }
    }

    private suspend fun runExtraction(folderUri: Uri, outputUri: Uri) {
        val folderDoc = DocumentFile.fromTreeUri(this, folderUri)
            ?: throw IllegalStateException("Invalid folder")
        contentResolver.openOutputStream(outputUri, "w")?.use { outputStream ->
            val writer = OutputStreamWriter(outputStream, StandardCharsets.UTF_8)
            GaitCsvExporter.writeBatchHeader(writer)
            writer.flush()
            val total = metadataRows.size
            for ((index, meta) in metadataRows.withIndex()) {
                withContext(Dispatchers.Main) {
                    tvBatchOverall.text = "Overall: $index/$total videos"
                    tvBatchCurrentVideo.text = "Current: ${meta.filename}"
                    progressBar.progress = index
                    progressBarCurrent.max = 100
                    progressBarCurrent.progress = 0
                }
                val videoUri = videoUriByName[meta.filename]
                if (videoUri == null) {
                    val videoId = meta.filename.substringBeforeLast(".")
                    GaitCsvExporter.writeBatchRow(
                        writer, videoId, false, 0L, meta,
                        null, GaitDiagnostics.empty(), "File not found"
                    )
                    writer.flush()
                    continue
                }
                val videoId = meta.filename.substringBeforeLast(".")
                val startMs = System.currentTimeMillis()
                val progressCallback: (totalFrames: Int, currentFrame: Int) -> Unit = { totalFrames, currentFrame ->
                    runOnUiThread {
                        tvBatchCurrentVideo.text = "Current: ${meta.filename} — frame $currentFrame/$totalFrames"
                        progressBarCurrent.max = totalFrames
                        progressBarCurrent.progress = currentFrame
                    }
                }
                val result = runVideoPipelineHeadless(this, videoUri, progressCallback)
                val elapsed = System.currentTimeMillis() - startMs
                if (result == null) {
                    GaitCsvExporter.writeBatchRow(
                        writer, videoId, false, elapsed, meta,
                        null, GaitDiagnostics.empty(), "Pipeline failed"
                    )
                } else {
                    val (features, diagnostics) = result
                    GaitCsvExporter.writeBatchRow(
                        writer, videoId, features != null, elapsed, meta,
                        features, diagnostics, ""
                    )
                }
                writer.flush()
                withContext(Dispatchers.Main) {
                    progressBar.progress = index + 1
                    tvBatchOverall.text = "Overall: ${index + 1}/$total videos"
                }
            }
        } ?: throw IllegalStateException("Could not open output file")
    }
}
