package GaitVision.com.batch

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

private const val TAG = "BatchUtils"

/** Row from metadata.csv: filename, patient_id, subject_id, source, condition, severity, label, trial */
data class MetadataRow(
    val filename: String,
    val patientId: Int,
    val subjectId: String,
    val source: String,
    val condition: String,
    val severity: String,
    val label: String,
    val trial: String
)

/** Videos (name, uri) and metadata.csv URI if present. */
data class FolderContents(
    val videoUris: List<Pair<String, Uri>>,
    val metadataUri: Uri?,
    val error: String?
)

private val VIDEO_EXTENSIONS = setOf(".mov", ".mp4", ".MOV", ".MP4")

/** Parse metadata.csv. Expects standard header. */
fun parseMetadataCsv(inputStream: java.io.InputStream): List<MetadataRow> {
    val rows = mutableListOf<MetadataRow>()
    BufferedReader(InputStreamReader(inputStream, StandardCharsets.UTF_8)).use { reader ->
        val header = reader.readLine() ?: return emptyList()
        val cols = header.split(",").map { it.trim() }
        val filenameIdx = cols.indexOf("filename")
        val patientIdIdx = cols.indexOf("patient_id")
        val subjectIdIdx = cols.indexOf("subject_id")
        val sourceIdx = cols.indexOf("source")
        val conditionIdx = cols.indexOf("condition")
        val severityIdx = cols.indexOf("severity")
        val labelIdx = cols.indexOf("label")
        val trialIdx = cols.indexOf("trial")
        if (filenameIdx < 0 || patientIdIdx < 0 || subjectIdIdx < 0) {
            Log.e(TAG, "Missing required columns in metadata.csv")
            return emptyList()
        }
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            val parts = line!!.split(",").map { it.trim() }
            if (parts.size < 3) continue
            val filename = parts.getOrElse(filenameIdx) { "" }
            val patientId = parts.getOrElse(patientIdIdx) { "0" }.toIntOrNull() ?: 0
            val subjectId = parts.getOrElse(subjectIdIdx) { "" }
            val source = parts.getOrElse(sourceIdx) { "" }
            val condition = parts.getOrElse(conditionIdx) { "" }
            val severity = parts.getOrElse(severityIdx) { "" }
            val label = parts.getOrElse(labelIdx) { "" }
            val trial = parts.getOrElse(trialIdx) { "" }
            if (filename.isNotEmpty()) {
                rows.add(
                    MetadataRow(
                        filename = filename,
                        patientId = patientId,
                        subjectId = subjectId,
                        source = source,
                        condition = condition,
                        severity = severity,
                        label = label,
                        trial = trial
                    )
                )
            }
        }
    }
    return rows
}

/** List folder via SAF tree URI. Flat folder only. */
fun listFolderContents(context: Context, treeUri: Uri): FolderContents {
    return try {
        val doc = DocumentFile.fromTreeUri(context, treeUri)
            ?: return FolderContents(emptyList(), null, "Invalid folder URI")
        if (!doc.isDirectory) {
            return FolderContents(emptyList(), null, "Selected path is not a folder")
        }
        val videoUris = mutableListOf<Pair<String, Uri>>()
        var metadataUri: Uri? = null
        for (file in doc.listFiles()) {
            val name = file.name ?: continue
            when {
                name.equals("metadata.csv", ignoreCase = true) -> metadataUri = file.uri
                VIDEO_EXTENSIONS.any { name.endsWith(it) } -> videoUris.add(name to file.uri)
            }
        }
        if (metadataUri == null) {
            return FolderContents(videoUris, null, "metadata.csv not found in folder")
        }
        FolderContents(videoUris, metadataUri, null)
    } catch (e: Exception) {
        Log.e(TAG, "Error listing folder", e)
        FolderContents(emptyList(), null, e.message ?: "Failed to list folder")
    }
}

/** Check folder matches metadata (no extra/missing videos). */
fun validateFolderMatchesMetadata(
    videoNamesInFolder: Set<String>,
    metadataRows: List<MetadataRow>
): Pair<Boolean, String> {
    val metadataFilenames = metadataRows.map { it.filename }.toSet()
    val inMetadataNotInFolder = metadataFilenames - videoNamesInFolder
    val inFolderNotInMetadata = videoNamesInFolder - metadataFilenames
    return when {
        inMetadataNotInFolder.isNotEmpty() ->
            false to "Metadata lists files not in folder: ${inMetadataNotInFolder.take(5).joinToString()}" +
                if (inMetadataNotInFolder.size > 5) " (and ${inMetadataNotInFolder.size - 5} more)" else ""
        inFolderNotInMetadata.isNotEmpty() ->
            false to "Folder has videos not in metadata: ${inFolderNotInMetadata.take(5).joinToString()}" +
                if (inFolderNotInMetadata.size > 5) " (and ${inFolderNotInMetadata.size - 5} more)" else ""
        else -> true to ""
    }
}
