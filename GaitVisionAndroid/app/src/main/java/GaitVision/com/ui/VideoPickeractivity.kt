package GaitVision.com.ui

import GaitVision.com.AnalysisSession
import GaitVision.com.R
import GaitVision.com.extractRecordingDate
import android.app.DatePickerDialog
import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

class VideoPickerActivity : BaseActivity() {

    private var selectedVideo: Uri? = null
    private var pendingRecordingUri: Uri? = null

    private lateinit var videoView: VideoView
    private lateinit var tvPlaceholder: TextView
    private lateinit var tvStatus: TextView
    private lateinit var btnContinue: LinearLayout
    private lateinit var cardContinue: View
    private lateinit var tvDate: TextView
    private lateinit var btnEditDate: ImageButton

    private var selectedDateMillis: Long = Calendar.getInstance().apply {
        set(Calendar.HOUR_OF_DAY, 0)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
    }.timeInMillis

    private val dateFormat =
        SimpleDateFormat("MMM d, yyyy", Locale.getDefault())

    private val recordVideoLauncher =
        registerForActivityResult(ActivityResultContracts.CaptureVideo()) { success ->
            val recordedUri = pendingRecordingUri

            if (success && recordedUri != null) {
                finalizeRecording(recordedUri)

                showSelectedVideo(
                    recordedUri,
                    "Recording saved successfully"
                )

                selectedDateMillis = startOfToday()
                updateDateDisplay()

                Log.d("VideoPicker", "Recorded video saved: $recordedUri")
            } else {
                recordedUri?.let { deleteRecording(it) }

                tvStatus.text = "Recording cancelled"

                Toast.makeText(
                    this,
                    "Recording was cancelled or could not be saved.",
                    Toast.LENGTH_LONG
                ).show()

                Log.w("VideoPicker", "Recording cancelled or unsuccessful")
            }

            pendingRecordingUri = null
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_picker)

        videoView = findViewById(R.id.videoView)
        tvPlaceholder = findViewById(R.id.tvPlaceholder)
        tvStatus = findViewById(R.id.tvStatus)
        btnContinue = findViewById(R.id.btnContinue)

        // The parent CardView controls visibility.
        cardContinue = btnContinue.parent as View

        tvDate = findViewById(R.id.tvDate)
        btnEditDate = findViewById(R.id.btnEditDate)

        setupCommonHeader("Select Video")
        setupButtons()
    }

    private fun setupButtons() {
        val pickVideoLauncher =
            registerForActivityResult(
                ActivityResultContracts.PickVisualMedia()
            ) { uri: Uri? ->
                Log.d("VideoPicker", "Selected URI: $uri")

                if (uri != null) {
                    showSelectedVideo(
                        uri,
                        "Video loaded successfully"
                    )

                    lifecycleScope.launch(Dispatchers.IO) {
                        val metadataDate =
                            extractRecordingDate(
                                this@VideoPickerActivity,
                                uri
                            )

                        if (metadataDate != null) {
                            withContext(Dispatchers.Main) {
                                selectedDateMillis = metadataDate
                                updateDateDisplay()
                            }
                        }
                    }
                } else {
                    tvStatus.text = "No video selected"

                    Toast.makeText(
                        this,
                        "No video was selected.",
                        Toast.LENGTH_SHORT
                    ).show()

                    Log.w("VideoPicker", "No gallery video selected")
                }
            }

        findViewById<LinearLayout>(R.id.btnPick).setOnClickListener {
            val request =
                PickVisualMediaRequest.Builder()
                    .setMediaType(
                        ActivityResultContracts.PickVisualMedia.VideoOnly
                    )
                    .build()

            pickVideoLauncher.launch(request)
        }

        findViewById<LinearLayout>(R.id.btnRecord).setOnClickListener {
            startVideoRecording()
        }

        updateDateDisplay()

        btnEditDate.setOnClickListener {
            showDatePicker()
        }

        btnContinue.setOnClickListener {
            val uri = selectedVideo

            if (uri == null) {
                Toast.makeText(
                    this,
                    "Please select or record a video first.",
                    Toast.LENGTH_SHORT
                ).show()

                return@setOnClickListener
            }

            AnalysisSession.recordingDate = selectedDateMillis

            val shouldSave =
                intent.getBooleanExtra(
                    AnalysisActivity.EXTRA_SHOULD_SAVE,
                    true
                )

            val analysisIntent =
                Intent(this, AnalysisActivity::class.java).apply {
                    data = uri
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    putExtra(
                        AnalysisActivity.EXTRA_SHOULD_SAVE,
                        shouldSave
                    )
                }

            startActivity(analysisIntent)
        }
    }

    private fun startVideoRecording() {
        val recordingUri = createRecordingUri()

        if (recordingUri == null) {
            tvStatus.text = "Unable to start recording"

            Toast.makeText(
                this,
                "The app could not create a video file. Check your available storage.",
                Toast.LENGTH_LONG
            ).show()

            return
        }

        pendingRecordingUri = recordingUri
        tvStatus.text = "Opening camera..."

        try {
            recordVideoLauncher.launch(recordingUri)
        } catch (exception: Exception) {
            Log.e(
                "VideoPicker",
                "Could not open the camera",
                exception
            )

            deleteRecording(recordingUri)
            pendingRecordingUri = null
            tvStatus.text = "Camera unavailable"

            Toast.makeText(
                this,
                "The camera could not be opened.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun createRecordingUri(): Uri? {
        val fileName =
            "gait_video_${System.currentTimeMillis()}.mp4"

        val values = ContentValues().apply {
            put(
                MediaStore.Video.Media.DISPLAY_NAME,
                fileName
            )

            put(
                MediaStore.Video.Media.MIME_TYPE,
                "video/mp4"
            )

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.Video.Media.RELATIVE_PATH,
                    "Movies/GaitVision"
                )

                put(
                    MediaStore.Video.Media.IS_PENDING,
                    1
                )
            }
        }

        return try {
            contentResolver.insert(
                MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
                values
            )
        } catch (exception: Exception) {
            Log.e(
                "VideoPicker",
                "Could not create recording URI",
                exception
            )

            null
        }
    }

    private fun finalizeRecording(uri: Uri) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return
        }

        try {
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.IS_PENDING, 0)
            }

            contentResolver.update(
                uri,
                values,
                null,
                null
            )
        } catch (exception: Exception) {
            Log.e(
                "VideoPicker",
                "Could not finalize recording",
                exception
            )
        }
    }

    private fun deleteRecording(uri: Uri) {
        try {
            contentResolver.delete(
                uri,
                null,
                null
            )
        } catch (exception: Exception) {
            Log.e(
                "VideoPicker",
                "Could not delete cancelled recording",
                exception
            )
        }
    }

    private fun showSelectedVideo(
        videoUri: Uri,
        statusMessage: String
    ) {
        selectedVideo = videoUri

        videoView.visibility = View.VISIBLE
        tvPlaceholder.visibility = View.GONE
        cardContinue.visibility = View.VISIBLE
        tvStatus.text = statusMessage

        videoView.setVideoURI(videoUri)

        videoView.setOnPreparedListener {
            // Display the first frame instead of autoplaying.
            videoView.seekTo(1)
        }

        videoView.setOnErrorListener { _, what, extra ->
            Log.e(
                "VideoPicker",
                "Video preview error: what=$what, extra=$extra"
            )

            tvStatus.text = "Unable to preview video"

            Toast.makeText(
                this,
                "The video could not be previewed. Please try again.",
                Toast.LENGTH_LONG
            ).show()

            true
        }
    }

    private fun showDatePicker() {
        val calendar = Calendar.getInstance().apply {
            timeInMillis = selectedDateMillis
        }

        val year = calendar.get(Calendar.YEAR)
        val month = calendar.get(Calendar.MONTH)
        val dayOfMonth = calendar.get(Calendar.DAY_OF_MONTH)

        val datePickerDialog =
            DatePickerDialog(
                this,
                R.style.Theme_GaitVision_DatePicker,
                { _, selectedYear, selectedMonth, selectedDay ->
                    val selectedCalendar =
                        Calendar.getInstance().apply {
                            set(
                                selectedYear,
                                selectedMonth,
                                selectedDay
                            )

                            set(Calendar.HOUR_OF_DAY, 0)
                            set(Calendar.MINUTE, 0)
                            set(Calendar.SECOND, 0)
                            set(Calendar.MILLISECOND, 0)
                        }

                    selectedDateMillis =
                        selectedCalendar.timeInMillis

                    updateDateDisplay()
                },
                year,
                month,
                dayOfMonth
            )

        datePickerDialog.datePicker.maxDate =
            System.currentTimeMillis()

        datePickerDialog.show()

        val white =
            androidx.core.content.ContextCompat.getColor(
                this,
                R.color.text_white
            )

        datePickerDialog
            .getButton(DatePickerDialog.BUTTON_POSITIVE)
            ?.setTextColor(white)

        datePickerDialog
            .getButton(DatePickerDialog.BUTTON_NEGATIVE)
            ?.setTextColor(white)

        datePickerDialog
            .getButton(DatePickerDialog.BUTTON_NEUTRAL)
            ?.setTextColor(white)
    }

    private fun updateDateDisplay() {
        tvDate.text =
            if (isToday(selectedDateMillis)) {
                getString(R.string.label_date_today)
            } else {
                dateFormat.format(Date(selectedDateMillis))
            }
    }

    private fun startOfToday(): Long {
        return Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
    }

    private fun isToday(inTimeMillis: Long): Boolean {
        val calendarTarget =
            Calendar.getInstance().apply {
                timeInMillis = inTimeMillis
            }

        val calendarToday = Calendar.getInstance()

        return calendarTarget.get(Calendar.YEAR) ==
                calendarToday.get(Calendar.YEAR) &&
                calendarTarget.get(Calendar.DAY_OF_YEAR) ==
                calendarToday.get(Calendar.DAY_OF_YEAR)
    }
}
