package GaitVision.com.ui

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import GaitVision.com.R
import GaitVision.com.data.*
import GaitVision.com.participantId
import GaitVision.com.participantHeight
import GaitVision.com.currentPatientId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

class PatientProfileActivity : AppCompatActivity() {

    private lateinit var patientDao: PatientDao
    private lateinit var videoDao: VideoDao
    private lateinit var gaitScoreDao: GaitScoreDao
    
    private lateinit var tvPatientId: TextView
    private lateinit var tvPatientName: TextView
    private lateinit var tvAge: TextView
    private lateinit var tvGender: TextView
    private lateinit var tvHeight: TextView
    private lateinit var tvCreatedAt: TextView
    private lateinit var tvTotalAnalyses: TextView
    private lateinit var tvAvgScore: TextView
    private lateinit var rvAnalyses: RecyclerView
    private lateinit var emptyAnalysisState: View
    
    private lateinit var adapter: AnalysisAdapter
    private var patientIdArg: Int = -1
    private var currentPatient: Patient? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_patient_profile)

        patientIdArg = intent.getLongExtra("patientId", -1).toInt()
        if (patientIdArg <= 0) {
            Toast.makeText(this, "Invalid patient", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val database = AppDatabase.getDatabase(this)
        patientDao = database.patientDao()
        videoDao = database.videoDao()
        gaitScoreDao = database.gaitScoreDao()

        initViews()
        setupRecyclerView()
        loadPatientData()
    }

    private fun initViews() {
        tvPatientId = findViewById(R.id.tvPatientId)
        tvPatientName = findViewById(R.id.tvPatientName)
        tvAge = findViewById(R.id.tvAge)
        tvGender = findViewById(R.id.tvGender)
        tvHeight = findViewById(R.id.tvHeight)
        tvCreatedAt = findViewById(R.id.tvCreatedAt)
        tvTotalAnalyses = findViewById(R.id.tvTotalAnalyses)
        tvAvgScore = findViewById(R.id.tvAvgScore)
        rvAnalyses = findViewById(R.id.rvAnalyses)
        emptyAnalysisState = findViewById(R.id.emptyAnalysisState)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener {
            finish()
        }

        findViewById<ImageButton>(R.id.btnEdit).setOnClickListener {
            val intent = Intent(this, PatientCreateActivity::class.java)
            intent.putExtra("patientId", patientIdArg)
            startActivity(intent)
        }

        findViewById<ImageButton>(R.id.btnDelete).setOnClickListener {
            showDeleteConfirmation()
        }

        findViewById<Button>(R.id.btnNewAnalysis).setOnClickListener {
            startNewAnalysis()
        }
    }

    private fun setupRecyclerView() {
        adapter = AnalysisAdapter(
            onViewVideo = { video ->
                // TODO: Open video player
                Toast.makeText(this, "Video: ${video.editedVideoPath}", Toast.LENGTH_SHORT).show()
            },
            onViewCharts = { video, score ->
                // TODO: Open results activity with this specific analysis
                Toast.makeText(this, "Score: ${score?.overallScore}", Toast.LENGTH_SHORT).show()
            }
        )
        rvAnalyses.layoutManager = LinearLayoutManager(this)
        rvAnalyses.adapter = adapter
        rvAnalyses.isNestedScrollingEnabled = false
    }

    private fun loadPatientData() {
        lifecycleScope.launch {
            // Load patient
            val patient = withContext(Dispatchers.IO) {
                patientDao.getPatientById(patientIdArg)
            }

            if (patient == null) {
                Toast.makeText(this@PatientProfileActivity, "Patient not found", Toast.LENGTH_SHORT).show()
                finish()
                return@launch
            }

            currentPatient = patient
            displayPatientInfo(patient)

            // Load videos and scores
            videoDao.getVideosByPatientIdOrdered(patientIdArg).collectLatest { videos ->
                val analysisData = withContext(Dispatchers.IO) {
                    videos.map { video ->
                        val score = gaitScoreDao.getGaitScoreByVideoId(video.id)
                        Pair(video, score)
                    }
                }

                updateAnalysisList(analysisData)
                updateStats(analysisData)
            }
        }
    }

    private fun displayPatientInfo(patient: Patient) {
        tvPatientId.text = patient.participantId?.toString() ?: "N/A"
        tvPatientName.text = patient.fullName
        tvAge.text = if (patient.age != null) "${patient.age} years" else "—"
        tvGender.text = patient.gender ?: "—"
        
        val feet = patient.height / 12
        val inches = patient.height % 12
        tvHeight.text = "$feet'$inches\""
        
        val dateFormat = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
        tvCreatedAt.text = "Added: ${dateFormat.format(Date(patient.createdAt))}"
    }

    private fun updateAnalysisList(analyses: List<Pair<Video, GaitScore?>>) {
        if (analyses.isEmpty()) {
            rvAnalyses.visibility = View.GONE
            emptyAnalysisState.visibility = View.VISIBLE
        } else {
            rvAnalyses.visibility = View.VISIBLE
            emptyAnalysisState.visibility = View.GONE
            adapter.submitList(analyses)
        }
    }

    private fun updateStats(analyses: List<Pair<Video, GaitScore?>>) {
        tvTotalAnalyses.text = analyses.size.toString()
        
        val scores = analyses.mapNotNull { it.second?.overallScore }
        if (scores.isNotEmpty()) {
            val avgScore = scores.average().toInt()
            tvAvgScore.text = avgScore.toString()
        } else {
            tvAvgScore.text = "—"
        }
    }

    private fun startNewAnalysis() {
        currentPatient?.let { patient ->
            participantId = patient.participantId ?: 0 ?: 0
            participantHeight = patient.height
            currentPatientId = patient.participantId

            val intent = Intent(this, VideoPickerActivity::class.java)
            intent.putExtra("patientId", patient.participantId)
            intent.putExtra("fromPatientProfile", true)
            startActivity(intent)
        }
    }

    private fun showDeleteConfirmation() {
        AlertDialog.Builder(this)
            .setTitle("Delete Patient")
            .setMessage("Are you sure you want to delete this patient and all their analysis data? This cannot be undone.")
            .setPositiveButton("Delete") { _, _ ->
                deletePatient()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deletePatient() {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                patientDao.deletePatientById(patientIdArg)
            }
            Toast.makeText(this@PatientProfileActivity, "Patient deleted", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        loadPatientData()
    }
}

// Analysis Adapter
class AnalysisAdapter(
    private val onViewVideo: (Video) -> Unit,
    private val onViewCharts: (Video, GaitScore?) -> Unit
) : RecyclerView.Adapter<AnalysisAdapter.AnalysisViewHolder>() {

    private var analyses: List<Pair<Video, GaitScore?>> = emptyList()
    private val dateFormat = SimpleDateFormat("MMM d, yyyy • h:mm a", Locale.getDefault())

    fun submitList(newAnalyses: List<Pair<Video, GaitScore?>>) {
        analyses = newAnalyses
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): AnalysisViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_analysis_card, parent, false)
        return AnalysisViewHolder(view)
    }

    override fun onBindViewHolder(holder: AnalysisViewHolder, position: Int) {
        holder.bind(analyses[position])
    }

    override fun getItemCount() = analyses.size

    inner class AnalysisViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvDate: TextView = itemView.findViewById(R.id.tvDate)
        private val tvScore: TextView = itemView.findViewById(R.id.tvScore)
        private val scoreContainer: View = itemView.findViewById(R.id.scoreContainer)
        private val tvLeftKnee: TextView = itemView.findViewById(R.id.tvLeftKnee)
        private val tvRightKnee: TextView = itemView.findViewById(R.id.tvRightKnee)
        private val tvLeftHip: TextView = itemView.findViewById(R.id.tvLeftHip)
        private val tvRightHip: TextView = itemView.findViewById(R.id.tvRightHip)
        private val tvTorso: TextView = itemView.findViewById(R.id.tvTorso)
        private val btnViewVideo: Button = itemView.findViewById(R.id.btnViewVideo)
        private val btnViewCharts: Button = itemView.findViewById(R.id.btnViewCharts)

        fun bind(data: Pair<Video, GaitScore?>) {
            val (video, score) = data

            tvDate.text = dateFormat.format(Date(video.recordedAt))

            if (score != null) {
                tvScore.text = score.overallScore.toInt().toString()
                scoreContainer.setBackgroundResource(
                    when {
                        score.overallScore >= 80 -> R.drawable.score_badge_background // Green
                        score.overallScore >= 60 -> R.drawable.badge_background // Yellow-ish
                        else -> R.drawable.button_outline_background // Red-ish
                    }
                )
                
                tvLeftKnee.text = score.leftKneeScore?.let { "${it.toInt()}°" } ?: "—"
                tvRightKnee.text = score.rightKneeScore?.let { "${it.toInt()}°" } ?: "—"
                tvLeftHip.text = score.leftHipScore?.let { "${it.toInt()}°" } ?: "—"
                tvRightHip.text = score.rightHipScore?.let { "${it.toInt()}°" } ?: "—"
                tvTorso.text = score.torsoScore?.let { "${it.toInt()}°" } ?: "—"
            } else {
                tvScore.text = "—"
                tvLeftKnee.text = "—"
                tvRightKnee.text = "—"
                tvLeftHip.text = "—"
                tvRightHip.text = "—"
                tvTorso.text = "—"
            }

            btnViewVideo.setOnClickListener { onViewVideo(video) }
            btnViewCharts.setOnClickListener { onViewCharts(video, score) }
        }
    }
}

