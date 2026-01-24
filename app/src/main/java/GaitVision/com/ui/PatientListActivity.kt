package GaitVision.com.ui

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.Patient
import GaitVision.com.data.PatientDao
import GaitVision.com.data.VideoDao
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.runBlocking


class PatientListActivity : AppCompatActivity() {

    private lateinit var patientDao: PatientDao
    private lateinit var videoDao: VideoDao
    private lateinit var adapter: PatientAdapter
    private lateinit var rvPatients: RecyclerView
    private lateinit var emptyState: View
    private lateinit var tvPatientCount: TextView
    private lateinit var etSearch: EditText
    private lateinit var btnClearSearch: ImageButton
    private lateinit var tvHeaderId: TextView
    private lateinit var tvHeaderFirstName: TextView
    private lateinit var tvHeaderLastName: TextView
    private lateinit var tvHeaderAge: TextView
    private lateinit var tvHeaderVideos: TextView

    private var allPatients: List<Patient> = emptyList()
    private var currentFilter = "all"
    private var currentSortColumn: String = "id"
    private var currentSortOrder: Boolean = false // false for descending, true for ascending

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_patient_list)

        val database = AppDatabase.getDatabase(this)
        patientDao = database.patientDao()
        videoDao = database.videoDao()

        initViews()
        setupRecyclerView()
        setupSearch()
        setupFilters()
        loadPatients()
    }

    private fun initViews() {
        rvPatients = findViewById(R.id.rvPatients)
        emptyState = findViewById(R.id.emptyState)
        tvPatientCount = findViewById(R.id.tvPatientCount)
        etSearch = findViewById(R.id.etSearch)
        btnClearSearch = findViewById(R.id.btnClearSearch)
        tvHeaderId = findViewById(R.id.tvHeaderId)
        tvHeaderFirstName = findViewById(R.id.tvHeaderFirstName)
        tvHeaderLastName = findViewById(R.id.tvHeaderLastName)
        tvHeaderAge = findViewById(R.id.tvHeaderAge)
        tvHeaderVideos = findViewById(R.id.tvHeaderVideos)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener {
            finish()
        }

        findViewById<FloatingActionButton>(R.id.fabAddPatient).setOnClickListener {
            startActivity(Intent(this, PatientCreateActivity::class.java))
        }

        findViewById<Button>(R.id.btnAddFirstPatient).setOnClickListener {
            startActivity(Intent(this, PatientCreateActivity::class.java))
        }
    }

    private fun setupRecyclerView() {
        adapter = PatientAdapter(
            onPatientClick = { patient ->
                val intent = Intent(this, PatientProfileActivity::class.java)
                intent.putExtra("patientId", patient.id)
                startActivity(intent)
            },
            getVideoCount = { patientId ->
                 runBlocking(Dispatchers.IO) {
                 videoDao.getVideoCountForPatient(patientId)
                }
            }
        )
        rvPatients.layoutManager = LinearLayoutManager(this)
        rvPatients.adapter = adapter
    }

    private fun setupSearch() {
        etSearch.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val query = s?.toString() ?: ""
                btnClearSearch.visibility = if (query.isNotEmpty()) View.VISIBLE else View.GONE
                filterPatients(query)
            }
        })

        btnClearSearch.setOnClickListener {
            etSearch.text.clear()
        }
    }

    private fun setupFilters() {
        val btnAll = findViewById<Button>(R.id.btnFilterAll)
        val btnRecent = findViewById<Button>(R.id.btnFilterRecent)
        val btnWithVideos = findViewById<Button>(R.id.btnFilterWithVideos)

        fun updateFilterButtons(selected: String) {
            currentFilter = selected
            btnAll.setBackgroundResource(if (selected == "all") R.drawable.chip_selected_background else R.drawable.chip_background)
            btnRecent.setBackgroundResource(if (selected == "recent") R.drawable.chip_selected_background else R.drawable.chip_background)
            btnWithVideos.setBackgroundResource(if (selected == "videos") R.drawable.chip_selected_background else R.drawable.chip_background)
            filterPatients(etSearch.text.toString())
        }

        btnAll.setOnClickListener { updateFilterButtons("all") }
        btnRecent.setOnClickListener { updateFilterButtons("recent") }
        btnWithVideos.setOnClickListener { updateFilterButtons("videos") }

        fun updateSortUI() {
            val sortArrow = { ascending: Boolean -> if (ascending) " ↑" else " ↓" }

            tvHeaderId.text = "ID" + if (currentSortColumn == "id") sortArrow(currentSortOrder) else ""
            tvHeaderFirstName.text = "FIRST" + if (currentSortColumn == "firstName") sortArrow(currentSortOrder) else ""
            tvHeaderLastName.text = "LAST" + if (currentSortColumn == "lastName") sortArrow(currentSortOrder) else ""
            tvHeaderAge.text = "AGE" + if (currentSortColumn == "age") sortArrow(currentSortOrder) else ""
            tvHeaderVideos.text = "VIDEOS" + if (currentSortColumn == "videos") sortArrow(currentSortOrder) else ""
        }

        tvHeaderId.setOnClickListener {
            if (currentSortColumn == "id") {
                currentSortOrder = !currentSortOrder
            } else {
                currentSortColumn = "id"
                currentSortOrder = false // Default to descending for new sort
            }
            updateSortUI()
            filterPatients(etSearch.text.toString())
        }

        tvHeaderFirstName.setOnClickListener {
            if (currentSortColumn == "firstName") {
                currentSortOrder = !currentSortOrder
            } else {
                currentSortColumn = "firstName"
                currentSortOrder = true // Default to ascending for new sort
            }
            updateSortUI()
            filterPatients(etSearch.text.toString())
        }

        tvHeaderLastName.setOnClickListener {
            if (currentSortColumn == "lastName") {
                currentSortOrder = !currentSortOrder
            } else {
                currentSortColumn = "lastName"
                currentSortOrder = true // Default to ascending for new sort
            }
            updateSortUI()
            filterPatients(etSearch.text.toString())
        }

        tvHeaderAge.setOnClickListener {
            if (currentSortColumn == "age") {
                currentSortOrder = !currentSortOrder
            } else {
                currentSortColumn = "age"
                currentSortOrder = true // Default to ascending for new sort
            }
            updateSortUI()
            filterPatients(etSearch.text.toString())
        }

        tvHeaderVideos.setOnClickListener {
            if (currentSortColumn == "videos") {
                currentSortOrder = !currentSortOrder
            } else {
                currentSortColumn = "videos"
                currentSortOrder = true // Default to ascending for new sort
            }
            updateSortUI()
            filterPatients(etSearch.text.toString())
        }
        updateSortUI()
    }

    private fun loadPatients() {
        lifecycleScope.launch {
            patientDao.getAllPatients().collectLatest { patients ->
                allPatients = patients
                filterPatients(etSearch.text.toString())
            }
        }
    }

    private fun filterPatients(query: String) {
        lifecycleScope.launch {
            var filtered = allPatients

            // Apply search filter
            if (query.isNotEmpty()) {
                val lowerQuery = query.lowercase()
                filtered = filtered.filter {
                    it.firstName.lowercase().contains(lowerQuery) ||
                            it.lastName.lowercase().contains(lowerQuery) ||
                            it.participantId?.lowercase()?.contains(lowerQuery) == true
                }
            }

            // Apply category filter
            when (currentFilter) {
                "recent" -> {
                    val oneWeekAgo = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000)
                    filtered = filtered.filter { it.createdAt >= oneWeekAgo }
                }
                "videos" -> {
                    filtered = withContext(Dispatchers.IO) {
                        filtered.filter {
                            videoDao.getVideoCountForPatient(it.id) > 0
                        }
                    }
                }
            }

            // Apply sorting
            filtered = when (currentSortColumn) {
                "id" -> if (currentSortOrder) filtered.sortedBy { it.id } else filtered.sortedByDescending { it.id }
                "firstName" -> if (currentSortOrder) filtered.sortedBy { it.firstName.lowercase() } else filtered.sortedByDescending { it.firstName.lowercase() }
                "lastName" -> if (currentSortOrder) filtered.sortedBy { it.lastName.lowercase() } else filtered.sortedByDescending { it.lastName.lowercase() }
                "age" -> if (currentSortOrder) filtered.sortedBy { it.age } else filtered.sortedByDescending { it.age }
                "videos" -> {
                    val patientVideoCounts = withContext(Dispatchers.IO) {
                        filtered.associateWith { videoDao.getVideoCountForPatient(it.id) }
                    }
                    if (currentSortOrder) patientVideoCounts.entries.sortedBy { it.value }.map { it.key }
                    else patientVideoCounts.entries.sortedByDescending { it.value }.map { it.key }
                }
                else -> filtered // Default or fallback
            }

            // Update UI
            adapter.submitList(filtered)
            tvPatientCount.text = "${filtered.size} patient${if (filtered.size != 1) "s" else ""}"

            if (filtered.isEmpty()) {
                rvPatients.visibility = View.GONE
                emptyState.visibility = View.VISIBLE
                findViewById<TextView>(R.id.tvEmptyMessage).text = 
                    if (query.isNotEmpty()) "No patients match \"$query\"" 
                    else "No patients found"
            } else {
                rvPatients.visibility = View.VISIBLE
                emptyState.visibility = View.GONE
            }
        }
    }

    override fun onResume() {
        super.onResume()
        loadPatients()
    }
}

// Patient Adapter
class PatientAdapter(
    private val onPatientClick: (Patient) -> Unit,
    private val getVideoCount: (Long) -> Int
) : RecyclerView.Adapter<PatientAdapter.PatientViewHolder>() {

    private var patients: List<Patient> = emptyList()
    private val videoCounts = mutableMapOf<Long, Int>()

    fun submitList(newPatients: List<Patient>) {
        patients = newPatients
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PatientViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_patient_row, parent, false)
        return PatientViewHolder(view)
    }

    override fun onBindViewHolder(holder: PatientViewHolder, position: Int) {
        holder.bind(patients[position])
    }

    override fun getItemCount() = patients.size

    inner class PatientViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvPatientId: TextView = itemView.findViewById(R.id.tvPatientId)
        private val tvPatientFirstName: TextView = itemView.findViewById(R.id.tvPatientFirstName)
        private val tvPatientLastName: TextView = itemView.findViewById(R.id.tvPatientLastName)
        private val tvPatientAge: TextView = itemView.findViewById(R.id.tvPatientAge)
        private val tvVideoCount: TextView = itemView.findViewById(R.id.tvVideoCount)

   fun bind(patient: Patient) {
        tvPatientId.text = patient.participantId ?: "GV-${String.format("%04d", patient.id)}"
        tvPatientFirstName.text = patient.firstName
        tvPatientLastName.text = patient.lastName
        tvPatientAge.text = patient.age?.toString() ?: "—"

        val count = getVideoCount(patient.id)
        tvVideoCount.text = count.toString()

        itemView.setOnClickListener { onPatientClick(patient) }
}

    }
}

