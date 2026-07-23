package GaitVision.com.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.PatientDao
import GaitVision.com.data.PreferencesManager
import GaitVision.com.ui.adapter.RecentPatientsAdapter
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class DashboardActivity : AppCompatActivity() {

    private val REQUEST_CODE_PERMISSIONS = 101
    private lateinit var patientDao: PatientDao
    private lateinit var adapter: RecentPatientsAdapter
    private lateinit var rvRecentPatients: RecyclerView
    private lateinit var tvEmptyRecent: TextView
    private lateinit var tvGreeting: TextView
    private lateinit var preferencesManager: PreferencesManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dashboard)

        val database = AppDatabase.getDatabase(this)
        patientDao = database.patientDao()
        preferencesManager = PreferencesManager(this)

        initViews()
        checkPermissions()
        setupButtons()
        setupRecentPatients()
    }

    override fun onResume() {
        super.onResume()
        GaitVision.com.AnalysisSession.reset()
        updateGreeting()
    }

    private fun updateGreeting() {
        val calendar = java.util.Calendar.getInstance()
        val hour = calendar.get(java.util.Calendar.HOUR_OF_DAY)

        val greeting = when (hour) {
            in 0..11 -> "Good morning"
            in 12..16 -> "Good afternoon"
            else -> "Good evening"
        }

        val name = preferencesManager.userName
        val finalGreeting = if (name.isNotEmpty()) {
            "$greeting, $name!"
        } else {
            "$greeting!"
        }

        tvGreeting.text = finalGreeting
    }

    private fun initViews() {
        rvRecentPatients = findViewById(R.id.rvRecentPatients)
        tvEmptyRecent = findViewById(R.id.tvEmptyRecent)
        tvGreeting = findViewById(R.id.tvGreeting)
    }

    private fun setupRecentPatients() {
        adapter = RecentPatientsAdapter { patient ->
            patient.participantId?.let { patientId ->
                val intent = Intent(this, PatientProfileActivity::class.java)
                intent.putExtra("patientId", patientId.toLong())
                startActivity(intent)
            }
        }

        rvRecentPatients.layoutManager = LinearLayoutManager(this)
        rvRecentPatients.adapter = adapter

        lifecycleScope.launch {
            patientDao.getRecentPatients(5).collectLatest { patients ->
                 if (patients.isEmpty()) {
                     rvRecentPatients.visibility = View.GONE
                     tvEmptyRecent.visibility = View.VISIBLE
                 } else {
                     rvRecentPatients.visibility = View.VISIBLE
                     tvEmptyRecent.visibility = View.GONE
                     adapter.submitList(patients)
                 }
            }
        }
    }

    private fun setupButtons() {
        // Patient Management Cards
        findViewById<View>(R.id.cardSearchPatient).setOnClickListener {
            startActivity(Intent(this, PatientListActivity::class.java))
        }

        findViewById<View>(R.id.cardNewPatient).setOnClickListener {
            startActivity(Intent(this, PatientCreateActivity::class.java))
        }

        // Quick Analysis (Main Content)
        findViewById<View>(R.id.cardQuickAnalysis).setOnClickListener {
            startActivity(Intent(this, QuickAnalysisActivity::class.java))
        }

        findViewById<View>(R.id.cardMultiviewAnalysis).setOnClickListener {
            startActivity(Intent(this, MultiviewAnalysisActivity::class.java))
        }

        // Bottom Navigation
        NavigationHelper.setupBottomNav(this)
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.READ_MEDIA_IMAGES,
            Manifest.permission.READ_MEDIA_VIDEO,
            Manifest.permission.READ_MEDIA_AUDIO,
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )

        if (!hasPermissions(*permissions)) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun hasPermissions(vararg permissions: String): Boolean {
        for (permission in permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false
            }
        }
        return true
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.isEmpty() || !grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Toast.makeText(
                    this,
                    "Permissions are required to access media files and camera.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}
