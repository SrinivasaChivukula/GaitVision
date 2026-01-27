package GaitVision.com.ui

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import GaitVision.com.R
import GaitVision.com.data.AppDatabase
import GaitVision.com.data.Patient
import GaitVision.com.data.PatientDao
import GaitVision.com.participantId
import GaitVision.com.participantHeight
import GaitVision.com.currentPatientId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PatientCreateActivity : AppCompatActivity() {

    private lateinit var patientDao: PatientDao
    
    private lateinit var tvPatientId: TextView
    private lateinit var etFirstName: EditText
    private lateinit var etLastName: EditText
    private lateinit var etAge: EditText
    private lateinit var spinnerGender: Spinner
    private lateinit var etFeet: EditText
    private lateinit var etInches: EditText
    private lateinit var etNotes: EditText
    private lateinit var btnCreatePatient: Button
    private lateinit var btnCreateAndAnalyze: Button
    private lateinit var tvTitle: TextView

    private var editingPatientId: Int = -1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_patient_create)

        val database = AppDatabase.getDatabase(this)
        patientDao = database.patientDao()

        editingPatientId = intent.getLongExtra("patientId", -1).toInt()

        initViews()
        setupSpinners()
        
        if (editingPatientId > 0) {
            loadPatientForEditing()
        }
    }

    private fun initViews() {
        tvPatientId = findViewById(R.id.tvPatientId)
        tvTitle = findViewById(R.id.tvTitle)
        etFirstName = findViewById(R.id.etFirstName)
        etLastName = findViewById(R.id.etLastName)
        etAge = findViewById(R.id.etAge)
        spinnerGender = findViewById(R.id.spinnerGender)
        etFeet = findViewById(R.id.etFeet)
        etInches = findViewById(R.id.etInches)
        etNotes = findViewById(R.id.etNotes)
        btnCreatePatient = findViewById(R.id.btnCreatePatient)
        btnCreateAndAnalyze = findViewById(R.id.btnCreateAndAnalyze)

        findViewById<ImageButton>(R.id.btnBack).setOnClickListener {
            finish()
        }

        btnCreatePatient.setOnClickListener {
            if (validateInputs()) {
                savePatient(startAnalysis = false)
            }
        }

        btnCreateAndAnalyze.setOnClickListener {
            if (validateInputs()) {
                savePatient(startAnalysis = true)
            }
        }

        if (editingPatientId > 0) {
            tvTitle.text = "Edit Patient"
            btnCreatePatient.text = "Save Changes"
            btnCreateAndAnalyze.text = "Save & Start Analysis →"
        } else {
            // Load next patient ID for new patient
            loadNextPatientId()
        }
    }

    private fun setupSpinners() {
        // Gender spinner with custom adapter for proper text colors
        val genderArray = resources.getStringArray(R.array.gender_array)
        val genderAdapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, genderArray) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                val view = super.getDropDownView(position, convertView, parent)
                (view as TextView).setTextColor(Color.WHITE)
                return view
            }
        }.apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinnerGender.adapter = genderAdapter

        // Set default height values
        etFeet.setText("5")
        etInches.setText("9")
    }

    private fun loadPatientForEditing() {
        lifecycleScope.launch {
            val patient = withContext(Dispatchers.IO) {
                patientDao.getPatientById(editingPatientId)
            }
            
            patient?.let {
                tvPatientId.text = it.participantId?.toString() ?: "N/A"
                etFirstName.setText(it.firstName)
                etLastName.setText(it.lastName)
                etAge.setText(it.age?.toString() ?: "")
                
                // Set gender spinner
                val genderArray = resources.getStringArray(R.array.gender_array)
                val genderIndex = genderArray.indexOf(it.gender ?: "")
                if (genderIndex >= 0) spinnerGender.setSelection(genderIndex)
                
                // Set height EditTexts
                val feet = it.height / 12
                val inches = it.height % 12
                etFeet.setText(feet.toString())
                etInches.setText(inches.toString())
            }
        }
    }

    private fun loadNextPatientId() {
        lifecycleScope.launch {
            val nextId = withContext(Dispatchers.IO) {
                patientDao.getNextPatientId()
            }
            tvPatientId.text = nextId.toString()
        }
    }

    private fun validateInputs(): Boolean {
        val firstName = etFirstName.text.toString().trim()
        val lastName = etLastName.text.toString().trim()

        if (firstName.isEmpty()) {
            etFirstName.error = "First name is required"
            etFirstName.requestFocus()
            return false
        }

        if (lastName.isEmpty()) {
            etLastName.error = "Last name is required"
            etLastName.requestFocus()
            return false
        }

        // Validate feet and inches
        for ((editText, name, max) in listOf(
            Triple(etFeet, "Feet", 8),
            Triple(etInches, "Inches", 11)
        )) {
            val str = editText.text.toString().trim()
            if (str.isEmpty()) {
                editText.error = "$name is required"
                editText.requestFocus()
                return false
            }

            val value = str.toIntOrNull()
            if (value == null || value < 0 || value > max) {
                editText.error = "$name must be between 0 and $max"
                editText.requestFocus()
                return false
            }
        }

        return true
    }

    private fun savePatient(startAnalysis: Boolean) {
        val firstName = etFirstName.text.toString().trim()
        val lastName = etLastName.text.toString().trim()
        val age = etAge.text.toString().toIntOrNull()
        val gender = if (spinnerGender.selectedItemPosition > 0) {
            spinnerGender.selectedItem.toString()
        } else null
        
        val feet = etFeet.text.toString().trim().toInt()
        val inches = etInches.text.toString().trim().toInt()
        val heightInInches = (feet * 12) + inches

        lifecycleScope.launch {
            val patient = if (editingPatientId > 0) {
                // Update existing patient
                val existing = withContext(Dispatchers.IO) {
                    patientDao.getPatientById(editingPatientId)
                }
                existing?.copy(
                    firstName = firstName,
                    lastName = lastName,
                    age = age,
                    gender = gender,
                    height = heightInInches,
                    participantId = editingPatientId // Ensure participantId is passed for update
                ) ?: return@launch
            } else {
                // Create new patient
                Patient(
                    firstName = firstName,
                    lastName = lastName,
                    age = age,
                    gender = gender,
                    height = heightInInches
                )
            }

            val patientId = withContext(Dispatchers.IO) {
                val newId = patientDao.insertPatient(patient)
                // For a new patient, the newId is the generated participantId
                // For an update, patient.participantId is already set
                newId
            }

            withContext(Dispatchers.Main) {
                Toast.makeText(
                    this@PatientCreateActivity,
                    if (editingPatientId > 0) "Patient updated" else "Patient created",
                    Toast.LENGTH_SHORT
                ).show()

                if (startAnalysis) {
                    // Set global variables for analysis flow
                    participantId = patientId.toInt()
                    participantHeight = heightInInches
                    currentPatientId = patientId.toInt()

                    // Go to video picker
                    val intent = Intent(this@PatientCreateActivity, VideoPickerActivity::class.java)
                    intent.putExtra("patientId", patientId)
                    intent.putExtra("fromPatientProfile", true)
                    startActivity(intent)
                }
                
                finish()
            }
        }
    }
}

