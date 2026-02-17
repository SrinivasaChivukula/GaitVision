package GaitVision.com.ui

import android.content.Intent
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.graphics.Color
import GaitVision.com.R
import GaitVision.com.data.PreferencesManager

class SettingsActivity : BaseActivity() {

    private lateinit var preferencesManager: PreferencesManager
    private lateinit var etUserName: EditText
    private lateinit var spinnerLanguage: Spinner
    private lateinit var spinnerTheme: Spinner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        preferencesManager = PreferencesManager(this)

        setupCommonHeader("Settings")
        NavigationHelper.setupBottomNav(this)

        initViews()
        setupSpinners()
        setupBatchExtract()
        loadSettings()
    }

    private fun setupBatchExtract() {
        val section = findViewById<android.view.View>(R.id.batchExtractSection)
        val btn = findViewById<Button>(R.id.btnBatchExtract)
        if (GaitVision.com.BuildConfig.DEBUG) {
            section.visibility = android.view.View.VISIBLE
            btn.setOnClickListener {
                startActivity(Intent(this, BatchExtractionActivity::class.java))
            }
        } else {
            section.visibility = android.view.View.GONE
        }
    }

    private fun initViews() {
        etUserName = findViewById(R.id.etUserName)
        spinnerLanguage = findViewById(R.id.spinnerLanguage)
        spinnerTheme = findViewById(R.id.spinnerTheme)
    }

    private fun setupSpinners() {
        // White text adapter
        fun createWhiteAdapter(items: Array<String>): ArrayAdapter<String> {
            return object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, items) {
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
        }

        // Language Spinner
        val languages = arrayOf("English", "Espanol")
        val languageAdapter = createWhiteAdapter(languages)
        spinnerLanguage.adapter = languageAdapter

        spinnerLanguage.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                preferencesManager.language = languages[position]
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        // Theme Spinner
        val themes = arrayOf("Dark", "Light")
        val themeAdapter = createWhiteAdapter(themes)
        spinnerTheme.adapter = themeAdapter

        spinnerTheme.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                preferencesManager.theme = themes[position]
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }
    }

    private fun loadSettings() {
        // Load Name
        etUserName.setText(preferencesManager.userName)

        // Load Language
        val savedLanguage = preferencesManager.language
        val languageAdapter = spinnerLanguage.adapter as ArrayAdapter<String>
        val languagePosition = languageAdapter.getPosition(savedLanguage)
        if (languagePosition >= 0) {
            spinnerLanguage.setSelection(languagePosition)
        }

        // Load Theme
        val savedTheme = preferencesManager.theme
        val themeAdapter = spinnerTheme.adapter as ArrayAdapter<String>
        val themePosition = themeAdapter.getPosition(savedTheme)
        if (themePosition >= 0) {
            spinnerTheme.setSelection(themePosition)
        }
    }

    override fun onPause() {
        super.onPause()
        // Save Name on pause to ensure it's remembered
        preferencesManager.userName = etUserName.text.toString()
    }
}
