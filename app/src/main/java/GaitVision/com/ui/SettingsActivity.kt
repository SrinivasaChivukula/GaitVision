package GaitVision.com.ui

import android.os.Bundle
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity
import GaitVision.com.R

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        setupBackButton()
    }

    private fun setupBackButton() {
        findViewById<ImageButton>(R.id.btnBack).setOnClickListener {
            finish()
        }
    }
}
