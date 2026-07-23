package GaitVision.com.ui

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import GaitVision.com.R

class MultiviewResultsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_multiview_results)

        val tvScore = findViewById<TextView>(R.id.tvMultiviewScore)
        val tvMessage = findViewById<TextView>(R.id.tvMultiviewMessage)

        val score = intent.getDoubleExtra(EXTRA_MULTIVIEW_SCORE, Double.NaN)

        if (score.isNaN()) {
            tvScore.text = "Score unavailable"
            tvMessage.text = "The multiview analysis result could not be loaded."
        } else {
            tvScore.text = String.format("%.1f / 100", score)
            tvMessage.text =
                "Preliminary multiview gait score based on side and back video analysis."
        }
    }

    companion object {
        const val EXTRA_MULTIVIEW_SCORE = "multiview_score"
    }
}