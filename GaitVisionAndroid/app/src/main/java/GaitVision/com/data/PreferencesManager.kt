package GaitVision.com.data

import android.content.Context
import android.content.SharedPreferences

class PreferencesManager(context: Context) {
    private val sharedPreferences: SharedPreferences = context.getSharedPreferences("GaitVisionPrefs", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_LANGUAGE = "language"
        private const val KEY_THEME = "theme"
        private const val KEY_USER_NAME = "user_name"
        private const val KEY_POSITION_SMOOTHING = "position_smoothing"
    }

    var language: String
        get() = sharedPreferences.getString(KEY_LANGUAGE, "English")!!
        set(value) = sharedPreferences.edit().putString(KEY_LANGUAGE, value).apply()

    var theme: String
        get() = sharedPreferences.getString(KEY_THEME, "Dark")!!
        set(value) = sharedPreferences.edit().putString(KEY_THEME, value).apply()

    var userName: String
        get() = sharedPreferences.getString(KEY_USER_NAME, "")!!
        set(value) = sharedPreferences.edit().putString(KEY_USER_NAME, value).apply()

    var positionSmoothingEnabled: Boolean
        get() = sharedPreferences.getBoolean(KEY_POSITION_SMOOTHING, false)
        set(value) = sharedPreferences.edit().putBoolean(KEY_POSITION_SMOOTHING, value).apply()
}
