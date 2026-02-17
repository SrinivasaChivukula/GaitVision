package GaitVision.com

import android.net.Uri
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.Signals
import GaitVision.com.gait.Stride
import GaitVision.com.mediapipe.PoseFrame

/** Session state for analysis flow. */
object AnalysisSession {
    var galleryUri: Uri? = null
    var editedUri: Uri? = null
    var poseFrames: MutableList<PoseFrame> = mutableListOf()
    var extractedFeatures: GaitFeatures? = null
    var extractionDiagnostics: GaitDiagnostics? = null
    var scoringResult: ScoringResult? = null
    var extractedSignals: Signals? = null
    var extractedStrides: List<Stride>? = null
    var selectedStrideIndices: List<Int>? = null
    var stepSignalMode: String? = null
    var participantId: Int = 0
    var participantHeight: Int = 0
    var currentPatientId: Int? = null
    var currentResultId: Long? = null
    var videoLength: Long = 0

    fun clear() {
        galleryUri = null
        editedUri = null
        poseFrames.clear()
        extractedFeatures = null
        extractionDiagnostics = null
        scoringResult = null
        extractedSignals = null
        extractedStrides = null
        selectedStrideIndices = null
        stepSignalMode = null
        participantId = 0
        participantHeight = 0
        currentPatientId = null
        currentResultId = null
        videoLength = 0
    }
}
