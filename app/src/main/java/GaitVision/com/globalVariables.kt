package GaitVision.com

import android.graphics.Bitmap
import android.net.Uri
import GaitVision.com.gait.GaitFeatures
import GaitVision.com.gait.GaitDiagnostics
import GaitVision.com.gait.ScoringResult
import GaitVision.com.gait.Signals
import GaitVision.com.gait.Stride
import GaitVision.com.mediapipe.PoseFrame

//Any URI or frames used in application
var galleryUri : Uri? = null
var editedUri : Uri? = null
var frameList : MutableList<Bitmap> = mutableListOf()

// Pose frames for feature extraction (PC pipeline compatibility)
var poseFrames: MutableList<PoseFrame> = mutableListOf()

// Extracted gait features (16 features matching PC pipeline)
var extractedFeatures: GaitFeatures? = null
var extractionDiagnostics: GaitDiagnostics? = null
var scoringResult: ScoringResult? = null

// Signals for visualization (populated during feature extraction)
var extractedSignals: Signals? = null
var extractedStrides: List<Stride>? = null
var selectedStrideIndices: List<Int>? = null  // Indices of the 2 strides used for features
var stepSignalMode: String? = null

//All angles list for use in application
var leftAnkleAngles : MutableList<Float> = mutableListOf()
var rightAnkleAngles : MutableList<Float> = mutableListOf()
var leftKneeAngles : MutableList<Float> = mutableListOf()
var rightKneeAngles : MutableList<Float> = mutableListOf()
var leftHipAngles : MutableList<Float> = mutableListOf()
var rightHipAngles : MutableList<Float> = mutableListOf()
var torsoAngles : MutableList<Float> = mutableListOf()
var strideAngles: MutableList<Float> = mutableListOf()

//User input for ID and height
var participantId: Int = 0
var participantHeight: Int = 0

//Database IDs for current session
var currentPatientId: Int? = null
var currentVideoId: Long? = null

//Variable for counting angle faults
var count: Int = 0;


//Variable for keeping track of video length we processed on
var videoLength: Long = 0

// Video processing options (mirrors PC pipeline options)
var enableCLAHE: Boolean = false  // CLAHE disabled - testing without for parity comparison
var enableROIRetry: Boolean = true  // Retry with ROI tracking if first pass fails
var forceCpuInference: Boolean = true  // Force CPU inference for parity with PC (GPU can produce slight differences)