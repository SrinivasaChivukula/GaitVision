package GaitVision.com

import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.sqrt


/*
Name          : calcTorso
Parameters    :

    hipX      : x coordinate of the hip average on the bitmap.
    hipY      : y coordinate of the hip average on the bitmap.
    shoulderX : x coordinate of the shoulder on the bitmap.
    shoulderY : y coordinate of the shoulder on the bitmap.
Description   : This function takes the input of the position of the hip average position and the
                shoulder average position and gets the angle between then and the vertical.
                0 degrees means hips and shoulder are vertical with each other.
                Positive degrees means the person is leaning back from the vertical.
                Negative degrees means the person is leaning forward from the vertical.
Return        :
      Float   : Returns the angle in degrees between the hips and shoulder from the vertical.
 */
fun calcTorso(hipX: Float, hipY: Float, shoulderX: Float, shoulderY: Float): Float
{
    // Calculate the differences in X and Y coordinates
    val deltaX = hipX - shoulderX
    val deltaY = hipY - shoulderY

    // Calculate the torso angle relative to the vertical axis
    val angle = atan2(deltaX.toDouble(), deltaY.toDouble()) * (180 / PI) // Convert to degrees

    return angle.toFloat()
}

/*
Name           : calcStrideAngle
Parameters     :
    leftHeelX  : x coordinate of the left heel on the bitmap.
    leftHeelY  : y coordinate of the left heel on the bitmap.
    hipX       : x coordinate of the hip average on the bitmap.
    hipY       : y coordinate of the hip average on the bitmap.
    rightHeelX : x coordinate of the right heel on the bitmap.
    rightHeelY : y coordinate of the right heel on the bitmap.
Description    : This function will take the position of the left heel, hip, and right heel
                 and calculate the angle between the heels and hip at the current time and
                 add it it a running list to use for calculating average stride length and
                 total stride distance.
Return         :
    Float      : Returns the angle in degrees between the heels and the hip.

 */
fun calcStrideAngle(leftHeelX: Float, leftHeelY: Float, hipx: Float, hipY: Float, rightHeelX: Float, rightHeelY: Float) : Float
{
    var htol = sqrt(((leftHeelX - hipx)*(leftHeelX - hipx))+((leftHeelY-hipY)*(leftHeelY-hipY)))
    var htor = sqrt(((rightHeelX - hipx)*(rightHeelX - hipx))+((rightHeelY-hipY)*(rightHeelY-hipY)))
    var ltor = sqrt(((leftHeelX - rightHeelX)*(leftHeelX - rightHeelX))+((leftHeelY-rightHeelY)*(leftHeelY-rightHeelY)))

    var cosAngle : Float = ((htol * htol) + (htor * htor) - (ltor * ltor))/(2*htol*htor)
    var angle = acos(cosAngle) * (180/PI.toFloat())
    return String.format("%.2f", angle).toFloat()
}

/*
Name           : smoothDataUsingMovingAverage
Parameters     :
    data       : This is the list of angles values to use for smoothing.
    windowSize : This is the size of values we want to use.
Description    : This function will take a list of values and a window size and calculate an
                 average of those values to use for graph smoothing. There is some data lost with
                 graph smoothing, but allows for better viewing and calculations.
Return         :
    NONE
 */
fun smoothDataUsingMovingAverage(data: MutableList<Float>, windowSize: Int)
{
    val smoothedData = mutableListOf<Float>()

    for (i in data.indices) {
        // Calculate the window range
        val windowStart = maxOf(i - windowSize / 2, 0)
        val windowEnd = minOf(i + windowSize / 2, data.size - 1)

        // Calculate the average for the current window
        val window = data.subList(windowStart, windowEnd + 1)
        val avg = window.average().toFloat()

        smoothedData.add(avg)
    }

    // Update the original list with the smoothed values
    data.clear()  // Clear the original list
    data.addAll(smoothedData)  // Add the smoothed data back
}

/*
Name        : GetAnglesA
Parameters  :
    x1      : x coordinate of the ball of the respective side.
    y1      : y coordinate of the ball of the respective side.
    x2      : x coordinate of the ankle of the respective side.
    y2      : y coordinate of the ankle of the respective side.
    x3      : x coordinate of the knee of the respective side.
    y3      : y coordinate of the knee of the respective side.
Description : This functions takes the position of the ball of the foot, the ankle, and the knee
              and calculates the ankle angle using the law of cosine.
              0 degrees means the foot is flat on the ground making a 90 degree angle from the foot
              to ankle to knee.
              Negative degrees means the foot is flexed up toward the knee.
              Positive degrees means the foot is flexed down toward the ground.
Return      :
    Float   : Returns the angle in degrees between the ball of the foot, the ankle, and the knee.
 */
fun GetAnglesA(x1: Float,y1: Float, x2: Float, y2: Float, x3: Float, y3: Float): Float
{
    // Get Distances
    val Length1: Float = CalculateDistance(x1,y1,x2,y2)
    val Length2: Float = CalculateDistance(x2,y2,x3,y3)
    val Length3: Float = CalculateDistance(x1,y1,x3,y3)
    //use law of COS to help find angle
    val CosAngle: Float = (Length1 * Length1 + Length2 * Length2 - Length3 * Length3) / (2 * Length1 * Length2)
    //find Angle then convert from radian to degree
    val Angle: Float = acos(CosAngle) * (180/PI.toFloat())
    //Might change later
    val smallerAngle: Float = Angle - 90f
    return String.format("%.2f", smallerAngle).toFloat() //round to 2 decimal places
}

