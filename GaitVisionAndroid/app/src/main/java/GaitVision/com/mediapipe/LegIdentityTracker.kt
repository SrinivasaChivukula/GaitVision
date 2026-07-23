package GaitVision.com.mediapipe

import kotlin.math.sqrt

/**
 * Tracks and corrects leg-label identity across frames.
 *
 * MediaPipe occasionally swaps left/right leg labels mid-video (especially
 * during crossovers or brief occlusions). This tracker detects those swaps by
 * comparing labeling costs against the previous frame, votes to confirm a
 * genuine swap, and rewrites keypoints/confidences accordingly.
 *
 * Mirrors PC `LegIdentityTracker` for feature parity.
 *
 * Labeling convention: landmarks are MediaPipe screen-relative (not
 * anatomical). For a frontal subject this means MediaPipe's LEFT_HIP can be
 * the person's right hip. The tracker preserves whatever convention the
 * upstream MediaPipe call uses — downstream stride/asymmetry metrics must
 * agree on this convention.
 *
 * Thread safety: this class holds mutable state (the last-seen PoseFrame and
 * the swap-vote counter) without synchronization. It is designed to be called
 * serially from a single producer. Concurrent callers (e.g. parallel preview
 * inference) must wrap `correct()` in their own mutex.
 *
 * Buffer-reuse caveat: MediaPipe producers commonly reuse the inner
 * `keypoints` / `confidences` arrays of a `PoseFrame` across calls. To remain
 * robust against that pattern, every `PoseFrame` stored in [previous] is a
 * defensive deep copy (see [cloneFrame]) — callers may freely hand us any
 * PoseFrame without worrying about downstream mutation of its arrays.
 */
class LegIdentityTracker(
    private val confidenceThreshold: Float = 0.25f,
    private val swapCostRatio: Float = 0.75f,        // swap wins only if swap < keep * swapCostRatio
    private val swapVotesRequired: Int = 3,
    private val jumpDistanceThreshold: Float = 0.15f // normalized (0..1) movement
) {
    private class LegIndices(val left: IntArray, val right: IntArray) {
        init {
            // Guard against a MediaPipe model variant that drops, e.g.,
            // LEFT_HEEL / LEFT_FOOT_INDEX — without this, the fixed-bound
            // iteration in `labelingCost` / `swapLegs` would silently throw
            // ArrayIndexOutOfBoundsException deep inside a per-frame loop.
            require(left.size == right.size) {
                "left/right index arrays must be the same length (got ${left.size} vs ${right.size})"
            }
            require(left.size >= 2) {
                "at least one leg-landmark pair required (got ${left.size})"
            }
            // Ensure every declared index actually fits inside a 33-landmark
            // MediaPipe pose buffer.
            val all = left + right
            require(all.all { it in 0 until 33 }) {
                "leg index out of MediaPipe 33-landmark range: $all"
            }
        }
    }

    private val indices = LegIndices(
        intArrayOf(
            MediaPipePoseBackend.LEFT_HIP,
            MediaPipePoseBackend.LEFT_KNEE,
            MediaPipePoseBackend.LEFT_ANKLE,
            MediaPipePoseBackend.LEFT_HEEL,
            MediaPipePoseBackend.LEFT_FOOT_INDEX,
        ),
        intArrayOf(
            MediaPipePoseBackend.RIGHT_HIP,
            MediaPipePoseBackend.RIGHT_KNEE,
            MediaPipePoseBackend.RIGHT_ANKLE,
            MediaPipePoseBackend.RIGHT_HEEL,
            MediaPipePoseBackend.RIGHT_FOOT_INDEX,
        ),
    )

    // Precompute once instead of allocating per rejectBigJumps call.
    private val allLegIndices: IntArray = indices.left + indices.right

    private var previous: PoseFrame? = null
    private var swapVotes = 0
    private var labelsSwapped = false

    // Number of frames the current frame is ahead of `previous`. Normally 1,
    // but grows while swap votes accumulate because `previous` is
    // intentionally frozen on the last accepted identity frame during that
    // window. rejectBigJumps scales its per-frame threshold by this so
    // legitimate fast motion spanning the vote window is not zeroed against
    // a stale reference.
    private var prevAgeFrames = 1

    /**
     * True when the most recent [correct] call changed the L/R mapping.
     * Downstream temporal filters (e.g. One-Euro smoothing) should treat that
     * frame as a discontinuity for leg landmarks and reset their state, since
     * each leg slot's trajectory jumps to the mirror side at the flip.
     */
    var mappingFlippedOnLastFrame = false
        private set

    /** Leg landmark slots managed by this tracker (both sides). */
    val legLandmarkIndices: IntArray
        get() = allLegIndices.copyOf()

    /**
     * Clear tracker state (call at start of a new processing session).
     */
    fun reset() {
        previous = null
        swapVotes = 0
        labelsSwapped = false
        mappingFlippedOnLastFrame = false
        prevAgeFrames = 1
    }

    /**
     * Decide whether to swap leg labels for [frame] and reject implausibly large
     * single-frame jumps, then remember the result for the next call.
     *
     * NaN handling: [labelingCost] divides by `count`, so a count of 0
     * returns +Infinity. NaN values entering [euclidean] are not currently
     * screened. Because `NaN < anything` is `false`, a NaN-bearing frame
     * would fail the `swap < keep * swapCostRatio` test and fall through to
     * the "no evidence to swap" branch (swapVotes reset). That coincidentally
     * matches the all-occluded branch's behavior — we explicitly accept this
     * collision rather than special-casing NaN.
     */
    fun correct(frame: PoseFrame): PoseFrame {
        mappingFlippedOnLastFrame = false
        val prev = previous
        if (prev == null) {
            // Defensive deep copy: MediaPipe producers often reuse inner arrays
            // of PoseFrame across calls. Without cloning, our stored `previous`
            // would alias the caller's mutable buffer and later-frame cost
            // calculations would silently see in-place mutations.
            previous = cloneFrame(frame)
            return frame
        }

        val keep = labelingCost(frame, prev, hypothesizeSwap = labelsSwapped)
        val alternate = labelingCost(frame, prev, hypothesizeSwap = !labelsSwapped)
        val bothInfinite = keep.isInfinite() && alternate.isInfinite()

        if (bothInfinite) {
            // No evidence either way: pass through unchanged. Also reset
            // swapVotes so stale votes accumulated before the blackout don't
            // prematurely trigger a swap once we re-acquire detection.
            //
            // Note: rejectBigJumps is then evaluated against prev (which may
            // be the pre-blackout previous with high-conf leg joints). If the
            // blackout was bidirectional, that snap-back will zero every
            // leg slot's confidence — that's the known cost documented in
            // the class header, accepted here for simplicity.
            swapVotes = 0
            val cleaned = rejectBigJumps(applyCurrentMapping(frame), prev)
            previous = cloneFrame(cleaned)
            prevAgeFrames = 1
            return cleaned
        }

        if (
            // Guard: keep must be finite. If keep is infinite (all-occluded
            // hypothesis) any finite alternate trivially satisfies alternate < inf,
            // which inverts the intended comparison. Treat that case as
            // "no evidence to change the current mapping".
            keep.isFinite() && alternate.isFinite() && alternate < keep * swapCostRatio
        ) {
            swapVotes++
            if (swapVotes >= swapVotesRequired) {
                labelsSwapped = !labelsSwapped
                mappingFlippedOnLastFrame = true
                val remapped = applyCurrentMapping(frame)
                // Vote threshold reached and we just changed mapping — reset vote
                // counter so the next genuine mapping change can accumulate again.
                swapVotes = 0
                // One-frame grace window: skip rejectBigJumps on the frame
                // where a mapping change just landed. rejectBigJumps uses
                // idx-aligned prev comparison, so the remapped leg joints may
                // look like the intentional teleport we are correcting.
                // Rejecting here would zero the newly corrected data. We still
                // update previous to the remapped frame so subsequent frames
                // compute costs against the corrected reference.
                previous = cloneFrame(remapped)
                prevAgeFrames = 1
                return remapped
            } else {
                val cleaned = rejectBigJumps(applyCurrentMapping(frame), prev)
                // Keep comparing pending mapping-change candidates against the
                // last accepted identity frame. If we saved this unconfirmed
                // frame as previous, a persistent MediaPipe L/R swap would look
                // "consistent" on the very next frame and the vote counter
                // would reset before it could reach swapVotesRequired.
                // `previous` therefore ages by one frame per pending vote.
                prevAgeFrames++
                return cleaned
            }
        } else {
            swapVotes = 0
        }

        val cleaned = rejectBigJumps(applyCurrentMapping(frame), prev)
        previous = cloneFrame(cleaned)
        prevAgeFrames = 1
        return cleaned
    }

    private fun applyCurrentMapping(frame: PoseFrame): PoseFrame =
        if (labelsSwapped) swapLegs(frame) else frame

    /**
     * Average per-joint Euclidean distance between [cur] and [prev] across the
     * five leg pairs. If [hypothesizeSwap] is true, compare each side of [cur]
     * against the opposite side of [prev] to estimate the cost of swapping.
     *
     * Returns +Infinity if no joint pair has usable confidence in both frames.
     */
    private fun labelingCost(cur: PoseFrame, prev: PoseFrame, hypothesizeSwap: Boolean): Float {
        var sum = 0f
        var count = 0
        for (k in indices.left.indices) {
            // Left side of cur vs (left or right) side of prev
            val curLeftIdx = indices.left[k]
            val prevLeftIdx = if (hypothesizeSwap) indices.right[k] else indices.left[k]
            if (cur.confidences[curLeftIdx] >= confidenceThreshold &&
                prev.confidences[prevLeftIdx] >= confidenceThreshold
            ) {
                sum += euclidean(cur.keypoints[curLeftIdx], prev.keypoints[prevLeftIdx])
                count++
            }

            // Right side of cur vs (left or right) side of prev
            val curRightIdx = indices.right[k]
            val prevRightIdx = if (hypothesizeSwap) indices.left[k] else indices.right[k]
            if (cur.confidences[curRightIdx] >= confidenceThreshold &&
                prev.confidences[prevRightIdx] >= confidenceThreshold
            ) {
                sum += euclidean(cur.keypoints[curRightIdx], prev.keypoints[prevRightIdx])
                count++
            }
        }
        return if (count == 0) Float.POSITIVE_INFINITY else sum / count
    }

    /**
     * Return a new [PoseFrame] with every leg keypoint (and its confidence)
     * swapped between left and right. Non-leg landmarks pass through unchanged.
     *
     * Note: every inner FloatArray for a leg slot is freshly `.copyOf()`-ed,
     * so the returned PoseFrame does not alias the input for those slots.
     * Non-leg slots are also `.copyOf()`-ed below for full isolation —
     * do NOT rely on data-class `.copy(keypoints, ...)` alone for isolation,
     * since that performs only a shallow field copy.
     *
     * Convention: indices.left and indices.right must be the same length and
     * each entry pairs the MediaPipe landmark indices for the same anatomical
     * joint on each side.
     */
    private fun swapLegs(frame: PoseFrame): PoseFrame {
        val newKps = Array(frame.keypoints.size) { i -> frame.keypoints[i].copyOf() }
        val newConf = frame.confidences.copyOf()
        // Swap leg slots in a single zipped pass so we don't double-scan via
        // membership test + indexOf for each entry.
        for (k in indices.left.indices) {
            val l = indices.left[k]
            val r = indices.right[k]
            newKps[l] = frame.keypoints[r].copyOf()
            newKps[r] = frame.keypoints[l].copyOf()
            newConf[l] = frame.confidences[r]
            newConf[r] = frame.confidences[l]
        }
        return frame.copy(keypoints = newKps, confidences = newConf)
    }

    /**
     * For leg joints that just appeared (low confidence last frame, high this
     * frame) but moved implausibly far from the previous frame, snap them back
     * to the previous keypoint and zero confidence so downstream code treats
     * them as undetected rather than trusting a teleport.
     *
     * The threshold is normalized (0..1) image-space distance — the tracker
     * assumes consecutive frames. If you replay with skipped frame indices,
     * scale this threshold by the number of skipped frames.
     *
     * Note on swap window: rejected frames under idx-aligned prev comparison,
     * so a leg joint that legitimately moved to the mirror side (idx-aligned
     * distance = the previous OTHER leg's position) would be snapped back to
     * its pre-swap position and zeroed. The swap-confirmed branch in
     * [correct] therefore skips [rejectBigJumps] for a single frame after a
     * successful swap, so the swapped-frame data is preserved.
     */
    private fun rejectBigJumps(frame: PoseFrame, prev: PoseFrame): PoseFrame {
        val newKps: Array<FloatArray> = Array(frame.keypoints.size) { i ->
            frame.keypoints[i].copyOf()
        }
        val newConf = frame.confidences.copyOf()

        // `previous` can lag several frames behind while swap votes accumulate;
        // scale the per-frame threshold so genuine fast motion over that span
        // is not mistaken for a teleport.
        val effectiveThreshold = jumpDistanceThreshold * prevAgeFrames

        for (idx in allLegIndices) {
            if (frame.confidences[idx] >= confidenceThreshold &&
                prev.confidences[idx] < confidenceThreshold &&
                euclidean(frame.keypoints[idx], prev.keypoints[idx]) > effectiveThreshold
            ) {
                newKps[idx] = prev.keypoints[idx].copyOf()
                newConf[idx] = 0f
            }
        }
        return frame.copy(keypoints = newKps, confidences = newConf)
    }

    private fun euclidean(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Deep copy of [frame]'s keypoint / confidence arrays. Used so the
     * tracker's stored `previous` is isolated from any buffer the caller may
     * subsequently mutate (a common MediaPipe producer pattern).
     */
    private fun cloneFrame(frame: PoseFrame): PoseFrame =
        frame.copy(
            keypoints = Array(frame.keypoints.size) { i -> frame.keypoints[i].copyOf() },
            confidences = frame.confidences.copyOf(),
        )

    /**
     * Short status for debugging — e.g. via Log.d or unit-test assertions.
     */
    override fun toString(): String =
        "LegIdentityTracker(swapVotes=$swapVotes, labelsSwapped=$labelsSwapped, hasPrevious=${previous != null})"
}
