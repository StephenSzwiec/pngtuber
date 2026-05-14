"""
pngtuber_skeleton.py
--------------------
Pose-driven PNGtuber prototype.

Architecture:
    Webcam -> MediaPipe Holistic -> Landmark dict
                                        |
                                        v
                 +------------------+--------+---------+
                 |                  |        |         |
                 v                  v        v         v
            PoseClassifier  ExprClassifier  Blink   VoiceActivity
                 \\                 |        /         /
                  \\                |       /         /
                   +----- StateTuple(pose, expr, blink, talk) ------+
                                        |
                                        v
                                  Debouncer
                                        |
                                        v
                                  SpriteRenderer
                                  (transparent window
                                   captured by OBS)

Run:
    python pngtuber_skeleton.py --sprites ./sprites
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# Audio is optional; voice activity can also come from your existing PNGtuber
# tool if you'd rather let it own the talk/quiet axis. See note at bottom.
try:
    import sounddevice as sd
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False


# ---------------------------------------------------------------------------
# State axes
# ---------------------------------------------------------------------------

class Pose(Enum):
    NEUTRAL = "neutral"
    HAND_ON_CHIN = "hand_on_chin"
    HANDS_OVER_HEAD = "hands_over_head"


class Expression(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"


@dataclass(frozen=True)
class FaceState:
    """The two boolean sub-axes that together make the 'face' axis of 4."""
    blinking: bool
    talking: bool

    @property
    def label(self) -> str:
        b = "blink" if self.blinking else "noblink"
        t = "yap" if self.talking else "quiet"
        return f"{b}_{t}"


@dataclass(frozen=True)
class AvatarState:
    pose: Pose
    expression: Expression
    face: FaceState

    @property
    def key(self) -> tuple[str, str, str]:
        """The (pose, expression, face) tuple used to look up a sprite."""
        return (self.pose.value, self.expression.value, self.face.label)


# ---------------------------------------------------------------------------
# Perception: pull a clean landmark dict out of MediaPipe each frame
# ---------------------------------------------------------------------------

@dataclass
class Landmarks:
    """
    Normalized landmark snapshot for one frame. All coords are in [0, 1]
    image space; z is approximate depth (smaller = closer to camera).
    None fields mean 'not detected this frame'.
    """
    pose: Optional[np.ndarray] = None   # (33, 4)  x, y, z, visibility
    face: Optional[np.ndarray] = None   # (478, 3) x, y, z
    left_hand: Optional[np.ndarray] = None   # (21, 3)
    right_hand: Optional[np.ndarray] = None  # (21, 3)


_MODEL_URLS: dict[str, str] = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def _ensure_model(name: str, model_dir: Path) -> Path:
    """Return the local path for *name*, downloading from Google Storage if absent."""
    import urllib.request
    path = model_dir / name
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} ...", end=" ", flush=True)
        urllib.request.urlretrieve(_MODEL_URLS[name], path)
        print("done")
    return path


class HolisticTracker:
    """
    Wraps the MediaPipe Tasks API (PoseLandmarker + FaceLandmarker +
    HandLandmarker). Replaces the removed mp.solutions.holistic.Holistic.
    Model files are downloaded automatically on first run.
    """

    _POSE_MODELS = [
        "pose_landmarker_lite.task",
        "pose_landmarker_full.task",
        "pose_landmarker_heavy.task",
    ]

    def __init__(self, model_complexity: int = 1,
                 model_dir: Path = Path("models")):
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv

        pose_name = self._POSE_MODELS[min(model_complexity, 2)]

        self._pose = _mpv.PoseLandmarker.create_from_options(
            _mpv.PoseLandmarkerOptions(
                base_options=_mpt.BaseOptions(
                    model_asset_path=str(_ensure_model(pose_name, model_dir))),
                running_mode=_mpv.RunningMode.IMAGE,
                num_poses=1,
            )
        )
        self._face = _mpv.FaceLandmarker.create_from_options(
            _mpv.FaceLandmarkerOptions(
                base_options=_mpt.BaseOptions(
                    model_asset_path=str(_ensure_model("face_landmarker.task", model_dir))),
                running_mode=_mpv.RunningMode.IMAGE,
                num_faces=1,
            )
        )
        self._hand = _mpv.HandLandmarker.create_from_options(
            _mpv.HandLandmarkerOptions(
                base_options=_mpt.BaseOptions(
                    model_asset_path=str(_ensure_model("hand_landmarker.task", model_dir))),
                running_mode=_mpv.RunningMode.IMAGE,
                num_hands=2,
            )
        )

    def process(self, bgr_frame: np.ndarray) -> Landmarks:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_res = self._pose.detect(mp_image)
        face_res = self._face.detect(mp_image)
        hand_res = self._hand.detect(mp_image)

        def pose_to_array(res) -> Optional[np.ndarray]:
            if not res.pose_landmarks:
                return None
            return np.array(
                [[l.x, l.y, l.z, getattr(l, "visibility", 1.0)]
                 for l in res.pose_landmarks[0]],
                dtype=np.float32,
            )

        def face_to_array(res) -> Optional[np.ndarray]:
            if not res.face_landmarks:
                return None
            return np.array(
                [[l.x, l.y, l.z] for l in res.face_landmarks[0]],
                dtype=np.float32,
            )

        def hand_to_array(res, side: str) -> Optional[np.ndarray]:
            if not res.hand_landmarks:
                return None
            for i, handedness in enumerate(res.handedness):
                if handedness[0].category_name == side:
                    return np.array(
                        [[l.x, l.y, l.z] for l in res.hand_landmarks[i]],
                        dtype=np.float32,
                    )
            return None

        return Landmarks(
            pose=pose_to_array(pose_res),
            face=face_to_array(face_res),
            left_hand=hand_to_array(hand_res, "Left"),
            right_hand=hand_to_array(hand_res, "Right"),
        )

    def close(self) -> None:
        self._pose.close()
        self._face.close()
        self._hand.close()


# MediaPipe Pose landmark indices we'll reference (BlazePose 33-point topology).
# See: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
class PL:
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


# ---------------------------------------------------------------------------
# Classifiers — rule-based to start; each one is a pure function of Landmarks.
# Swap any of these for a small MLP later without touching the rest.
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """2D Euclidean distance in normalized image space."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def _shoulder_width(pose: np.ndarray) -> float:
    """Used as a scale reference so thresholds aren't pixel-dependent."""
    return _dist(pose[PL.LEFT_SHOULDER], pose[PL.RIGHT_SHOULDER]) or 1e-6


class PoseClassifier:
    """
    Returns one of {NEUTRAL, HAND_ON_CHIN, HANDS_OVER_HEAD}.

    All thresholds are expressed as multiples of shoulder width so they're
    invariant to how close she sits to the camera.
    """

    # Tune these by running the script with --debug and watching the printed
    # feature values for each pose.
    CHIN_RADIUS = 0.6        # wrist within 0.6 * shoulder_width of mouth
    OVERHEAD_MARGIN = 0.3    # wrist y must be this far above shoulder y
                              #   (recall y grows downward, so "above" = smaller y)

    def classify(self, lm: Landmarks) -> Pose:
        if lm.pose is None:
            return Pose.NEUTRAL

        p = lm.pose
        sw = _shoulder_width(p)

        # Chin proxy: midpoint of mouth landmarks
        chin = (p[PL.MOUTH_LEFT] + p[PL.MOUTH_RIGHT]) * 0.5
        shoulder_y = (p[PL.LEFT_SHOULDER, 1] + p[PL.RIGHT_SHOULDER, 1]) * 0.5

        lw, rw = p[PL.LEFT_WRIST], p[PL.RIGHT_WRIST]

        # HANDS_OVER_HEAD wins if both wrists are clearly above shoulder line.
        if (shoulder_y - lw[1] > self.OVERHEAD_MARGIN and
                shoulder_y - rw[1] > self.OVERHEAD_MARGIN):
            return Pose.HANDS_OVER_HEAD

        # HAND_ON_CHIN if either wrist is close to mouth center.
        if (_dist(lw, chin) / sw < self.CHIN_RADIUS or
                _dist(rw, chin) / sw < self.CHIN_RADIUS):
            return Pose.HAND_ON_CHIN

        return Pose.NEUTRAL


class ExpressionClassifier:
    """
    Returns one of {NEUTRAL, HAPPY, ANGRY}.

    Facial expression from raw landmarks is the brittlest of the three axes.
    This is a placeholder using two coarse geometric features:
        - mouth_curvature: corners up vs down vs flat
        - brow_lowering:   inner brow distance to eye
    If you find this insufficient, this is the natural place to plug in a
    small MLP trained on flattened face landmarks (Kazuhito00-style).
    """

    # MediaPipe FaceMesh indices (478-point with refine_face_landmarks=True)
    MOUTH_LEFT_CORNER = 61
    MOUTH_RIGHT_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14
    LEFT_BROW_INNER = 55
    RIGHT_BROW_INNER = 285
    LEFT_EYE_TOP = 159
    RIGHT_EYE_TOP = 386

    SMILE_THRESHOLD = 0.0   # tune: corners above mouth-center y => smiling
    FROWN_THRESHOLD = 0.01

    def classify(self, lm: Landmarks) -> Expression:
        if lm.face is None:
            return Expression.NEUTRAL

        f = lm.face
        mouth_center_y = (f[self.UPPER_LIP_CENTER, 1]
                          + f[self.LOWER_LIP_CENTER, 1]) * 0.5
        corner_y = (f[self.MOUTH_LEFT_CORNER, 1]
                    + f[self.MOUTH_RIGHT_CORNER, 1]) * 0.5

        # Negative = corners higher on screen than mouth center => smile.
        curvature = corner_y - mouth_center_y

        # Brow lowering: smaller distance = brows pulled down toward eyes.
        brow_eye = (
            abs(f[self.LEFT_BROW_INNER, 1] - f[self.LEFT_EYE_TOP, 1])
            + abs(f[self.RIGHT_BROW_INNER, 1] - f[self.RIGHT_EYE_TOP, 1])
        ) * 0.5

        if curvature < self.SMILE_THRESHOLD:
            return Expression.HAPPY
        if brow_eye < 0.015 and curvature > self.FROWN_THRESHOLD:
            return Expression.ANGRY
        return Expression.NEUTRAL


class BlinkDetector:
    """
    Returns True if currently blinking, using eye-aspect-ratio (EAR).
    Standard trick from Soukupová & Čech 2016; works fine on MediaPipe
    FaceMesh with refine_face_landmarks=True.
    """

    # Six points around each eye (top, bottom, left, right, plus two for height)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    EAR_THRESHOLD = 0.20

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        # eye_pts: (6, 3)
        v1 = _dist(eye_pts[1], eye_pts[5])
        v2 = _dist(eye_pts[2], eye_pts[4])
        h = _dist(eye_pts[0], eye_pts[3]) or 1e-6
        return (v1 + v2) / (2.0 * h)

    def is_blinking(self, lm: Landmarks) -> bool:
        if lm.face is None:
            return False
        left = self._ear(lm.face[self.LEFT_EYE])
        right = self._ear(lm.face[self.RIGHT_EYE])
        return (left + right) * 0.5 < self.EAR_THRESHOLD


# ---------------------------------------------------------------------------
# Voice activity — independent of MediaPipe, runs on its own audio callback.
# ---------------------------------------------------------------------------

class VoiceActivity:
    """RMS-threshold VAD on the default input device."""

    def __init__(self, threshold: float = 0.015, samplerate: int = 16000,
                 blocksize: int = 512):
        self._level = 0.0
        self._threshold = threshold
        if not _HAS_AUDIO:
            self._stream = None
            return
        self._stream = sd.InputStream(
            channels=1, samplerate=samplerate, blocksize=blocksize,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, t, status):
        # RMS of this block; cheap and good enough for mouth flap.
        self._level = float(np.sqrt(np.mean(indata ** 2)))

    @property
    def talking(self) -> bool:
        return self._level > self._threshold

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


# ---------------------------------------------------------------------------
# Debouncer — hysteresis on state transitions to prevent flicker
# ---------------------------------------------------------------------------

class Debouncer:
    """
    A new state is only committed once it has been observed continuously for
    `hold_frames` frames. Blink and talk are passed through without hold
    (they're fast by design); pose and expression are smoothed.
    """

    def __init__(self, pose_hold: int = 5, expr_hold: int = 5):
        self._pose_hold = pose_hold
        self._expr_hold = expr_hold
        self._committed_pose: Pose = Pose.NEUTRAL
        self._committed_expr: Expression = Expression.NEUTRAL
        self._candidate_pose: tuple[Pose, int] = (Pose.NEUTRAL, 0)
        self._candidate_expr: tuple[Expression, int] = (Expression.NEUTRAL, 0)

    def update(self, pose: Pose, expr: Expression,
               blinking: bool, talking: bool) -> AvatarState:
        self._committed_pose = self._step(
            pose, self._committed_pose, "_candidate_pose", self._pose_hold)
        self._committed_expr = self._step(
            expr, self._committed_expr, "_candidate_expr", self._expr_hold)
        return AvatarState(
            pose=self._committed_pose,
            expression=self._committed_expr,
            face=FaceState(blinking=blinking, talking=talking),
        )

    def _step(self, observed, committed, attr, hold):
        cand, count = getattr(self, attr)
        if observed == committed:
            setattr(self, attr, (committed, 0))
            return committed
        if observed == cand:
            count += 1
            setattr(self, attr, (cand, count))
            if count >= hold:
                return observed
            return committed
        setattr(self, attr, (observed, 1))
        return committed


# ---------------------------------------------------------------------------
# Sprite renderer — loads the 3 x 3 x 4 grid and shows the current cell
# ---------------------------------------------------------------------------

class SpriteRenderer:
    """
    Looks for files named: sprites/{pose}__{expression}__{face}.png
    e.g.  sprites/neutral__happy__noblink_yapping.png

    Falls back to a labeled placeholder if a file is missing, so you can
    iterate on art incrementally.
    """

    def __init__(self, sprite_dir: Path, window_name: str = "PNGtuber"):
        self._dir = sprite_dir
        self._window = window_name
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def _load(self, key: tuple[str, str, str]) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        path = self._dir / f"{key[0]}__{key[1]}__{key[2]}.png"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        else:
            img = self._placeholder(key)
        self._cache[key] = img
        return img

    @staticmethod
    def _placeholder(key: tuple[str, str, str]) -> np.ndarray:
        img = np.zeros((512, 512, 4), dtype=np.uint8)
        img[:, :, :3] = 40
        img[:, :, 3] = 255
        for i, line in enumerate(key):
            cv2.putText(img, line, (20, 60 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (220, 220, 220, 255), 2)
        return img

    def show(self, state: AvatarState) -> None:
        img = self._load(state.key)
        cv2.imshow(self._window, img)

    def close(self) -> None:
        cv2.destroyWindow(self._window)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", type=Path, default=Path("./sprites"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--models", type=Path, default=Path("./models"),
                    help="Directory for MediaPipe .task model files (auto-downloaded if absent).")
    ap.add_argument("--debug", action="store_true",
                    help="Print classification features each frame.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    tracker = HolisticTracker(model_dir=args.models)
    pose_clf = PoseClassifier()
    expr_clf = ExpressionClassifier()
    blink = BlinkDetector()
    vad = VoiceActivity()
    debouncer = Debouncer()
    renderer = SpriteRenderer(args.sprites)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural feedback

            lm = tracker.process(frame)
            state = debouncer.update(
                pose=pose_clf.classify(lm),
                expr=expr_clf.classify(lm),
                blinking=blink.is_blinking(lm),
                talking=vad.talking,
            )
            renderer.show(state)

            if args.debug:
                print(state.key, end="\r")

            if cv2.waitKey(1) & 0xFF == 27:  # esc
                break
    finally:
        cap.release()
        tracker.close()
        vad.close()
        renderer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__": 
    sys.exit(main())

# ---------------------------------------------------------------------------
# Note on voice activity:
# If she'd rather let an existing PNGtuber app (Veadotube etc.) own the mouth
# flap, drop VoiceActivity from this script entirely and just render two face
# variants per (pose, expr) cell — blink/noblink — then layer Veadotube on top
# of this sprite in OBS, set to a transparent talking-mouth-only image. That
# splits the responsibilities cleanly: this script owns pose+expr+blink,
# Veadotube owns mouth shape.
# ---------------------------------------------------------------------------
