"""Detect players and estimate 2D poses using YOLOv8 pose models.

Uses the *ultralytics* library for both person detection and keypoint
estimation via a single pose-variant model (e.g. ``yolov8m-pose``).
Each detected person yields 17 COCO keypoints with (x, y, confidence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..constants import COCO_KEYPOINTS, COCO_KEYPOINT_NAMES

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Single person detection with bounding box and pose keypoints."""

    bbox: np.ndarray  # (4,) — x1, y1, x2, y2 in pixels
    bbox_confidence: float
    keypoints: np.ndarray  # (17, 3) — x, y, confidence per COCO keypoint
    person_id: int = -1  # Tracking ID if available, else -1

    @property
    def num_visible(self) -> int:
        """Number of keypoints with confidence above a minimal threshold."""
        return int(np.sum(self.keypoints[:, 2] > 0.1))

    @property
    def mean_keypoint_confidence(self) -> float:
        """Average confidence across all keypoints."""
        return float(np.mean(self.keypoints[:, 2]))


@dataclass
class FrameDetections:
    """All person detections for a single frame."""

    frame_index: int
    detections: list[PersonDetection] = field(default_factory=list)

    @property
    def num_persons(self) -> int:
        return len(self.detections)


class PoseDetector:
    """YOLOv8-based person detection and 2D pose estimation.

    Parameters
    ----------
    model_name : str
        Name or path of the YOLOv8 pose model.  The ultralytics library
        will download weights automatically if not cached.
    detection_conf : float
        Minimum bounding-box confidence to keep a detection.
    keypoint_conf : float
        Minimum per-keypoint confidence; keypoints below this threshold
        are zeroed out (marked missing).
    device : str
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"cuda:0"``).
    min_visible_keypoints : int
        Discard detections with fewer visible keypoints than this.
    """

    def __init__(
        self,
        model_name: str = "yolov8m-pose",
        detection_conf: float = 0.5,
        keypoint_conf: float = 0.3,
        device: str = "cpu",
        min_visible_keypoints: int = 6,
    ) -> None:
        self.model_name = model_name
        self.detection_conf = detection_conf
        self.keypoint_conf = keypoint_conf
        self.device = device
        self.min_visible_keypoints = min_visible_keypoints
        self._model = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for pose detection. "
                "Install it with: pip install ultralytics"
            ) from exc

        logger.info("Loading pose model: %s on %s", self.model_name, self.device)
        self._model = YOLO(self.model_name)
        self._model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_frame(self, image: np.ndarray, frame_index: int = 0) -> FrameDetections:
        """Run detection on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image of shape (H, W, 3), dtype uint8.
        frame_index : int
            Index to attach to the result (for bookkeeping).

        Returns
        -------
        FrameDetections
            Detected persons with keypoints for this frame.
        """
        self._load_model()

        if image is None or image.size == 0:
            logger.warning("Empty image at frame %d, returning no detections", frame_index)
            return FrameDetections(frame_index=frame_index)

        results = self._model.predict(
            image,
            conf=self.detection_conf,
            verbose=False,
        )

        detections: list[PersonDetection] = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes
            kpts = result.keypoints

            # kpts.data shape: (num_persons, 17, 3) — x, y, conf
            kpts_data = kpts.data.cpu().numpy() if hasattr(kpts.data, "cpu") else np.array(kpts.data)
            boxes_xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            boxes_conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)

            for i in range(len(boxes_xyxy)):
                keypoints = np.zeros((COCO_KEYPOINTS, 3), dtype=np.float32)

                if i < len(kpts_data):
                    raw_kpts = kpts_data[i]  # (17, 3) or (17, 2)
                    n_kpts = min(raw_kpts.shape[0], COCO_KEYPOINTS)

                    if raw_kpts.shape[1] >= 3:
                        keypoints[:n_kpts] = raw_kpts[:n_kpts, :3]
                    else:
                        keypoints[:n_kpts, :2] = raw_kpts[:n_kpts, :2]
                        keypoints[:n_kpts, 2] = 1.0  # Assume visible

                # Zero out low-confidence keypoints
                low_conf_mask = keypoints[:, 2] < self.keypoint_conf
                keypoints[low_conf_mask] = 0.0

                det = PersonDetection(
                    bbox=boxes_xyxy[i].astype(np.float32),
                    bbox_confidence=float(boxes_conf[i]),
                    keypoints=keypoints,
                )

                # Filter out detections with too few visible keypoints
                if det.num_visible < self.min_visible_keypoints:
                    continue

                detections.append(det)

        logger.debug("Frame %d: %d persons detected", frame_index, len(detections))
        return FrameDetections(frame_index=frame_index, detections=detections)

    def detect_batch(
        self,
        images: Sequence[np.ndarray],
        start_index: int = 0,
    ) -> list[FrameDetections]:
        """Run detection on a batch of images.

        Parameters
        ----------
        images : sequence of np.ndarray
            BGR images, each of shape (H, W, 3).
        start_index : int
            Frame index of the first image (indices increment by 1).

        Returns
        -------
        list[FrameDetections]
            One ``FrameDetections`` per input image, in order.
        """
        return [
            self.detect_frame(img, frame_index=start_index + i)
            for i, img in enumerate(images)
        ]
