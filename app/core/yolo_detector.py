"""
YOLO目标检测模块
使用Ultralytics YOLO进行篮球辅助检测
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

BALL_LABEL_ALIASES = {
    "sports ball",
    "basketball",
    "basketball ball",
}

HOOP_LABEL_ALIASES = {
    "basketball hoop",
    "basketball rim",
    "basket rim",
    "hoop",
    "rim",
}


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


@dataclass
class YOLOSportsDetector:
    """YOLO运动目标检测器"""

    model_path: str = "yolo11n.pt"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: int = 640
    device: str = "auto"
    model_names: dict[int, str] = field(init=False, default_factory=dict)
    ball_class_ids: set[int] = field(init=False, default_factory=set)
    hoop_class_ids: set[int] = field(init=False, default_factory=set)
    supports_hoop_detection: bool = field(init=False, default=False)

    def __post_init__(self):
        """初始化YOLO模型并解析类别映射"""
        try:
            import torch

            if torch.cuda.is_available():
                device = self.device
            else:
                device = "cpu"
                logger.info("⚠️ CUDA不可用，使用CPU")

            self.model = YOLO(self.model_path)
            self.device = device
            self.model_names = self._normalize_model_names(getattr(self.model, "names", {}))
            self.ball_class_ids = self._resolve_class_ids(BALL_LABEL_ALIASES)
            self.hoop_class_ids = self._resolve_class_ids(HOOP_LABEL_ALIASES)
            self.supports_hoop_detection = bool(self.hoop_class_ids)

            logger.info(
                "✅ YOLO模型加载成功: %s, 设备: %s, 篮球类别: %s",
                self.model_path,
                device,
                sorted(self.ball_class_ids) or "无",
            )
            if not self.supports_hoop_detection:
                logger.info("ℹ️ 当前YOLO模型不支持篮筐类别，仍将使用传统篮筐定位")
        except Exception as e:
            logger.error(f"❌ YOLO模型加载失败: {e}")
            self.model = None
            self.model_names = {}
            self.ball_class_ids = set()
            self.hoop_class_ids = set()
            self.supports_hoop_detection = False

    def _normalize_model_names(self, names) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(idx): str(label) for idx, label in names.items()}
        if isinstance(names, (list, tuple)):
            return {idx: str(label) for idx, label in enumerate(names)}
        return {}

    def _resolve_class_ids(self, aliases: set[str]) -> set[int]:
        resolved = set()
        for idx, label in self.model_names.items():
            if _normalize_label(label) in aliases:
                resolved.add(int(idx))
        return resolved

    def detect(self, frame: np.ndarray) -> list[dict]:
        """检测帧中的目标"""
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device="cpu" if self.device == "auto" else self.device,
                verbose=False,
            )

            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    detections.append({
                        "class": cls,
                        "conf": conf,
                        "bbox": bbox,
                        "label": self.model_names.get(cls, f"class_{cls}"),
                    })

            return detections
        except Exception as e:
            logger.error(f"❌ YOLO检测失败: {e}")
            return []

    def _detect_by_class_ids(self, frame: np.ndarray, class_ids: set[int]) -> Optional[dict]:
        if not class_ids:
            return None

        detections = self.detect(frame)
        matches = [d for d in detections if d["class"] in class_ids]
        if not matches:
            return None

        best = max(matches, key=lambda x: x["conf"])
        x1, y1, x2, y2 = best["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return {
            "conf": best["conf"],
            "bbox": best["bbox"],
            "center": [cx, cy],
            "label": best["label"],
        }

    def detect_ball(self, frame: np.ndarray) -> Optional[dict]:
        """检测篮球"""
        return self._detect_by_class_ids(frame, self.ball_class_ids)

    def detect_hoop(self, frame: np.ndarray) -> Optional[dict]:
        """检测篮筐；通用COCO模型通常不会提供该类别"""
        if not self.supports_hoop_detection:
            return None
        return self._detect_by_class_ids(frame, self.hoop_class_ids)

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """在帧上绘制检测结果"""
        result_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            conf = det["conf"]

            if cls in self.ball_class_ids:
                color = (0, 0, 255)
                label = f"Ball: {conf:.2f}"
            elif cls in self.hoop_class_ids:
                color = (0, 255, 0)
                label = f"Hoop: {conf:.2f}"
            else:
                color = (255, 0, 0)
                label = f"{det.get('label', f'Class {cls}')}: {conf:.2f}"

            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                result_frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return result_frame
