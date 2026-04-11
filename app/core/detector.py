"""
detector.py
进球检测模块 — 固定机位单篮筐版本
核心逻辑：篮筐锁定 + 篮球检测 + 三阶段区域判定
"""
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.core.yolo_detector import YOLOSportsDetector

logger = logging.getLogger(__name__)

FAILURE_REASON_MESSAGES = {
    "hoop_not_locked": "未定位到篮筐",
    "ball_never_detected": "已定位篮筐但未检测到篮球",
    "no_goal_pattern": "检测到篮球但未形成进球轨迹",
    "cancelled": "检测已取消",
}


def describe_failure_reason(reason: Optional[str]) -> str:
    if not reason:
        return "无"
    return FAILURE_REASON_MESSAGES.get(reason, reason)


@dataclass
class DetectionConfig:
    pre_roll: float = 5.0
    post_roll: float = 3.0
    min_ball_radius: int = 8
    max_ball_radius: int = 60
    hoop_y_ratio: float = 0.55
    trajectory_min_frames: int = 3
    goal_cooldown: float = 2.0
    sample_every_n: int = 1
    canny_low_threshold: int = 40
    canny_high_threshold: int = 120
    canny_aperture_size: int = 3
    hough_dp: float = 1.0
    hough_min_dist: int = 100
    hough_param1: int = 50
    hough_param2: int = 25
    min_hoop_radius: int = 30
    max_hoop_radius: int = 100
    hoop_detection_interval: int = 20
    hoop_history_size: int = 10
    hoop_stability_threshold: int = 3
    high_zone_offset: float = 150.0
    goal_zone_offset: float = 150.0
    shot_window: float = 3.0
    zone_half_width_scale: float = 2.5
    rim_zone_padding: int = 20
    trajectory_fallback_window: float = 1.0
    trajectory_fallback_min_points: int = 3
    trajectory_fallback_vertical_margin: int = 12
    calibration_samples: int = 10
    hoop_lock_timeout_frames: int = 300
    use_yolo: bool = False
    yolo_model_path: str = "yolo11n.pt"
    yolo_conf_thres: float = 0.25
    yolo_iou_thres: float = 0.45
    yolo_imgsz: int = 640
    yolo_device: str = "auto"
    manual_hoop_rect: Optional[tuple[int, int, int, int]] = None


@dataclass
class DetectionStats:
    total_frames: int = 0
    processed_frames: int = 0
    hoop_detected_frames: int = 0
    ball_detected_frames: int = 0
    high_zone_hits: int = 0
    rim_touches: int = 0
    goal_zone_hits: int = 0
    goal_candidates: int = 0
    calibration_hits: int = 0
    hoop_locked: bool = False
    manual_hoop_used: bool = False
    yolo_ball_enabled: bool = False
    yolo_hoop_supported: bool = False
    fallback_goal_candidates: int = 0
    goal_zone_rejected_cooldown: int = 0
    goal_zone_rejected_no_interaction: int = 0
    goal_zone_rejected_timing: int = 0


@dataclass
class DetectionRunResult:
    timestamps: list[float] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
    config_snapshot: dict = field(default_factory=dict)
    summary: str = ""


@dataclass
class BallDetection:
    cx: float
    cy: float
    radius: float
    frame_idx: int
    timestamp: float


def format_detection_summary(result: DetectionRunResult) -> str:
    stats = result.stats
    sample_every_n = result.config_snapshot.get("sample_every_n")
    parts = [
        f"总帧数 {stats.get('total_frames', 0)}",
        f"已处理 {stats.get('processed_frames', 0)}",
        f"篮筐命中 {stats.get('hoop_detected_frames', 0)}",
        f"篮球命中 {stats.get('ball_detected_frames', 0)}",
        f"高位区 {stats.get('high_zone_hits', 0)}",
        f"触框 {stats.get('rim_touches', 0)}",
        f"进球区 {stats.get('goal_zone_hits', 0)}",
        f"候选进球 {len(result.timestamps)}",
        f"回退命中 {stats.get('fallback_goal_candidates', 0)}",
        f"冷却拦截 {stats.get('goal_zone_rejected_cooldown', 0)}",
        f"无交互拦截 {stats.get('goal_zone_rejected_no_interaction', 0)}",
        f"时间窗拦截 {stats.get('goal_zone_rejected_timing', 0)}",
    ]
    if sample_every_n:
        parts.append(f"采样间隔 {sample_every_n}")
    if result.failure_reason:
        parts.append(f"失败原因 {describe_failure_reason(result.failure_reason)}")
    return "检测摘要: " + " | ".join(parts)


class HoopDetector:
    """检测并缓存篮筐位置"""

    def __init__(self):
        self.hoop_rect: Optional[tuple[int, int, int, int]] = None
        self.net_region: Optional[tuple[int, int, int, int]] = None
        self.hoop_history: list[tuple[int, int, int, int]] = []
        self.last_detected_frame: int = -1
        self.stable_hoop_rect: Optional[tuple[int, int, int, int]] = None
        self.net_region_history: list[tuple[int, int, int, int]] = []
        self.net_region_stable: Optional[tuple[int, int, int, int]] = None

    def detect(self, frame: np.ndarray, config: DetectionConfig, frame_idx: int = 0) -> bool:
        """在一帧中检测篮筐，成功返回True"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        roi_h = int(h * config.hoop_y_ratio)
        roi = gray[:roi_h, :]

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            config.canny_low_threshold,
            config.canny_high_threshold,
            apertureSize=config.canny_aperture_size,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        min_radius = max(config.min_hoop_radius, w // 25)
        max_radius = min(config.max_hoop_radius, w // 5)
        min_dist = max(config.hough_min_dist, w // 4)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=config.hough_dp,
            minDist=min_dist,
            param1=config.hough_param1,
            param2=config.hough_param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is not None:
            best_circle = max(circles[0], key=lambda c: c[2])
            cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            hoop_rect = (cx - r, cy - r // 2, r * 2, r)
            net_region = (cx - r, cy, r * 2, int(r * 2.0))

            self.hoop_history.append(hoop_rect)
            if len(self.hoop_history) > config.hoop_history_size:
                self.hoop_history.pop(0)

            self._update_stable_hoop()

            if self._is_position_stable(config):
                self.hoop_rect = self.stable_hoop_rect
                self.net_region = self._get_stable_net_region(net_region, config)
                self.last_detected_frame = frame_idx
                return True

        if self.stable_hoop_rect and frame_idx - self.last_detected_frame < config.hoop_detection_interval * 2:
            self.hoop_rect = self.stable_hoop_rect
            self.net_region = self.net_region_stable
            return True
        if self.hoop_history and frame_idx - self.last_detected_frame < config.hoop_detection_interval:
            self.hoop_rect = self.hoop_history[-1]
            x, y, w_hoop, h_hoop = self.hoop_rect
            r = w_hoop // 2
            self.net_region = (x, y + h_hoop, w_hoop, int(r * 1.8))
            return True

        return False

    def _update_stable_hoop(self):
        if len(self.hoop_history) < 2:
            if self.hoop_history:
                self.stable_hoop_rect = self.hoop_history[0]
            return

        weights = np.linspace(0.5, 1.0, len(self.hoop_history))
        weights = weights / np.sum(weights)

        x = int(np.sum([r[0] * w for r, w in zip(self.hoop_history, weights)]))
        y = int(np.sum([r[1] * w for r, w in zip(self.hoop_history, weights)]))
        w_h = int(np.sum([r[2] * w for r, w in zip(self.hoop_history, weights)]))
        h_h = int(np.sum([r[3] * w for r, w in zip(self.hoop_history, weights)]))
        self.stable_hoop_rect = (x, y, w_h, h_h)

    def _get_stable_net_region(
        self,
        new_net_region: tuple[int, int, int, int],
        config: DetectionConfig,
    ) -> tuple[int, int, int, int]:
        self.net_region_history.append(new_net_region)
        if len(self.net_region_history) > config.hoop_history_size:
            self.net_region_history.pop(0)

        if len(self.net_region_history) < 2:
            self.net_region_stable = new_net_region
            return new_net_region

        weights = np.linspace(0.5, 1.0, len(self.net_region_history))
        weights = weights / np.sum(weights)
        x = int(np.sum([r[0] * w for r, w in zip(self.net_region_history, weights)]))
        y = int(np.sum([r[1] * w for r, w in zip(self.net_region_history, weights)]))
        w_n = int(np.sum([r[2] * w for r, w in zip(self.net_region_history, weights)]))
        h_n = int(np.sum([r[3] * w for r, w in zip(self.net_region_history, weights)]))
        self.net_region_stable = (x, y, w_n, h_n)
        return self.net_region_stable

    def _is_position_stable(self, config: DetectionConfig) -> bool:
        if len(self.hoop_history) < config.hoop_stability_threshold:
            return True

        recent = self.hoop_history[-config.hoop_stability_threshold:]
        avg_x = sum(r[0] for r in recent) / len(recent)
        avg_y = sum(r[1] for r in recent) / len(recent)
        avg_w = sum(r[2] for r in recent) / len(recent)
        avg_h = sum(r[3] for r in recent) / len(recent)

        for rect in recent:
            dx = abs(rect[0] - avg_x)
            dy = abs(rect[1] - avg_y)
            dw = abs(rect[2] - avg_w)
            dh = abs(rect[3] - avg_h)
            if dx > avg_w * 0.3 or dy > avg_h * 0.3 or dw > avg_w * 0.2 or dh > avg_h * 0.2:
                return False

        return True

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.hoop_rect:
            x, y, w, h = self.hoop_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        if self.net_region:
            x, y, w, h = self.net_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 1)
        return frame


class BallTracker:
    """橙色篮球检测与轨迹追踪"""

    def __init__(self):
        self.history: list[BallDetection] = []
        self.max_history = 30
        self.light_adaptation_factor = 1.0

    def _adaptive_light_compensation(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = max(1.0, float(np.mean(gray)))
        target_brightness = 128.0
        self.light_adaptation_factor = target_brightness / mean_brightness
        return cv2.convertScaleAbs(frame, alpha=self.light_adaptation_factor, beta=0)

    def _get_adaptive_hsv_ranges(self) -> tuple:
        base_lower1 = np.array([0, 100, 60])
        base_upper1 = np.array([15, 255, 255])
        base_lower2 = np.array([160, 100, 60])
        base_upper2 = np.array([180, 255, 255])

        light_factor = min(max(self.light_adaptation_factor, 0.5), 2.0)
        adjusted_lower1 = base_lower1.copy()
        adjusted_lower2 = base_lower2.copy()
        v_adjustment = int(30 * (1 - light_factor))
        adjusted_lower1[2] = max(30, base_lower1[2] + v_adjustment)
        adjusted_lower2[2] = max(30, base_lower2[2] + v_adjustment)

        return adjusted_lower1, base_upper1, adjusted_lower2, base_upper2

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        config: DetectionConfig,
    ) -> Optional[BallDetection]:
        adjusted_frame = self._adaptive_light_compensation(frame)
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
        lower1, upper1, lower2, upper2 = self._get_adaptive_hsv_ranges()
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, small_kernel, iterations=1)
        mask = cv2.dilate(mask, medium_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[BallDetection] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (config.min_ball_radius <= radius <= config.max_ball_radius):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            score = circularity * area
            if self.history:
                last_ball = self.history[-1]
                distance = np.sqrt((cx - last_ball.cx) ** 2 + (cy - last_ball.cy) ** 2)
                if distance < 100:
                    score *= (1 + 0.5 * (1 - distance / 100))

            if score > best_score:
                best_score = score
                best = BallDetection(cx, cy, radius, frame_idx, timestamp)

        if best:
            self.history.append(best)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        return best

    def get_recent_positions(self, n: int = 10) -> list[BallDetection]:
        return self.history[-n:]


class GoalDetector:
    """
    进球检测：
    1. 高位区 - 篮球在篮筐上方
    2. 触框区 - 篮球与篮筐重叠
    3. 进球区 - 篮球穿过篮筐下方
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.hoop = HoopDetector()
        self.tracker = BallTracker()
        self.last_goal_time: float = -999.0
        self.hoop_detected = False
        self.frame_count = 0
        self.ball_detected_count = 0
        self.last_high_zone_ts: float = -999.0
        self.last_rim_touch_ts: float = -999.0
        self.calibration_buffer: list[tuple[int, int, int, int]] = []
        self.is_calibrated: bool = False
        self.locked_hoop_rect: Optional[tuple[int, int, int, int]] = None
        self.high_zone: Optional[tuple[float, float, float, float]] = None
        self.rim_zone: Optional[tuple[float, float, float, float]] = None
        self.goal_zone: Optional[tuple[float, float, float, float]] = None
        self.stats = DetectionStats(
            manual_hoop_used=bool(config.manual_hoop_rect),
            yolo_ball_enabled=bool(config.use_yolo),
        )

        self.yolo_detector = None
        if config.use_yolo:
            try:
                self.yolo_detector = YOLOSportsDetector(
                    model_path=config.yolo_model_path,
                    conf_thres=config.yolo_conf_thres,
                    iou_thres=config.yolo_iou_thres,
                    imgsz=config.yolo_imgsz,
                    device=config.yolo_device,
                )
                self.stats.yolo_hoop_supported = self.yolo_detector.supports_hoop_detection
                logger.info("✅ YOLO辅助篮球检测已启用")
            except Exception as e:
                logger.error(f"❌ YOLO初始化失败: {e}")
                self.yolo_detector = None
                self.stats.yolo_hoop_supported = False

        if config.manual_hoop_rect:
            self._lock_hoop_rect(config.manual_hoop_rect, source="手动标定")

    def _build_net_region(self, hoop_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = hoop_rect
        return (x, y + h, w, int(max(h * 1.5, w * 0.9)))

    def _lock_hoop_rect(self, hoop_rect: tuple[int, int, int, int], source: str):
        normalized = tuple(int(v) for v in hoop_rect)
        self.locked_hoop_rect = normalized
        self.hoop.hoop_rect = normalized
        self.hoop.net_region = self._build_net_region(normalized)
        self.hoop.net_region_stable = self.hoop.net_region
        self.is_calibrated = True
        self.hoop_detected = True
        self.stats.hoop_locked = True
        self._define_zones()
        logger.info("✅ 篮筐锁定完成(%s): %s", source, normalized)

    def _define_zones(self):
        if not self.locked_hoop_rect:
            return

        hx, hy, hw, hh = self.locked_hoop_rect
        hoop_cx = hx + hw // 2
        hoop_cy = hy + hh // 2
        hoop_r = hw // 2
        half_width = hoop_r * self.config.zone_half_width_scale
        rim_padding = self.config.rim_zone_padding

        self.high_zone = (
            hoop_cx - half_width,
            hoop_cy - self.config.high_zone_offset,
            hoop_cx + half_width,
            hy + hh,
        )
        self.rim_zone = (
            hx - rim_padding,
            hy - rim_padding,
            hx + hw + rim_padding,
            hy + hh + rim_padding,
        )
        self.goal_zone = (
            hoop_cx - half_width,
            hy + hh,
            hoop_cx + half_width,
            hy + hh + self.config.goal_zone_offset,
        )

    def _is_in_zone(self, ball: BallDetection, zone: Optional[tuple[float, float, float, float]]) -> bool:
        if not zone:
            return False
        zx1, zy1, zx2, zy2 = zone
        return zx1 < ball.cx < zx2 and zy1 < ball.cy < zy2

    def _is_in_high_zone(self, ball: BallDetection) -> bool:
        return self._is_in_zone(ball, self.high_zone)

    def _is_touching_rim(self, ball: BallDetection) -> bool:
        return self._is_in_zone(ball, self.rim_zone)

    def _is_in_goal_zone(self, ball: BallDetection) -> bool:
        return self._is_in_zone(ball, self.goal_zone)

    def _trajectory_supports_goal(self, current_ball: BallDetection) -> bool:
        if not self.locked_hoop_rect:
            return False

        recent = [
            point
            for point in self.tracker.get_recent_positions(self.tracker.max_history)
            if current_ball.timestamp - point.timestamp <= self.config.trajectory_fallback_window
        ]
        if len(recent) < self.config.trajectory_fallback_min_points:
            return False

        hx, hy, hw, hh = self.locked_hoop_rect
        hoop_cx = hx + hw / 2
        hoop_top = hy - self.config.trajectory_fallback_vertical_margin
        hoop_bottom = hy + hh + self.config.trajectory_fallback_vertical_margin
        lateral_limit = (hw / 2) * self.config.zone_half_width_scale

        above_points = [
            point for point in recent
            if point.cy <= hoop_top and abs(point.cx - hoop_cx) <= lateral_limit
        ]
        below_points = [
            point for point in recent
            if point.cy >= hoop_bottom and abs(point.cx - hoop_cx) <= lateral_limit
        ]
        if not above_points or not below_points:
            return False

        first = recent[0]
        last = recent[-1]
        if last.cy <= first.cy:
            return False

        return True

    def _calibrate_hoop(self, frame: np.ndarray, frame_idx: int):
        if self.is_calibrated:
            return

        if self.hoop.detect(frame, self.config, frame_idx) and self.hoop.hoop_rect:
            self.calibration_buffer.append(self.hoop.hoop_rect)
            self.stats.calibration_hits += 1
            if len(self.calibration_buffer) >= self.config.calibration_samples:
                xs = [r[0] for r in self.calibration_buffer]
                ys = [r[1] for r in self.calibration_buffer]
                ws = [r[2] for r in self.calibration_buffer]
                hs = [r[3] for r in self.calibration_buffer]
                median_rect = (
                    int(np.median(xs)),
                    int(np.median(ys)),
                    int(np.median(ws)),
                    int(np.median(hs)),
                )
                self._lock_hoop_rect(median_rect, source="自动校准")

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> bool:
        self.frame_count += 1
        self.stats.processed_frames += 1

        if not self.is_calibrated:
            self._calibrate_hoop(frame, frame_idx)
            if not self.hoop_detected:
                return False

        self.stats.hoop_detected_frames += 1

        ball = None
        if self.yolo_detector:
            yolo_ball = self.yolo_detector.detect_ball(frame)
            if yolo_ball and yolo_ball["conf"] >= self.config.yolo_conf_thres:
                cx, cy = yolo_ball["center"]
                x1, y1, x2, y2 = yolo_ball["bbox"]
                radius = max(1.0, (x2 - x1) / 2)
                ball = BallDetection(cx, cy, radius, frame_idx, timestamp)
                self.tracker.history.append(ball)
                if len(self.tracker.history) > self.tracker.max_history:
                    self.tracker.history.pop(0)
        else:
            ball = self.tracker.detect(frame, frame_idx, timestamp, self.config)

        if not ball:
            return False

        self.ball_detected_count += 1
        self.stats.ball_detected_frames += 1

        if self._is_in_high_zone(ball):
            self.last_high_zone_ts = timestamp
            self.stats.high_zone_hits += 1

        if self._is_touching_rim(ball):
            self.last_rim_touch_ts = timestamp
            self.stats.rim_touches += 1

        if not self._is_in_goal_zone(ball):
            return False

        self.stats.goal_zone_hits += 1
        last_interaction = max(self.last_high_zone_ts, self.last_rim_touch_ts)
        time_diff = timestamp - last_interaction

        if timestamp - self.last_goal_time < self.config.goal_cooldown:
            self.stats.goal_zone_rejected_cooldown += 1
            return False

        if last_interaction < 0:
            if self._trajectory_supports_goal(ball):
                logger.info(f"✅ 进球检测成功(轨迹回退)！时间戳 {timestamp:.1f}s")
                self.last_goal_time = timestamp
                self.last_high_zone_ts = -999.0
                self.last_rim_touch_ts = -999.0
                self.stats.goal_candidates += 1
                self.stats.fallback_goal_candidates += 1
                return True
            self.stats.goal_zone_rejected_no_interaction += 1
            return False

        if 0.05 < time_diff < self.config.shot_window:
            logger.info(f"✅ 进球检测成功！时间戳 {timestamp:.1f}s, 时间差 {time_diff:.2f}s")
            self.last_goal_time = timestamp
            self.last_high_zone_ts = -999.0
            self.last_rim_touch_ts = -999.0
            self.stats.goal_candidates += 1
            return True

        if self._trajectory_supports_goal(ball):
            logger.info(f"✅ 进球检测成功(轨迹回退)！时间戳 {timestamp:.1f}s, 时间差 {time_diff:.2f}s")
            self.last_goal_time = timestamp
            self.last_high_zone_ts = -999.0
            self.last_rim_touch_ts = -999.0
            self.stats.goal_candidates += 1
            self.stats.fallback_goal_candidates += 1
            return True

        self.stats.goal_zone_rejected_timing += 1
        return False

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        frame = self.hoop.draw(frame)

        if self.high_zone:
            zx1, zy1, zx2, zy2 = self.high_zone
            cv2.rectangle(frame, (int(zx1), int(zy1)), (int(zx2), int(zy2)), (255, 255, 0), 2)
            cv2.putText(frame, "HIGH", (int(zx1), int(zy1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if self.rim_zone:
            zx1, zy1, zx2, zy2 = self.rim_zone
            cv2.rectangle(frame, (int(zx1), int(zy1)), (int(zx2), int(zy2)), (0, 255, 255), 2)

        if self.goal_zone:
            zx1, zy1, zx2, zy2 = self.goal_zone
            cv2.rectangle(frame, (int(zx1), int(zy1)), (int(zx2), int(zy2)), (0, 255, 0), 2)
            cv2.putText(frame, "GOAL", (int(zx1), int(zy1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pts = self.tracker.get_recent_positions(15)
        for idx in range(1, len(pts)):
            p1 = (int(pts[idx - 1].cx), int(pts[idx - 1].cy))
            p2 = (int(pts[idx].cx), int(pts[idx].cy))
            cv2.line(frame, p1, p2, (0, 100, 255), 2)
        if pts:
            p = pts[-1]
            cv2.circle(frame, (int(p.cx), int(p.cy)), int(p.radius), (0, 100, 255), 2)
        return frame


def _detect_failure_reason(
    detector: GoalDetector,
    timestamps: list[float],
    cancelled: bool,
) -> Optional[str]:
    if cancelled:
        return "cancelled"
    if timestamps:
        return None
    if not detector.hoop_detected:
        return "hoop_not_locked"
    if detector.stats.ball_detected_frames == 0:
        return "ball_never_detected"
    return "no_goal_pattern"


def run_detection(
    video_path: str,
    config: DetectionConfig,
    progress_callback=None,
    cancel_flag=None,
) -> DetectionRunResult:
    """
    对整个视频文件运行进球检测，返回结构化检测结果
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = GoalDetector(config)
    detector.stats.total_frames = total_frames
    goal_timestamps: list[float] = []
    frame_idx = 0
    cancelled = False
    timed_out_on_hoop = False

    logger.info(f"开始检测: {video_path}, 共{total_frames}帧, {fps:.1f}fps")

    while True:
        if cancel_flag and cancel_flag.is_set():
            cancelled = True
            logger.info("检测已取消")
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % config.sample_every_n != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        if detector.process_frame(frame, frame_idx, timestamp):
            goal_timestamps.append(timestamp)

        if (
            not detector.hoop_detected
            and not config.manual_hoop_rect
            and detector.stats.processed_frames >= config.hoop_lock_timeout_frames
        ):
            timed_out_on_hoop = True
            logger.warning(
                "⚠️ 在前 %s 个已处理帧内未能锁定篮筐，提前结束检测",
                config.hoop_lock_timeout_frames,
            )
            break

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()

    failure_reason = _detect_failure_reason(detector, goal_timestamps, cancelled)
    if timed_out_on_hoop:
        failure_reason = "hoop_not_locked"

    result = DetectionRunResult(
        timestamps=goal_timestamps,
        stats=asdict(detector.stats),
        failure_reason=failure_reason,
        config_snapshot=asdict(config),
    )
    result.summary = format_detection_summary(result)

    logger.info(result.summary)
    if result.failure_reason and result.failure_reason != "cancelled":
        logger.warning("⚠️ 本次检测未产出候选进球: %s", describe_failure_reason(result.failure_reason))
    else:
        logger.info(f"检测完成，共找到 {len(goal_timestamps)} 个进球")

    return result
