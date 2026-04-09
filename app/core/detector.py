"""
detector.py
进球检测模块 — 野球场版本
核心逻辑：篮球轨迹 + 篮筐区域穿越 + 球网像素变化
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    pre_roll: float = 5.0       # 进球前保留秒数
    post_roll: float = 3.0      # 进球后保留秒数
    min_ball_radius: int = 8    # 篮球最小半径(像素)
    max_ball_radius: int = 60   # 篮球最大半径(像素)
    hoop_y_ratio: float = 0.55  # 篮筐大致所在高度比例(画面上半部分)
    net_change_threshold: float = 0.03  # 球网变化阈值
    trajectory_min_frames: int = 5      # 判断轨迹所需最少帧数
    goal_cooldown: float = 8.0          # 两次进球最短间隔(秒), 防止重复
    sample_every_n: int = 2             # 每N帧处理一次，加快速度


@dataclass
class BallDetection:
    cx: float
    cy: float
    radius: float
    frame_idx: int
    timestamp: float


class HoopDetector:
    """检测并缓存篮筐位置"""

    def __init__(self):
        self.hoop_rect: Optional[tuple] = None  # (x, y, w, h)
        self.net_region: Optional[tuple] = None

    def detect(self, frame: np.ndarray, config: DetectionConfig) -> bool:
        """在一帧中检测篮筐，成功返回True"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # 只在画面上半部分找篮筐（野球场篮筐一般在中上方）
        roi_h = int(h * config.hoop_y_ratio)
        roi = gray[:roi_h, :]

        # 用Canny+HoughCircles检测圆形篮筐边缘
        edges = cv2.Canny(roi, 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w // 4,
            param1=50,
            param2=30,
            minRadius=w // 20,
            maxRadius=w // 6,
        )

        if circles is not None:
            c = circles[0][0]
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            # 篮筐矩形区域（椭圆形，取外接矩形）
            self.hoop_rect = (cx - r, cy - r // 2, r * 2, r)
            # 球网区域：篮筐正下方
            self.net_region = (cx - r, cy, r * 2, int(r * 1.5))
            return True
        return False

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.hoop_rect:
            x, y, w, h = self.hoop_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if self.net_region:
            x, y, w, h = self.net_region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 1)
        return frame


class BallTracker:
    """橙色篮球检测与轨迹追踪"""

    def __init__(self):
        self.history: list[BallDetection] = []
        self.max_history = 30  # 保留最近N帧的轨迹

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float,
               config: DetectionConfig) -> Optional[BallDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 橙色篮球的HSV范围
        lower1 = np.array([0,  120, 70])
        upper1 = np.array([15, 255, 255])
        lower2 = np.array([160, 120, 70])
        upper2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        # 形态学处理去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[BallDetection] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (config.min_ball_radius <= radius <= config.max_ball_radius):
                continue

            # 圆形度分数
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            score = circularity * area

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
    简化的进球检测：
    只需检测篮球从篮筐上方运动到篮筐下方（穿越）
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.hoop = HoopDetector()
        self.tracker = BallTracker()
        self.last_goal_time: float = -999.0
        self.hoop_detected = False
        self.frame_count = 0
        self.ball_detected_count = 0

    def _check_trajectory_through_hoop(self) -> bool:
        """检查最近轨迹是否从篮筐上方穿越到下方"""
        if not self.hoop.hoop_rect:
            return False
        positions = self.tracker.get_recent_positions(20)  # 增加追踪帧数
        if len(positions) < self.config.trajectory_min_frames:
            return False

        hx, hy, hw, hh = self.hoop.hoop_rect
        hoop_cx = hx + hw // 2
        hoop_cy = hy + hh // 2
        hoop_r = hw // 2

        # 找到在篮筐横向范围内的点
        near = [p for p in positions if abs(p.cx - hoop_cx) < hoop_r * 2.0]  # 扩大范围
        if len(near) < 2:
            return False

        # 看是否有从 cy<hoop_cy 到 cy>hoop_cy+hh 的趋势
        above = [p for p in near if p.cy < hoop_cy + hh * 0.5]  # 篮筐上半部分
        below = [p for p in near if p.cy > hoop_cy + hh * 1.5]  # 篮筐下方

        if not above or not below:
            return False

        # 时间顺序：先above后below
        last_above_idx = max(p.frame_idx for p in above)
        first_below_idx = min(p.frame_idx for p in below)
        
        # 记录轨迹信息
        logger.debug(f"轨迹检测: 上方{len(above)}, 下方{len(below)}, 上方帧{last_above_idx}, 下方帧{first_below_idx}")
        
        return first_below_idx > last_above_idx

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> bool:
        """
        处理单帧，返回是否检测到进球
        """
        self.frame_count += 1
        config = self.config

        # 首帧或每30帧重新检测篮筐
        if not self.hoop_detected or frame_idx % 30 == 0:
            if self.hoop.detect(frame, config):
                if not self.hoop_detected:
                    logger.info(f"✅ 篮筐检测成功")
                self.hoop_detected = True
            else:
                if self.frame_count % 100 == 0:
                    logger.debug(f"⏳ 尝试检测篮筐中...")

        # 检测篮球
        ball = self.tracker.detect(frame, frame_idx, timestamp, config)
        if ball:
            self.ball_detected_count += 1
            if self.ball_detected_count % 50 == 0:
                logger.debug(f"🏀 篮球检测中: 位置({ball.cx:.0f}, {ball.cy:.0f}), 半径{ball.radius:.0f}")

        # 判断进球（冷却时间内不重复触发）
        if timestamp - self.last_goal_time < config.goal_cooldown:
            return False

        trajectory_ok = self._check_trajectory_through_hoop()

        if trajectory_ok:
            self.last_goal_time = timestamp
            logger.info(f"✅ 进球检测成功！时间戳 {timestamp:.1f}s")
            return True

        return False

    def _update_net_frame(self, frame: np.ndarray):
        if self.hoop.net_region:
            x, y, w, h = self.hoop.net_region
            fh, fw = frame.shape[:2]
            x = max(0, x); y = max(0, y)
            x2 = min(fw, x + w); y2 = min(fh, y + h)
            roi = frame[y:y2, x:x2]
            if roi.size > 0:
                self.prev_net_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制调试信息"""
        frame = self.hoop.draw(frame)
        # 画最近轨迹
        pts = self.tracker.get_recent_positions(15)
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1].cx), int(pts[i-1].cy))
            p2 = (int(pts[i].cx), int(pts[i].cy))
            cv2.line(frame, p1, p2, (0, 100, 255), 2)
        if pts:
            p = pts[-1]
            cv2.circle(frame, (int(p.cx), int(p.cy)), int(p.radius), (0, 100, 255), 2)
        return frame


def run_detection(
    video_path: str,
    config: DetectionConfig,
    progress_callback=None,   # callback(current_frame, total_frames)
    cancel_flag=None,         # threading.Event, set it to stop
) -> list[float]:
    """
    对整个视频文件运行进球检测，返回进球时间戳列表（秒）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = GoalDetector(config)
    goal_timestamps: list[float] = []
    frame_idx = 0

    logger.info(f"开始检测: {video_path}, 共{total_frames}帧, {fps:.1f}fps")

    while True:
        if cancel_flag and cancel_flag.is_set():
            logger.info("检测已取消")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧加速
        if frame_idx % config.sample_every_n != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        if detector.process_frame(frame, frame_idx, timestamp):
            goal_timestamps.append(timestamp)

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx, total_frames)

        frame_idx += 1

    cap.release()
    logger.info(f"检测完成，共找到 {len(goal_timestamps)} 个进球")
    return goal_timestamps
