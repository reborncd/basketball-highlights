"""
detector.py
进球检测模块 — 野球场版本
核心逻辑：篮球轨迹 + 篮筐区域穿越 + 球网像素变化
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging
from app.core.yolo_detector import YOLOSportsDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    pre_roll: float = 5.0       # 进球前保留秒数
    post_roll: float = 3.0      # 进球后保留秒数
    min_ball_radius: int = 8    # 篮球最小半径(像素)
    max_ball_radius: int = 60   # 篮球最大半径(像素)
    hoop_y_ratio: float = 0.55  # 篮筐大致所在高度比例(画面上半部分)
    trajectory_min_frames: int = 3      # 判断轨迹所需最少帧数
    goal_cooldown: float = 3.0          # 两次进球最短间隔(秒), 防止重复
    sample_every_n: int = 1             # 每N帧处理一次，加快速度
    # 篮筐检测参数
    canny_low_threshold: int = 40       # Canny边缘检测低阈值
    canny_high_threshold: int = 120     # Canny边缘检测高阈值
    canny_aperture_size: int = 3        # Canny边缘检测孔径大小
    hough_dp: float = 1.0               # Hough圆变换分辨率
    hough_min_dist: int = 100           # Hough圆变换最小距离
    hough_param1: int = 50              # Hough圆变换参数1
    hough_param2: int = 25              # Hough圆变换参数2
    min_hoop_radius: int = 30           # 篮筐最小半径
    max_hoop_radius: int = 100          # 篮筐最大半径
    hoop_detection_interval: int = 20    # 篮筐检测间隔帧数
    hoop_history_size: int = 10         # 篮筐位置历史大小
    hoop_stability_threshold: int = 3    # 篮筐位置稳定阈值
    # 三阶段区域判定参数
    high_zone_offset: float = 150.0     # 高位区偏移(像素)
    goal_zone_offset: float = 150.0      # 进球区偏移(像素)
    shot_window: float = 2.5             # 时间窗口(秒)
    # 篮筐校准参数
    calibration_samples: int = 30         # 校准样本数
    # YOLO检测参数（暂时禁用）
    use_yolo: bool = False              # 是否使用YOLO检测
    yolo_model_path: str = "yolo11n.pt"  # YOLO模型路径
    yolo_conf_thres: float = 0.25       # YOLO置信度阈值
    yolo_iou_thres: float = 0.45         # YOLO IoU阈值
    yolo_imgsz: int = 640               # YOLO推理分辨率
    yolo_device: str = "auto"           # YOLO设备选择 (auto, cpu, cuda, mps)


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
        self.hoop_history: List[Tuple[int, int, int, int]] = []  # 篮筐位置历史 [(x, y, w, h)]
        self.last_detected_frame: int = -1
        self.stable_hoop_rect: Optional[tuple] = None  # 稳定的篮筐位置（加权平均）
        self.net_region_history: List[Tuple[int, int, int, int]] = []  # 球网区域历史
        self.net_region_stable: Optional[tuple] = None  # 稳定的球网区域

    def detect(self, frame: np.ndarray, config: DetectionConfig, frame_idx: int = 0) -> bool:
        """在一帧中检测篮筐，成功返回True"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # 只在画面上半部分找篮筐（野球场篮筐一般在中上方）
        roi_h = int(h * config.hoop_y_ratio)
        roi = gray[:roi_h, :]

        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # 用Canny边缘检测，使用配置参数
        edges = cv2.Canny(
            blurred,
            config.canny_low_threshold,
            config.canny_high_threshold,
            apertureSize=config.canny_aperture_size
        )
        
        # 膨胀操作增强边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 计算篮筐半径范围
        min_radius = max(config.min_hoop_radius, w // 25)
        max_radius = min(config.max_hoop_radius, w // 5)
        min_dist = max(config.hough_min_dist, w // 4)

        # 用HoughCircles检测圆形篮筐边缘
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
            # 选择最可能的篮筐（最大的圆）
            best_circle = max(circles[0], key=lambda c: c[2])
            cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            
            # 篮筐矩形区域（椭圆形，取外接矩形）
            hoop_rect = (cx - r, cy - r // 2, r * 2, r)
            # 球网区域：篮筐正下方，根据篮筐大小动态调整
            net_width = r * 2
            net_height = int(r * 2.0)  # 增加球网区域高度
            net_region = (cx - r, cy, net_width, net_height)
            
            # 更新历史记录
            self.hoop_history.append(hoop_rect)
            if len(self.hoop_history) > config.hoop_history_size:
                self.hoop_history.pop(0)
            
            # 计算稳定的篮筐位置（加权平均）
            self._update_stable_hoop(config)
            
            # 检查位置稳定性
            if self._is_position_stable(config):
                self.hoop_rect = self.stable_hoop_rect
                self.net_region = self._get_stable_net_region(net_region, config)
                self.last_detected_frame = frame_idx
                return True
        
        # 如果当前帧未检测到，尝试使用历史稳定位置
        if self.stable_hoop_rect and frame_idx - self.last_detected_frame < config.hoop_detection_interval * 2:
            # 使用稳定的篮筐位置
            self.hoop_rect = self.stable_hoop_rect
            self.net_region = self.net_region_stable
            return True
        elif self.hoop_history and frame_idx - self.last_detected_frame < config.hoop_detection_interval:
            # 备用：使用最近的历史位置
            self.hoop_rect = self.hoop_history[-1]
            cx, cy, w_hoop, h_hoop = self.hoop_rect
            r = w_hoop // 2
            self.net_region = (cx - r, cy + h_hoop // 2, r * 2, int(r * 1.8))
            return True
        
        return False
    
    def _update_stable_hoop(self, config: DetectionConfig):
        """使用加权平均计算稳定的篮筐位置"""
        if len(self.hoop_history) < 2:
            if self.hoop_history:
                self.stable_hoop_rect = self.hoop_history[0]
            return
        
        # 加权平均，越新的位置权重越大
        weights = np.linspace(0.5, 1.0, len(self.hoop_history))
        weights = weights / np.sum(weights)
        
        x = int(np.sum([r[0] * w for r, w in zip(self.hoop_history, weights)]))
        y = int(np.sum([r[1] * w for r, w in zip(self.hoop_history, weights)]))
        w_h = int(np.sum([r[2] * w for r, w in zip(self.hoop_history, weights)]))
        h_h = int(np.sum([r[3] * w for r, w in zip(self.hoop_history, weights)]))
        
        self.stable_hoop_rect = (x, y, w_h, h_h)
    
    def _get_stable_net_region(self, new_net_region: tuple, config: DetectionConfig) -> tuple:
        """计算稳定的球网区域"""
        self.net_region_history.append(new_net_region)
        if len(self.net_region_history) > config.hoop_history_size:
            self.net_region_history.pop(0)
        
        if len(self.net_region_history) < 2:
            self.net_region_stable = new_net_region
            return new_net_region
        
        # 加权平均计算稳定的球网区域
        weights = np.linspace(0.5, 1.0, len(self.net_region_history))
        weights = weights / np.sum(weights)
        
        x = int(np.sum([r[0] * w for r, w in zip(self.net_region_history, weights)]))
        y = int(np.sum([r[1] * w for r, w in zip(self.net_region_history, weights)]))
        w_n = int(np.sum([r[2] * w for r, w in zip(self.net_region_history, weights)]))
        h_n = int(np.sum([r[3] * w for r, w in zip(self.net_region_history, weights)]))
        
        self.net_region_stable = (x, y, w_n, h_n)
        return self.net_region_stable
    
    def _is_position_stable(self, config: DetectionConfig) -> bool:
        """检查篮筐位置是否稳定"""
        if len(self.hoop_history) < config.hoop_stability_threshold:
            return True
        
        # 计算最近几个位置的平均值
        recent = self.hoop_history[-config.hoop_stability_threshold:]
        avg_x = sum(r[0] for r in recent) / len(recent)
        avg_y = sum(r[1] for r in recent) / len(recent)
        avg_w = sum(r[2] for r in recent) / len(recent)
        avg_h = sum(r[3] for r in recent) / len(recent)
        
        # 检查位置变化是否在允许范围内
        for r in recent:
            dx = abs(r[0] - avg_x)
            dy = abs(r[1] - avg_y)
            dw = abs(r[2] - avg_w)
            dh = abs(r[3] - avg_h)
            
            if dx > avg_w * 0.3 or dy > avg_h * 0.3 or dw > avg_w * 0.2 or dh > avg_h * 0.2:
                return False
        
        return True

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.hoop_rect:
            x, y, w, h = self.hoop_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # 绘制篮筐中心
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
        self.max_history = 30  # 保留最近N帧的轨迹
        self.light_adaptation_factor = 1.0  # 光线适应因子

    def _adaptive_light_compensation(self, frame: np.ndarray) -> np.ndarray:
        """自适应光线补偿"""
        # 转换为灰度图计算亮度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算平均亮度
        mean_brightness = np.mean(gray)
        # 目标亮度（中等亮度）
        target_brightness = 128
        # 计算亮度调整因子
        self.light_adaptation_factor = target_brightness / mean_brightness
        # 应用亮度调整
        adjusted = cv2.convertScaleAbs(frame, alpha=self.light_adaptation_factor, beta=0)
        return adjusted

    def _get_adaptive_hsv_ranges(self) -> tuple:
        """根据光线条件获取自适应HSV范围"""
        # 基础HSV范围
        base_lower1 = np.array([0, 100, 60])
        base_upper1 = np.array([15, 255, 255])
        base_lower2 = np.array([160, 100, 60])
        base_upper2 = np.array([180, 255, 255])
        
        # 根据光线适应因子调整V通道范围
        light_factor = min(max(self.light_adaptation_factor, 0.5), 2.0)
        
        # 调整V通道的下限，光线暗时降低阈值
        adjusted_lower1 = base_lower1.copy()
        adjusted_lower2 = base_lower2.copy()
        
        # 光线越暗，V通道下限越低
        v_adjustment = int(30 * (1 - light_factor))
        adjusted_lower1[2] = max(30, base_lower1[2] + v_adjustment)
        adjusted_lower2[2] = max(30, base_lower2[2] + v_adjustment)
        
        return adjusted_lower1, base_upper1, adjusted_lower2, base_upper2

    def detect(self, frame: np.ndarray, frame_idx: int, timestamp: float,
               config: DetectionConfig) -> Optional[BallDetection]:
        # 自适应光线补偿
        adjusted_frame = self._adaptive_light_compensation(frame)
        hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)

        # 获取自适应HSV范围
        lower1, upper1, lower2, upper2 = self._get_adaptive_hsv_ranges()

        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        # 改进的形态学处理
        # 先使用小 kernel 去除噪点
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 再使用稍大的 kernel 填充内部
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 最后使用更大的 kernel 连接可能的断裂部分
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # 形态学操作序列
        mask = cv2.erode(mask, small_kernel, iterations=1)
        mask = cv2.dilate(mask, medium_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=1)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[BallDetection] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:  # 降低面积阈值，适应不同大小的篮球
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (config.min_ball_radius <= radius <= config.max_ball_radius):
                continue

            # 圆形度分数
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 改进的评分机制，结合圆形度和面积，同时考虑轨迹连续性
            score = circularity * area
            
            # 如果有历史轨迹，考虑位置连续性
            if self.history:
                last_ball = self.history[-1]
                distance = np.sqrt((cx - last_ball.cx) ** 2 + (cy - last_ball.cy) ** 2)
                # 距离越近，得分越高
                if distance < 100:  # 合理的移动距离
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
    进球检测：采用griftt/ball-yolo的三阶段区域判定法
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
        
        # 三阶段判定状态
        self.last_high_zone_ts: float = -999.0
        self.last_rim_touch_ts: float = -999.0
        
        # 篮筐校准
        self.calibration_buffer: list = []
        self.is_calibrated: bool = False
        self.locked_hoop_rect: Optional[tuple] = None
        
        # 区域定义
        self.high_zone: Optional[tuple] = None
        self.rim_zone: Optional[tuple] = None
        self.goal_zone: Optional[tuple] = None

    def _define_zones(self):
        """根据篮筐位置定义三个关键区域"""
        if not self.locked_hoop_rect:
            return
            
        hx, hy, hw, hh = self.locked_hoop_rect
        hoop_cx = hx + hw // 2
        hoop_cy = hy + hh // 2
        hoop_r = hw // 2
        
        # 高位区：篮筐上方
        offset = self.config.high_zone_offset
        self.high_zone = (
            hoop_cx - hoop_r * 2,
            hoop_cy - offset,
            hoop_cx + hoop_r * 2,
            hoop_cy + hh * 0.5
        )
        
        # 篮筐区
        self.rim_zone = (hx - 10, hy - 10, hx + hw + 10, hy + hh + 10)
        
        # 进球区：篮筐下方
        goal_offset = self.config.goal_zone_offset
        self.goal_zone = (
            hoop_cx - hoop_r * 2,
            hoop_cy + hh,
            hoop_cx + hoop_r * 2,
            hoop_cy + hh + goal_offset
        )
    
    def _is_in_zone(self, ball: BallDetection, zone: tuple) -> bool:
        """检查球是否在指定区域内"""
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

    def _calibrate_hoop(self, frame: np.ndarray, config: DetectionConfig, frame_idx: int):
        """篮筐位置校准"""
        if self.is_calibrated:
            return
        
        if self.hoop.detect(frame, config, frame_idx):
            if self.hoop.hoop_rect:
                self.calibration_buffer.append(self.hoop.hoop_rect)
                if len(self.calibration_buffer) >= config.calibration_samples:
                    # 使用中位数计算稳定的篮筐位置
                    xs = [r[0] for r in self.calibration_buffer]
                    ys = [r[1] for r in self.calibration_buffer]
                    ws = [r[2] for r in self.calibration_buffer]
                    hs = [r[3] for r in self.calibration_buffer]
                    
                    self.locked_hoop_rect = (
                        int(np.median(xs)),
                        int(np.median(ys)),
                        int(np.median(ws)),
                        int(np.median(hs))
                    )
                    
                    # 更新篮筐位置
                    self.hoop.hoop_rect = self.locked_hoop_rect
                    cx, cy, w, h = self.locked_hoop_rect
                    r = w // 2
                    self.hoop.net_region = (cx, cy + h, w, int(r * 1.8))
                    
                    # 定义三个关键区域
                    self._define_zones()
                    
                    self.is_calibrated = True
                    self.hoop_detected = True
                    logger.info(f"✅ 篮筐校准完成: 位置{self.locked_hoop_rect}")

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> bool:
        """
        处理单帧，返回是否检测到进球
        """
        self.frame_count += 1
        config = self.config

        # 首帧或未校准前进行篮筐校准
        if not self.is_calibrated:
            self._calibrate_hoop(frame, config, frame_idx)
            if not self.hoop_detected:
                if self.frame_count % 50 == 0:
                    logger.debug(f"⏳ 正在校准篮筐位置... ({len(self.calibration_buffer)}/{config.calibration_samples})")
                return False

        # 检测篮球
        ball = self.tracker.detect(frame, frame_idx, timestamp, config)
        if ball:
            self.ball_detected_count += 1
            if self.ball_detected_count % 50 == 0:
                logger.debug(f"🏀 篮球检测: 位置({ball.cx:.0f}, {ball.cy:.0f}), 半径{ball.radius:.0f}")
            
            # 检查是否在高位区
            if self._is_in_high_zone(ball):
                self.last_high_zone_ts = timestamp
                logger.debug(f"📍 篮球进入高位区: {timestamp:.1f}s")
            
            # 检查是否触框
            if self._is_touching_rim(ball):
                self.last_rim_touch_ts = timestamp
                logger.debug(f"🎯 篮球触框: {timestamp:.1f}s")
            
            # 检查是否在进球区
            if self._is_in_goal_zone(ball):
                logger.debug(f"⬇️  篮球进入进球区: {timestamp:.1f}s")
                
                # 检查时间窗口
                last_interaction = max(self.last_high_zone_ts, self.last_rim_touch_ts)
                time_diff = timestamp - last_interaction
                
                # 冷却时间检查
                if timestamp - self.last_goal_time < config.goal_cooldown:
                    return False
                
                # 检查是否在合理的时间窗口内（0.05秒到2.5秒）
                if 0.05 < time_diff < config.shot_window:
                    logger.info(f"✅ 进球检测成功！时间戳 {timestamp:.1f}s, 时间差 {time_diff:.2f}s")
                    self.last_goal_time = timestamp
                    # 重置状态
                    self.last_high_zone_ts = -999.0
                    self.last_rim_touch_ts = -999.0
                    return True

        return False

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制调试信息，包括三阶段区域"""
        frame = self.hoop.draw(frame)
        
        # 绘制三阶段区域
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
