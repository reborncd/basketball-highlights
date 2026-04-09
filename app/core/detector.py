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
    net_change_threshold: float = 0.025  # 球网变化阈值(降低以提高灵敏度)
    trajectory_min_frames: int = 5      # 判断轨迹所需最少帧数
    goal_cooldown: float = 8.0          # 两次进球最短间隔(秒), 防止重复
    sample_every_n: int = 2             # 每N帧处理一次，加快速度
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
    # 球网检测增强参数
    net_light_compensation: bool = True  # 是否启用光线补偿
    net_edge_weight: float = 0.6        # 边缘变化权重
    net_intensity_weight: float = 0.4   # 强度变化权重
    net_histogram_bins: int = 16        # 直方图 bins 数
    net_histogram_threshold: float = 0.15  # 直方图变化阈值
    net_motion_threshold: int = 15      # 运动检测阈值
    # YOLO检测参数
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
    进球检测：结合传统方法和YOLO深度学习
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.hoop = HoopDetector()
        self.tracker = BallTracker()
        self.last_goal_time: float = -999.0
        self.hoop_detected = False
        self.frame_count = 0
        self.ball_detected_count = 0
        # 球网检测相关变量
        self.prev_net_frame = None
        self.prev_net_edge = None
        self.prev_net_hist = None
        self.net_change_history = []
        self.net_change_window = 7  # 扩大球网变化检测窗口大小
        self.net_stability_threshold = 0.015  # 球网稳定阈值
        self.light_level_history = []  # 光线强度历史
        self.max_light_history = 10
        # YOLO检测相关
        self.yolo_detector = None
        if config.use_yolo:
            try:
                self.yolo_detector = YOLOSportsDetector(
                    model_path=config.yolo_model_path,
                    conf_thres=config.yolo_conf_thres,
                    iou_thres=config.yolo_iou_thres,
                    imgsz=config.yolo_imgsz,
                    device=config.yolo_device
                )
                logger.info("✅ YOLO检测已启用")
            except Exception as e:
                logger.error(f"❌ YOLO初始化失败: {e}")
                self.yolo_detector = None

    def _predict_trajectory(self, positions: list[BallDetection]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """使用线性回归预测轨迹
        返回 (a, b, r2): y = a*x + b, r2是拟合度
        """
        if len(positions) < 3:
            return None, None, None
        
        x = np.array([p.cx for p in positions], dtype=np.float32)
        y = np.array([p.cy for p in positions], dtype=np.float32)
        
        # 线性回归
        if len(x) > 1:
            coefficients = np.polyfit(x, y, 1)
            a, b = coefficients
            # 计算R²
            y_pred = a * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return a, b, r2
        return None, None, None

    def _check_trajectory_through_hoop(self) -> bool:
        """检查最近轨迹是否从篮筐上方穿越到下方（简化版）"""
        if not self.hoop.hoop_rect:
            return False
        positions = self.tracker.get_recent_positions(20)
        if len(positions) < self.config.trajectory_min_frames:
            return False

        hx, hy, hw, hh = self.hoop.hoop_rect
        hoop_cx = hx + hw // 2
        hoop_cy = hy + hh // 2
        hoop_r = hw // 2

        # 找到在篮筐横向范围内的点
        near = [p for p in positions if abs(p.cx - hoop_cx) < hoop_r * 2.5]  # 扩大范围
        if len(near) < 2:
            logger.debug(f"轨迹检测失败: 篮筐附近点不够 {len(near)}")
            return False

        # 传统的上下穿越判断（简化条件）
        above = [p for p in near if p.cy < hoop_cy + hh * 0.8]  # 篮筐上方（扩大范围）
        below = [p for p in near if p.cy > hoop_cy + hh * 1.2]  # 篮筐下方（扩大范围）

        if not above or not below:
            logger.debug(f"轨迹检测失败: 上方{len(above)}, 下方{len(below)}")
            return False

        # 时间顺序：先above后below
        last_above = max(above, key=lambda p: p.frame_idx)
        first_below = min(below, key=lambda p: p.frame_idx)
        
        if first_below.frame_idx <= last_above.frame_idx:
            logger.debug(f"轨迹检测失败: 时间顺序不对")
            return False
        
        # 记录轨迹信息
        logger.info(f"✅ 轨迹检测成功: 上方{len(above)}, 下方{len(below)}, 上方帧{last_above.frame_idx}, 下方帧{first_below.frame_idx}")
        
        return True

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> bool:
        """
        处理单帧，返回是否检测到进球
        """
        self.frame_count += 1
        config = self.config

        # 首帧或按配置间隔重新检测篮筐
        if not self.hoop_detected or frame_idx % config.hoop_detection_interval == 0:
            # 优先使用YOLO检测篮筐
            if self.yolo_detector:
                yolo_hoop = self.yolo_detector.detect_hoop(frame)
                if yolo_hoop and yolo_hoop["conf"] > config.yolo_conf_thres:
                    x1, y1, x2, y2 = yolo_hoop["bbox"]
                    # 设置篮筐位置
                    self.hoop.hoop_rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    self.hoop.net_region = (int(x1), int(y1), int(x2 - x1), int((y2 - y1) * 1.5))
                    if not self.hoop_detected:
                        logger.info(f"✅ YOLO篮筐检测成功")
                    self.hoop_detected = True
            else:
                # 传统方法检测篮筐
                if self.hoop.detect(frame, config, frame_idx):
                    if not self.hoop_detected:
                        logger.info(f"✅ 传统篮筐检测成功")
                    self.hoop_detected = True
                else:
                    if self.frame_count % 100 == 0:
                        logger.debug(f"⏳ 尝试检测篮筐中...")

        # 检测篮球
        ball = None
        if self.yolo_detector:
            # 使用YOLO检测篮球
            yolo_ball = self.yolo_detector.detect_ball(frame)
            if yolo_ball and yolo_ball["conf"] > config.yolo_conf_thres:
                cx, cy = yolo_ball["center"]
                x1, y1, x2, y2 = yolo_ball["bbox"]
                radius = (x2 - x1) / 2
                ball = BallDetection(cx, cy, radius, frame_idx, timestamp)
                self.tracker.history.append(ball)
                if len(self.tracker.history) > self.tracker.max_history:
                    self.tracker.history.pop(0)
                self.ball_detected_count += 1
                if self.ball_detected_count % 50 == 0:
                    logger.debug(f"🏀 YOLO篮球检测: 位置({cx:.0f}, {cy:.0f}), 置信度{yolo_ball['conf']:.2f}")
        else:
            # 传统方法检测篮球
            ball = self.tracker.detect(frame, frame_idx, timestamp, config)
            if ball:
                self.ball_detected_count += 1
                if self.ball_detected_count % 50 == 0:
                    logger.debug(f"🏀 传统篮球检测: 位置({ball.cx:.0f}, {ball.cy:.0f}), 半径{ball.radius:.0f}")

        # 判断进球（冷却时间内不重复触发）
        if timestamp - self.last_goal_time < config.goal_cooldown:
            return False

        # 只有当篮筐检测到后才进行轨迹分析（篮球不是必须每帧都检测到）
        if not self.hoop_detected:
            return False

        # 检查是否有足够的轨迹点
        positions = self.tracker.get_recent_positions(20)
        if len(positions) < config.trajectory_min_frames:
            return False

        trajectory_ok = self._check_trajectory_through_hoop()

        # 只要轨迹检测OK就认为进球
        if trajectory_ok:
            # 记录进球时间
            self.last_goal_time = timestamp
            # 清空轨迹历史，避免重复检测
            self.tracker.history = []
            # 清空球网变化历史
            self.net_change_history = []
            logger.info(f"✅ 进球检测成功！时间戳 {timestamp:.1f}s")
            return True

        return False

    def _compute_light_level(self, gray: np.ndarray) -> float:
        """计算图像的光线强度水平"""
        return np.mean(gray)
    
    def _adaptive_light_compensation(self, gray: np.ndarray) -> np.ndarray:
        """自适应光线补偿，减少光线变化影响"""
        if not self.config.net_light_compensation:
            return gray
        
        current_light = self._compute_light_level(gray)
        self.light_level_history.append(current_light)
        
        if len(self.light_level_history) > self.max_light_history:
            self.light_level_history.pop(0)
        
        if len(self.light_level_history) < 3:
            return gray
        
        # 计算历史平均光线水平
        avg_light = np.mean(self.light_level_history[:-1])
        if avg_light == 0:
            return gray
        
        # 补偿因子，避免过度补偿
        light_ratio = avg_light / current_light
        light_ratio = np.clip(light_ratio, 0.7, 1.4)
        
        # 应用光线补偿
        compensated = cv2.convertScaleAbs(gray, alpha=light_ratio, beta=0)
        return compensated
    
    def _extract_net_features(self, roi: np.ndarray):
        """从球网区域提取多种特征"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 光线补偿
        compensated = self._adaptive_light_compensation(blurred)
        
        # 边缘检测
        edges = cv2.Canny(compensated, 30, 80)
        
        # 计算直方图
        hist = cv2.calcHist([compensated], [0], None, [self.config.net_histogram_bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        return compensated, edges, hist
    
    def _update_net_frame(self, frame: np.ndarray):
        if self.hoop.net_region:
            x, y, w, h = self.hoop.net_region
            fh, fw = frame.shape[:2]
            x = max(0, x); y = max(0, y)
            x2 = min(fw, x + w); y2 = min(fh, y + h)
            roi = frame[y:y2, x:x2]
            if roi.size > 0:
                # 提取多种特征
                gray, edges, hist = self._extract_net_features(roi)
                self.prev_net_frame = gray
                self.prev_net_edge = edges
                self.prev_net_hist = hist

    def _detect_net_change(self, frame: np.ndarray) -> float:
        """检测球网区域的像素变化（改进版）
        结合强度变化、边缘变化和直方图变化
        """
        if not self.hoop.net_region or self.prev_net_frame is None:
            return 0.0
        
        x, y, w, h = self.hoop.net_region
        fh, fw = frame.shape[:2]
        x = max(0, x); y = max(0, y)
        x2 = min(fw, x + w); y2 = min(fh, y + h)
        roi = frame[y:y2, x:x2]
        
        if roi.size == 0:
            return 0.0
        
        # 提取当前帧的特征
        current_gray, current_edges, current_hist = self._extract_net_features(roi)
        
        # 检查历史帧和当前帧的大小是否匹配
        if (self.prev_net_frame.shape != current_gray.shape or 
            self.prev_net_edge.shape != current_edges.shape):
            # 大小不匹配，更新历史帧并返回0
            self.prev_net_frame = current_gray
            self.prev_net_edge = current_edges
            self.prev_net_hist = current_hist
            return 0.0
        
        # 1. 强度变化
        diff_intensity = cv2.absdiff(self.prev_net_frame, current_gray)
        _, thresh_intensity = cv2.threshold(diff_intensity, self.config.net_motion_threshold, 255, cv2.THRESH_BINARY)
        change_intensity = np.sum(thresh_intensity) / (thresh_intensity.size * 255)
        
        # 2. 边缘变化
        diff_edges = cv2.absdiff(self.prev_net_edge, current_edges)
        change_edges = np.sum(diff_edges) / (diff_edges.size * 255)
        
        # 3. 直方图变化（卡方距离）
        hist_diff = cv2.compareHist(self.prev_net_hist, current_hist, cv2.HISTCMP_CHISQR)
        max_hist_diff = 2.0  # 经验最大差异值
        change_hist = min(hist_diff / max_hist_diff, 1.0)
        
        # 综合变化值（加权平均）
        total_change = (
            self.config.net_intensity_weight * change_intensity +
            self.config.net_edge_weight * change_edges +
            0.3 * change_hist  # 直方图变化权重
        )
        
        # 更新历史特征
        self.prev_net_frame = current_gray
        self.prev_net_edge = current_edges
        self.prev_net_hist = current_hist
        
        # 记录变化历史
        self.net_change_history.append(total_change)
        if len(self.net_change_history) > self.net_change_window:
            self.net_change_history.pop(0)
        
        return total_change

    def _is_net_change_significant(self) -> bool:
        """判断球网变化是否显著（改进版）
        考虑历史变化趋势、峰值检测和变化持续性
        """
        if len(self.net_change_history) < 4:
            return False
        
        # 获取最近的变化值
        recent_changes = self.net_change_history[-5:] if len(self.net_change_history) >= 5 else self.net_change_history
        
        # 计算统计指标
        avg_change = np.mean(self.net_change_history[:-1])  # 排除当前帧
        std_change = np.std(self.net_change_history[:-1])
        current_change = self.net_change_history[-1]
        prev_change = self.net_change_history[-2]
        
        # 条件1：当前变化显著大于平均水平
        threshold = max(
            self.config.net_change_threshold,
            avg_change + 1.5 * std_change
        )
        
        # 条件2：变化呈上升趋势或出现峰值
        is_rising = current_change > prev_change * 1.2
        is_peak = (current_change > avg_change * 2 and 
                   current_change > self.net_stability_threshold)
        
        # 条件3：变化在合理范围内，避免因光线突变导致的误检
        is_reasonable = current_change < 0.6
        
        # 条件4：检查最近几帧是否有持续的变化
        has_sustained_change = sum(1 for c in recent_changes if c > self.config.net_change_threshold * 0.5) >= 2
        
        # 综合判断
        significant = (
            current_change > threshold and
            (is_rising or is_peak) and
            is_reasonable and
            has_sustained_change
        )
        
        if significant:
            logger.debug(f"球网变化显著: 当前={current_change:.3f}, 平均={avg_change:.3f}, 阈值={threshold:.3f}")
        
        return significant

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
