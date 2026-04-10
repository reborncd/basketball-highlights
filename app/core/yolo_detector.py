"""
YOLO目标检测模块
使用Ultralytics YOLOv11进行篮球和篮筐检测
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


@dataclass
class YOLOSportsDetector:
    """YOLO运动目标检测器"""
    model_path: str = "yolo11n.pt"  # 默认使用YOLO11 Nano模型
    conf_thres: float = 0.25  # 置信度阈值
    iou_thres: float = 0.45   # IoU阈值
    imgsz: int = 640         # 推理分辨率
    device: str = "auto"     # 设备选择 (auto, cpu, cuda, mps)
    
    def __post_init__(self):
        """初始化YOLO模型"""
        try:
            # 检测设备可用性
            import torch
            if torch.cuda.is_available():
                device = self.device
            else:
                # 没有CUDA时使用CPU
                device = "cpu"
                logger.info("⚠️ CUDA不可用，使用CPU")
            
            self.model = YOLO(self.model_path)
            # 显式设置设备
            self.device = device
            logger.info(f"✅ YOLO模型加载成功: {self.model_path}, 设备: {device}")
        except Exception as e:
            logger.error(f"❌ YOLO模型加载失败: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """检测帧中的篮球和篮筐
        
        Args:
            frame: 输入帧
            
        Returns:
            检测结果列表，每个元素包含: 
            {"class": int, "conf": float, "bbox": [x1, y1, x2, y2]}
        """
        if self.model is None:
            return []
        
        try:
            # 确保使用正确的设备
            results = self.model.predict(
                frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device="cpu" if self.device == "auto" else self.device,
                verbose=False
            )
            
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    detections.append({
                        "class": cls,
                        "conf": conf,
                        "bbox": bbox
                    })
            
            return detections
        except Exception as e:
            logger.error(f"❌ YOLO检测失败: {e}")
            return []
    
    def detect_ball(self, frame: np.ndarray) -> Optional[dict]:
        """检测篮球
        
        Args:
            frame: 输入帧
            
        Returns:
            篮球检测结果，包含: 
            {"conf": float, "bbox": [x1, y1, x2, y2], "center": [cx, cy]}
        """
        detections = self.detect(frame)
        
        # 假设篮球的类别ID为0
        ball_detections = [d for d in detections if d["class"] == 0]
        
        if not ball_detections:
            return None
        
        # 选择置信度最高的篮球
        best_ball = max(ball_detections, key=lambda x: x["conf"])
        
        # 计算中心点
        x1, y1, x2, y2 = best_ball["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        return {
            "conf": best_ball["conf"],
            "bbox": best_ball["bbox"],
            "center": [cx, cy]
        }
    
    def detect_hoop(self, frame: np.ndarray) -> Optional[dict]:
        """检测篮筐
        
        Args:
            frame: 输入帧
            
        Returns:
            篮筐检测结果，包含: 
            {"conf": float, "bbox": [x1, y1, x2, y2], "center": [cx, cy]}
        """
        detections = self.detect(frame)
        
        # 假设篮筐的类别ID为1
        hoop_detections = [d for d in detections if d["class"] == 1]
        
        if not hoop_detections:
            return None
        
        # 选择置信度最高的篮筐
        best_hoop = max(hoop_detections, key=lambda x: x["conf"])
        
        # 计算中心点
        x1, y1, x2, y2 = best_hoop["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        return {
            "conf": best_hoop["conf"],
            "bbox": best_hoop["bbox"],
            "center": [cx, cy]
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            detections: 检测结果列表
            
        Returns:
            绘制了检测结果的帧
        """
        result_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class"]
            conf = det["conf"]
            
            # 根据类别设置颜色
            if cls == 0:  # 篮球
                color = (0, 0, 255)  # 红色
                label = f"Ball: {conf:.2f}"
            elif cls == 1:  # 篮筐
                color = (0, 255, 0)  # 绿色
                label = f"Hoop: {conf:.2f}"
            else:
                color = (255, 0, 0)  # 蓝色
                label = f"Class {cls}: {conf:.2f}"
            
            # 绘制边界框
            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签
            cv2.putText(result_frame, label, (int(x1), int(y1) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_frame