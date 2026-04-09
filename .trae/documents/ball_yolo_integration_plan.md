# 借鉴 griftt/ball-yolo 改进进球检测方案

## 📊 仓库分析总结

### griftt/ball-yolo 的核心优势

#### 1. **YOLO深度学习目标检测**
- 使用Ultralytics YOLOv11进行篮球和篮筐检测
- 专门训练的模型，检测准确率高
- 支持Apple MPS和CoreML加速

#### 2. **三阶段区域判定法**
```
1. 高位区检测 (High Zone)
   └─> 球在篮筐上方
   
2. 触框检测 (Rim Touch)
   └─> 球与篮筐重叠
   
3. 进球区检测 (Goal Zone)
   └─> 球穿过篮筐下方
   
判定逻辑：
- 如果球先经过「高位区」或「触框」
- 然后在 2.5 秒内进入「进球区」
- 则触发进球事件
```

#### 3. **篮筐自动校准机制**
- 先采集30个篮筐检测样本
- 使用中位数计算稳定的篮筐位置
- 锁定篮筐后不再频繁检测

#### 4. **跳帧检测优化**
- 每3帧检测一次，速度提升3倍
- 使用`cap.grab()`快速跳过帧，不解码图像
- 仅在需要检测的帧才真正解码

#### 5. **时间窗口验证**
- 记录球在高位区/触框的时间点
- 检查后续2.5秒内球是否进入进球区
- 3秒冷却时间避免重复检测

---

## 🎯 我们可以借鉴的改进方案

### 方案一：集成YOLO检测（推荐）

#### 目标
引入YOLO深度学习模型替代传统的颜色检测方法。

#### 实施步骤

1. **添加依赖**
   - 新增 `ultralytics>=8.0.0` 到 requirements.txt
   - 新增 `torch>=2.0.0` 和 `torchvision>=0.15.0`

2. **创建YOLO检测模块**
   ```python
   # app/core/yolo_detector.py
   class YOLOSportsDetector:
       def __init__(self, model_path=None):
           self.model = YOLO(model_path or 'yolo11n.pt')
       
       def detect_ball(self, frame):
           # 检测篮球
           pass
       
       def detect_hoop(self, frame):
           # 检测篮筐
           pass
   ```

3. **修改DetectionConfig**
   - 添加YOLO模型路径配置
   - 添加置信度阈值配置
   - 添加检测类别配置

4. **集成到GoalDetector**
   - 保留现有传统方法作为备用
   - 优先使用YOLO检测
   - 提供切换选项

---

### 方案二：改进现有判定逻辑（低风险，快速见效）

#### 目标
在不引入新依赖的前提下，借鉴ball-yolo的判定逻辑改进现有方案。

#### 实施步骤

1. **实现三阶段区域判定**
   ```python
   # 在detector.py中
   class GoalDetector:
       def __init__(self, config):
           # ... 现有代码 ...
           self.last_high_zone_ts = -999.0  # 记录进入高位区的时间
           self.last_rim_touch_ts = -999.0  # 记录触框时间
           self.high_zone = None  # 高位区
           self.rim_zone = None   # 篮筐区
           self.goal_zone = None  # 进球区
           self.shot_window = 2.5  # 时间窗口2.5秒
   ```

2. **定义三个关键区域**
   - **高位区**：篮筐上方150px范围
   - **篮筐区**：篮筐边界
   - **进球区**：篮筐下方150px范围

3. **修改判定逻辑**
   ```python
   def process_frame(self, frame, frame_idx, timestamp):
       # 1. 检测篮球
       ball = self.tracker.detect(...)
       
       if ball:
           # 2. 检查是否在高位区
           if self._is_in_high_zone(ball):
               self.last_high_zone_ts = timestamp
           
           # 3. 检查是否触框
           if self._is_touching_rim(ball):
               self.last_rim_touch_ts = timestamp
           
           # 4. 检查是否在进球区
           if self._is_in_goal_zone(ball):
               # 检查时间窗口
               last_interaction = max(self.last_high_zone_ts, self.last_rim_touch_ts)
               time_diff = timestamp - last_interaction
               
               if 0.05 < time_diff < self.shot_window:
                   return True  # 进球！
       
       return False
   ```

4. **篮筐位置自动校准**
   - 采集多个篮筐检测结果
   - 使用中位数计算稳定位置
   - 锁定后减少检测频率

---

### 方案三：混合方案（最佳平衡）

#### 目标
结合方案一和方案二，提供最佳体验。

#### 实施步骤

1. **同时支持两种检测方式**
   - 传统方法（颜色检测）：快速，不需要额外依赖
   - YOLO方法：更准确，但需要模型

2. **用户可选择**
   - 在UI中添加检测模式切换
   - 自动检测是否有YOLO模型可用

3. **渐进式改进**
   - 先实施方案二，快速改进现有逻辑
   - 后续再集成YOLO，提升准确率

---

## 📋 推荐实施路径

### 阶段一：快速改进（1-2天）
**实施方案二：改进现有判定逻辑**

1. 实现三阶段区域判定
2. 添加时间窗口验证
3. 实现篮筐位置校准
4. 优化冷却时间逻辑

### 阶段二：可选增强（3-5天）
**集成YOLO检测**

1. 添加YOLO依赖
2. 创建YOLO检测模块
3. 提供预训练模型或训练指导
4. 添加检测模式切换UI

---

## 🎨 具体代码改进建议

### 1. 修改DetectionConfig
```python
@dataclass
class DetectionConfig:
    # ... 现有配置 ...
    
    # 新增：三阶段判定参数
    high_zone_offset: float = 150.0  # 高位区偏移(像素)
    goal_zone_offset: float = 150.0   # 进球区偏移(像素)
    shot_window: float = 2.5          # 时间窗口(秒)
    
    # 新增：篮筐校准参数
    calibration_samples: int = 30      # 校准样本数
    hoop_stability_threshold: int = 3  # 稳定阈值
```

### 2. 改进GoalDetector
```python
class GoalDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.hoop = HoopDetector()
        self.tracker = BallTracker()
        
        # 新增：三阶段判定状态
        self.last_high_zone_ts: float = -999.0
        self.last_rim_touch_ts: float = -999.0
        self.last_goal_time: float = -999.0
        
        # 新增：篮筐校准
        self.calibration_buffer: list = []
        self.is_calibrated: bool = False
        self.locked_hoop_rect: Optional[tuple] = None
        
        # 新增：区域定义
        self.high_zone: Optional[tuple] = None
        self.rim_zone: Optional[tuple] = None
        self.goal_zone: Optional[tuple] = None
```

### 3. 实现区域判定方法
```python
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
```

---

## ✅ 总结

### 立即可以做的改进（无需新依赖）

1. **实现三阶段区域判定**
   - 定义高位区、篮筐区、进球区
   - 记录球进入高位区/触框的时间
   - 检查时间窗口内是否进入进球区

2. **篮筐位置校准**
   - 采集多个篮筐检测样本
   - 使用中位数稳定位置
   - 锁定后减少检测频率

3. **时间窗口验证**
   - 2.5秒时间窗口
   - 3秒冷却时间

### 长期改进

1. **集成YOLO检测**
   - 更高的检测准确率
   - 更好的鲁棒性
   - 支持更多场景

2. **数据收集和模型训练**
   - 收集我们自己的篮球视频数据
   - 训练专用的YOLO模型
   - 持续优化检测效果

---

**建议先实施方案二（改进现有判定逻辑），快速看到改进效果，后续再考虑集成YOLO。**