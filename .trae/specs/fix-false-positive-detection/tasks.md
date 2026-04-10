# 篮球进球检测优化 - 实现计划

## [x] Task 1: 优化检测配置参数
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 调整冷却时间从3秒增加到5秒，更彻底地防止重复检测
  - 优化三阶段区域的大小，减少误检区域
  - 调整时间窗口参数，使其更严格
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-1.1: 验证冷却时间配置正确应用
  - `human-judgement` TR-1.2: 验证新参数不会过度降低检测灵敏度
- **Notes**: 这是快速见效的改进，可以立即实施

## [x] Task 2: 实现完整的进球状态机
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 定义明确的状态：WAITING（等待）→ IN_HIGH_ZONE（高位区）→ TOUCHING_RIM（触框）→ IN_GOAL_ZONE（进球区）→ CONFIRMED（确认进球）
  - 实现状态转换逻辑，只有按正确顺序通过状态才可能进球
  - 添加状态超时机制，避免状态卡住
- **Acceptance Criteria Addressed**: AC-1, AC-5
- **Test Requirements**:
  - `programmatic` TR-2.1: 验证状态机能够正确跟踪状态转换
  - `programmatic` TR-2.2: 验证状态超时机制工作正常
  - `human-judgement` TR-2.3: 验证只有正确顺序的状态转换才会触发进球
- **Notes**: 这是核心改进，将显著降低假阳性

## [x] Task 3: 添加轨迹连续性验证
- **Priority**: P0
- **Depends On**: Task 2
- **Description**: 
  - 检查最近10-15帧的篮球轨迹
  - 验证轨迹是从高位区向下移动到进球区的
  - 计算轨迹的下降趋势，确保是真正的投篮
  - 验证篮球位置在横向范围内（不偏离篮筐太远）
- **Acceptance Criteria Addressed**: AC-2, AC-5
- **Test Requirements**:
  - `programmatic` TR-3.1: 验证轨迹下降趋势检测逻辑
  - `programmatic` TR-3.2: 验证横向范围检查
  - `human-judgement` TR-3.3: 验证只有连续向下的轨迹才会被接受
- **Notes**: 这将显著减少假阳性，特别是篮球在篮筐附近晃动的情况

## [x] Task 4: 优化进球确认机制
- **Priority**: P1
- **Depends On**: Task 3
- **Description**: 
  - 要求篮球在进球区内停留至少2-3帧才确认进球
  - 添加多重条件验证：位置、轨迹、时间窗口都满足
  - 实现更严格的冷却逻辑，检测到进球后立即重置所有状态
- **Acceptance Criteria Addressed**: AC-3, AC-5
- **Test Requirements**:
  - `programmatic` TR-4.1: 验证多帧停留要求
  - `programmatic` TR-4.2: 验证冷却逻辑正确重置状态
  - `human-judgement` TR-4.3: 验证没有重复检测
- **Notes**: 进一步巩固检测的可靠性

## [x] Task 5: 增加详细的调试日志
- **Priority**: P1
- **Depends On**: Task 4
- **Description**: 
  - 在每个状态转换时记录详细日志
  - 记录每次检测决策的原因（为什么接受或拒绝）
  - 记录轨迹分析的关键数据
  - 添加日志级别控制，便于调试
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `human-judgement` TR-5.1: 验证日志包含足够的调试信息
  - `human-judgement` TR-5.2: 验证日志格式清晰易读
- **Notes**: 便于后续调试和优化

## [x] Task 6: 测试和验证改进效果
- **Priority**: P2
- **Depends On**: Task 5
- **Description**: 
  - 使用多个测试视频验证改进效果
  - 比较改进前后的检测结果
  - 收集假阳性率和召回率数据
  - 根据测试结果微调参数
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `human-judgement` TR-6.1: 验证假阳性率明显降低
  - `human-judgement` TR-6.2: 验证没有重复检测
  - `human-judgement` TR-6.3: 验证真实进球仍然能被检测到
- **Notes**: 最终验证改进是否达到预期目标
