# EVSP-ALNS集成RL完整迁移指南

## 目录
1. [系统对应关系分析](#系统对应关系分析)
2. [核心替换：WeightsManagement → Actor](#核心替换)
3. [EVSP图表示设计](#EVSP图表示设计)
4. [完整迁移代码](#完整迁移代码)
5. [迁移步骤](#迁移步骤)
6. [预期效果](#预期效果)

---

## 系统对应关系分析

### 架构对比

| 组件 | VRP-ALNS-RL | 你的EVSP-ALNS | 关系 |
|------|-------------|---------------|------|
| **问题模型** | `VRPData` | `ScheduleClass` | 完全对应 |
| **子解** | Route | `DutyClass` | 完全对应 |
| **销毁操作符** | 5个destroy | 3个remove | 对应 |
| **修复操作符** | 4个repair | 2个insert | 对应 |
| **操作符选择** | **Actor网络** | **WeightsManagement** | 🔥核心替换点 |
| **价值评估** | **Critic网络** | 无 | RL新增 |
| **接受准则** | 奖励机制 | 模拟退火 | 可共存 |

### 核心发现

```
┌─────────────────────────────────────────────────────────┐
│  关键洞察：                                              │
│                                                         │
│  你的EVSP-ALNS已经有完整的ALNS框架！                     │
│  只需要用RL的Actor网络替换WeightsManagement             │
│  其他所有代码（操作符、数据结构）都保持不变！             │
└─────────────────────────────────────────────────────────┘
```

---

## 核心替换

### 当前机制：WeightsManagement（人工自适应）

```python
# ALNS/WeightsManagement.py 当前逻辑

class WeightsManagement:
    def __init__(self):
        # 初始权重相等
        self.removeWeights = [1.0, 1.0, 1.0]  # 3个remove操作符
        self.insertWeights = [1.0, 1.0]       # 2个insert操作符

    def selectRemove(self):
        """轮盘赌选择销毁操作符"""
        total = sum(self.removeWeights)
        probs = [w/total for w in self.removeWeights]
        return np.random.choice([0, 1, 2], p=probs)

    def selectInsert(self):
        """轮盘赌选择修复操作符"""
        total = sum(self.insertWeights)
        probs = [w/total for w in self.insertWeights]
        return np.random.choice([0, 1], p=probs)

    def updateWeights(self, removeIdx, insertIdx, score):
        """根据得分更新权重（每100次迭代）"""
        # 累积得分
        self.removeScores[removeIdx] += score
        self.insertScores[insertIdx] += score

        # 每100次更新权重
        if iteration % 100 == 0:
            # w_new = w_old * (1-r) + r * (π/θ)
            self.removeWeights[i] = (1-r) * w_old + r * score/count
```

**问题**：
- 手工设计的权重更新规则
- 只考虑历史平均，不考虑当前状态
- 无法学习状态-操作符的映射关系

---

### 新机制：Actor网络（RL学习）

```python
# 用RL替换后的逻辑

class RLOperatorSelector:
    def __init__(self, actor_model, graph_converter):
        self.actor = actor_model
        self.converter = graph_converter

    def selectRemove(self, schedule):
        """RL选择销毁操作符"""
        # 1. 将Schedule转换为图
        graph = self.converter.schedule_to_graph(schedule)

        # 2. Actor网络选择（考虑当前状态）
        mask = torch.tensor([True, True, True, False, False])  # 只允许remove
        action, log_prob = self.actor(graph, mask)

        # 3. 返回remove索引（0, 1, 2）
        return action.item(), log_prob

    def selectInsert(self, schedule):
        """RL选择修复操作符"""
        graph = self.converter.schedule_to_graph(schedule)

        mask = torch.tensor([False, False, False, True, True])  # 只允许insert
        action, log_prob = self.actor(graph, mask)

        # 返回insert索引（0, 1）
        insert_idx = action.item() - 3
        return insert_idx, log_prob

    # 不需要updateWeights！Actor通过反向传播自动学习
```

**优势**：
- ✅ 根据当前Schedule状态选择操作符
- ✅ 学习复杂的状态-操作符映射
- ✅ 自动适应，无需手工调参
- ✅ 泛化能力强

---

## EVSP图表示设计

### 挑战：EVSP vs VRP

| 特征 | VRP | EVSP |
|------|-----|------|
| 元素 | 客户 | 任务（trip） |
| 连接 | 路线（route） | 调度（duty） |
| 约束 | 容量 | 电量、充电站容量、时间窗 |
| 特殊节点 | depot | 起点(o)、终点(d)、充电站(f) |

### 图结构设计

```python
节点类型：
  1. 任务节点（T1, T2, ..., Tn）
  2. 充电节点（F1, F2, ..., Fm）
  3. 起点节点（o）
  4. 终点节点（d）
  5. 中心节点（center）- 虚拟节点，聚合全局信息

边连接：
  - Duty内的相邻节点 ↔ 双向边
  - 所有节点 ↔ 中心节点（双向边）
```

### 节点特征设计（推荐12维）

```python
features = [
    # === 结构特征 (3维) ===
    position_in_duty,        # 在调度中的位置（0表示起点，-1表示未分配）
    duty_id,                 # 所属调度ID（-1表示未分配）
    node_type,               # 节点类型（0:任务, 1:充电, 2:起点, 3:终点, 4:中心）

    # === 时间特征 (2维) ===
    start_time_normalized,   # 归一化的开始时间（0-1）
    travel_time_normalized,  # 归一化的行驶时间（0-1）

    # === 能量特征 (3维) ===
    energy_consumption,      # 能耗需求（任务）或充电量（充电节点）
    battery_level,           # 当前电池电量（仅对已分配任务有意义）
    is_charging,             # 是否为充电节点（0/1）

    # === 空间特征 (2维) ===
    distance_to_center,      # 到图中心的距离（基于时间）
    time_similarity,         # 与其他任务的时间相似度（平均值）

    # === 全局特征 (2维) ===
    duty_load_ratio,         # 该duty的负载率（任务数/平均任务数）
    charging_urgency,        # 充电紧急度（电量/容量的倒数）
]
```

**特征设计原则**：
1. 归一化所有特征到 [0, 1] 或 [-1, 1]
2. 包含局部信息（duty内位置）和全局信息（中心距离）
3. 体现EVSP特有的能量和时间约束
4. 适度的维度（12维），不要过大

---

## 完整迁移代码

### 步骤1：创建图转换器

```python
# ALNS/GraphConverter.py (新建文件)

import torch
import numpy as np
from torch_geometric.data import Data

class EVSPGraphConverter:
    """将EVSP的Schedule转换为图表示"""

    def __init__(self, evsp):
        """
        参数：
            evsp: EVSP对象（包含问题数据）
        """
        self.evsp = evsp
        self.num_tasks = len(evsp.timetable)

        # 预计算一些全局统计量
        self.max_start_time = max(t['StartTimeMin'] for t in evsp.timetable.values())
        self.max_travel_time = max(t['TravelTimeMin'] for t in evsp.timetable.values())
        self.max_consumption = max(t['Consumption'] for t in evsp.timetable.values())

    def schedule_to_graph(self, schedule):
        """
        将Schedule对象转换为PyG Data对象

        参数：
            schedule: ScheduleClass对象

        返回：
            graph: PyG Data对象
        """
        # 1. 提取节点特征
        features = []
        node_to_id = {}  # 映射：(duty_idx, node) → node_id
        node_id = 0

        # 1a. 为每个duty中的节点创建特征
        for duty_idx, duty in enumerate(schedule.schedule):
            for pos, node in enumerate(duty.S):
                feature = self._extract_node_feature(duty, duty_idx, pos, node)
                features.append(feature)
                node_to_id[(duty_idx, node)] = node_id
                node_id += 1

        # 1b. 添加中心节点
        center_feature = self._extract_center_feature(schedule)
        features.append(center_feature)
        center_node_id = node_id

        # 2. 构建边
        edges = []

        # 2a. Duty内的边
        for duty_idx, duty in enumerate(schedule.schedule):
            for i in range(len(duty.S) - 1):
                node1_id = node_to_id[(duty_idx, duty.S[i])]
                node2_id = node_to_id[(duty_idx, duty.S[i+1])]
                edges.append([node1_id, node2_id])
                edges.append([node2_id, node1_id])  # 双向

        # 2b. 所有节点到中心节点的边
        for nid in range(center_node_id):
            edges.append([nid, center_node_id])
            edges.append([center_node_id, nid])

        # 3. 构建PyG Data对象
        graph = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            center_node_index=torch.tensor([center_node_id]),
            state=schedule,  # 保存原始Schedule对象
            cost=torch.tensor([schedule.cost])
        )

        return graph

    def _extract_node_feature(self, duty, duty_idx, pos, node):
        """提取单个节点的特征（12维）"""
        feature = []

        # === 结构特征 (3维) ===
        feature.append(pos)  # 位置
        feature.append(duty_idx)  # duty ID
        feature.append(self._get_node_type(node))  # 节点类型

        # === 时间特征 (2维) ===
        if node in self.evsp.timetable:
            # 任务节点
            start_time = self.evsp.timetable[node]['StartTimeMin']
            travel_time = self.evsp.timetable[node]['TravelTimeMin']
        elif isinstance(node, str) and node.startswith('f'):
            # 充电节点
            start_time = self._get_charging_start_time(duty, pos)
            travel_time = self._get_charging_duration(duty, node)
        else:
            # o 或 d
            start_time = 0
            travel_time = 0

        feature.append(start_time / self.max_start_time if self.max_start_time > 0 else 0)
        feature.append(travel_time / self.max_travel_time if self.max_travel_time > 0 else 0)

        # === 能量特征 (3维) ===
        if node in self.evsp.timetable:
            consumption = self.evsp.timetable[node]['Consumption']
            battery_level = self._estimate_battery_level(duty, pos)
            is_charging = 0
        elif isinstance(node, str) and node.startswith('f'):
            consumption = 0  # 充电节点
            battery_level = self._estimate_battery_level(duty, pos)
            is_charging = 1
        else:
            consumption = 0
            battery_level = 0
            is_charging = 0

        feature.append(consumption / self.max_consumption if self.max_consumption > 0 else 0)
        feature.append(battery_level)
        feature.append(is_charging)

        # === 空间特征 (2维) ===
        distance_to_center = self._calculate_time_distance_to_center(node)
        time_similarity = self._calculate_time_similarity(node)

        feature.append(distance_to_center)
        feature.append(time_similarity)

        # === 全局特征 (2维) ===
        duty_load = len([n for n in duty.S if n not in ['o', 'd'] and not (isinstance(n, str) and n.startswith('f'))])
        avg_load = sum(len([n for n in d.S if n not in ['o', 'd'] and not (isinstance(n, str) and n.startswith('f'))]) for d in schedule.schedule) / len(schedule.schedule) if schedule.schedule else 1
        duty_load_ratio = duty_load / avg_load if avg_load > 0 else 1

        charging_urgency = 1 - battery_level if battery_level > 0 else 0

        feature.append(duty_load_ratio)
        feature.append(charging_urgency)

        return feature

    def _extract_center_feature(self, schedule):
        """提取中心节点特征（全局统计）"""
        return [
            -1, -1, 4,  # position=-1, duty=-1, type=center
            0.5, 0.5,   # 平均时间特征
            0.5, 0.5, 0,  # 平均能量特征
            0, 0.5,     # 中心空间特征
            1.0, 0.5    # 全局特征
        ]

    def _get_node_type(self, node):
        """确定节点类型"""
        if node == 'o':
            return 2
        elif node == 'd':
            return 3
        elif isinstance(node, str) and node.startswith('f'):
            return 1
        else:
            return 0

    def _get_charging_start_time(self, duty, pos):
        """获取充电节点的开始时间"""
        # 简化：使用前一个任务的结束时间
        if pos > 0:
            prev_node = duty.S[pos-1]
            if prev_node in self.evsp.timetable:
                return self.evsp.timetable[prev_node]['StartTimeMin'] + \
                       self.evsp.timetable[prev_node]['TravelTimeMin']
        return 0

    def _get_charging_duration(self, duty, charging_node):
        """获取充电时长"""
        if charging_node in duty.R:
            # 从R中提取充电时间槽数
            charging_slot = duty.R[charging_node]  # 例如 'r15'
            if charging_slot.startswith('r'):
                return int(charging_slot[1:])
        return 0

    def _estimate_battery_level(self, duty, pos):
        """估算当前电池电量（简化版本）"""
        # 这里需要根据你的Duty类的逻辑来实现
        # 简化版本：基于位置估算
        if pos == 0:
            return 1.0  # 起点满电
        # 更复杂的版本需要模拟能量变化
        return 0.5  # 占位

    def _calculate_time_distance_to_center(self, node):
        """计算到时间中心的距离"""
        if node not in self.evsp.timetable:
            return 0.5
        start_time = self.evsp.timetable[node]['StartTimeMin']
        center_time = self.max_start_time / 2
        return abs(start_time - center_time) / center_time if center_time > 0 else 0

    def _calculate_time_similarity(self, node):
        """计算与其他任务的时间相似度"""
        if node not in self.evsp.timetable:
            return 0.5
        # 简化：返回归一化的时间窗口位置
        return self.evsp.timetable[node]['StartTimeMin'] / self.max_start_time if self.max_start_time > 0 else 0.5

    def update_graph(self, graph):
        """
        在操作符执行后更新图表示
        （Schedule已经被修改，需要同步到Graph）
        """
        schedule = graph.state
        new_graph = self.schedule_to_graph(schedule)
        return new_graph
```

---

### 步骤2：修改ALNS主循环

```python
# ALNS/ALNS_RL.py (新建文件，基于原ALNS.py修改)

import torch
from torch_geometric.data import Data
from ALNS.GraphConverter import EVSPGraphConverter
from actor import actor
from critic import critic

class ALNS_RL:
    """集成RL的ALNS求解器"""

    def __init__(self, evsp, use_rl=True):
        self.evsp = evsp
        self.use_rl = use_rl

        # 原ALNS参数
        self.iterMax = 15000
        self.nMax, self.nMin = 10, 1
        self.T0 = 100
        self.alpha = 0.9997
        self.enePenalty = 700
        self.capPenalty = 700

        # 操作符列表（保持原有实现）
        from ALNS.RemoveOperators import randomRemoval, timeRelatedRemoval, neighborRemoval
        from ALNS.InsertOperators import randomInsert, greedyInsert

        self.removeOps = [randomRemoval, timeRelatedRemoval, neighborRemoval]
        self.insertOps = [randomInsert, greedyInsert]

        # RL组件（新增）
        if use_rl:
            self.graph_converter = EVSPGraphConverter(evsp)

            # 创建Actor和Critic网络
            feature_dim = 12  # 根据特征设计
            num_actions = len(self.removeOps) + len(self.insertOps)  # 3 + 2 = 5

            self.actor = actor(
                num_in_features=feature_dim,
                num_embedding=128,
                num_GNNs=2,
                num_out_layers=3,
                num_actions=num_actions
            )

            self.critic = critic(
                num_in_features=feature_dim,
                num_embedding=128,
                num_GNNs=2,
                num_out_layers=3
            )

            # 优化器
            self.optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=1e-5
            )

            # 训练记录
            self.log_probs = []
            self.state_values = []
            self.rewards = []

        else:
            # 使用原有的权重管理
            from ALNS.WeightsManagement import WeightsManagement
            self.weights = WeightsManagement(...)

    def solve(self):
        """主求解循环"""
        from ALNS.InitialSolution import initialize

        # 1. 获取初始解
        iniSchedule = initialize(self.evsp)
        self.bestSchedule = iniSchedule
        self.currentSchedule = iniSchedule
        bestCost = iniSchedule.cost

        T = self.T0

        for iteration in range(self.iterMax):
            # 2. 选择操作符
            if self.use_rl:
                removeIdx, insertIdx, log_prob_r, log_prob_i, value = self._select_operators_rl()
            else:
                removeIdx, insertIdx = self._select_operators_traditional()

            # 3. 执行操作符（保持原逻辑）
            n = random.randint(self.nMin, self.nMax)

            # Remove
            tempSchedule = copy.deepcopy(self.currentSchedule)
            tempSchedule, removedTrips = self.removeOps[removeIdx](tempSchedule, n)

            # Insert
            tempSchedule = self.insertOps[insertIdx](self.evsp, tempSchedule, removedTrips)

            # 4. 接受准则（模拟退火）
            newCost = tempSchedule.cost
            accept = False

            if newCost < bestCost:
                # 找到更优解
                self.bestSchedule = copy.deepcopy(tempSchedule)
                bestCost = newCost
                accept = True
                score = 33  # 用于传统权重更新
            elif newCost < self.currentSchedule.cost:
                # 比当前解好
                accept = True
                score = 9
            elif random.random() < math.exp((self.currentSchedule.cost - newCost) / T):
                # 模拟退火接受
                accept = True
                score = 3
            else:
                score = 0

            # 5. 更新解
            if accept:
                self.currentSchedule = tempSchedule

            # 6. 更新操作符选择机制
            if self.use_rl:
                # RL：记录奖励
                reward = max(0, self.currentSchedule.cost - newCost)  # 成本改进
                self.rewards.append(reward)

                # 每100次迭代更新RL网络
                if iteration % 100 == 0 and iteration > 0:
                    self._update_rl()

            else:
                # 传统：更新权重
                self.weights.updateWeights(removeIdx, insertIdx, score)

            # 7. 冷却
            T *= self.alpha

        return self.bestSchedule

    def _select_operators_rl(self):
        """使用RL选择操作符"""
        # 1. 转换为图
        graph = self.graph_converter.schedule_to_graph(self.currentSchedule)

        # 2. 选择remove操作符
        mask_remove = torch.tensor([True, True, True, False, False])
        action_r, log_prob_r = self.actor(graph, mask_remove)
        removeIdx = action_r.item()

        # 3. 选择insert操作符
        mask_insert = torch.tensor([False, False, False, True, True])
        action_i, log_prob_i = self.actor(graph, mask_insert)
        insertIdx = action_i.item() - 3  # 索引从3开始，调整为0,1

        # 4. 价值估计
        value = self.critic(graph)

        # 5. 记录轨迹
        self.log_probs.extend([log_prob_r, log_prob_i])
        self.state_values.extend([value, value])

        return removeIdx, insertIdx, log_prob_r, log_prob_i, value

    def _select_operators_traditional(self):
        """使用传统权重管理选择操作符"""
        removeIdx = self.weights.selectRemove()
        insertIdx = self.weights.selectInsert()
        return removeIdx, insertIdx

    def _update_rl(self):
        """更新RL网络"""
        if len(self.rewards) == 0:
            return

        # 计算returns（rewards-to-go）
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)

        # 转换为tensor
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.log_probs)
        state_values = torch.stack(self.state_values)

        # 计算优势
        advantages = returns - state_values.detach()

        # 损失
        actor_loss = -(advantages * log_probs).mean()
        critic_loss = ((returns - state_values) ** 2).mean()

        total_loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5)
        self.optimizer.step()

        # 清空轨迹
        self.log_probs = []
        self.state_values = []
        self.rewards = []

    def save_model(self, path):
        """保存训练好的模型"""
        if self.use_rl:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict()
            }, path)

    def load_model(self, path):
        """加载训练好的模型"""
        if self.use_rl:
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
```

---

### 步骤3：训练脚本

```python
# train_evsp_rl.py (新建文件)

from EVSPModel.EVSPClass import EVSP
from ALNS.ALNS_RL import ALNS_RL
import pandas as pd

# 1. 加载数据
timetable = pd.read_excel('Data/T100.xlsx', index_col=0).T.to_dict()

# 2. 创建EVSP实例
evsp = EVSP(timetable)
evsp.setVehTypes(E_k=[200, 250])  # 设置车型
evsp.setCosts(c_k=[500, 600], c_e=0.7, c_t=0.5)
evsp.setChargingFunc(func='linear', alpha=2.0, beta=1.0)
evsp.createModel()

# 3. 训练RL-ALNS
alns_rl = ALNS_RL(evsp, use_rl=True)

# 运行多个epoch
for epoch in range(10):  # 10个epoch
    print(f"\nEpoch {epoch+1}/10")
    best_schedule = alns_rl.solve()
    print(f"Best cost: {best_schedule.cost:.2f}")

    # 保存模型
    if epoch % 2 == 0:
        alns_rl.save_model(f'checkpoints/evsp_epoch_{epoch+1}.pth')

# 4. 保存最终模型
alns_rl.save_model('checkpoints/evsp_final.pth')

# 5. 可视化
best_schedule.plotTimetable()
best_schedule.costBar()
```

---

### 步骤4：测试脚本

```python
# test_evsp_rl.py (新建文件)

from EVSPModel.EVSPClass import EVSP
from ALNS.ALNS_RL import ALNS_RL
import pandas as pd

# 1. 加载数据
timetable = pd.read_excel('Data/T100.xlsx', index_col=0).T.to_dict()

# 2. 创建EVSP实例
evsp = EVSP(timetable)
evsp.setVehTypes(E_k=[200, 250])
evsp.setCosts(c_k=[500, 600], c_e=0.7, c_t=0.5)
evsp.setChargingFunc(func='linear', alpha=2.0, beta=1.0)
evsp.createModel()

# 3. 使用训练好的模型
alns_rl = ALNS_RL(evsp, use_rl=True)
alns_rl.load_model('checkpoints/evsp_final.pth')
alns_rl.actor.eval()  # 设置为评估模式

# 4. 求解（使用学到的策略）
best_schedule = alns_rl.solve()

# 5. 输出结果
print(f"Best cost: {best_schedule.cost:.2f}")
print(f"Number of vehicles: {len(best_schedule.schedule)}")
print(f"Vehicle cost: {best_schedule.vehicleCost:.2f}")
print(f"Charging cost: {best_schedule.chargingCost:.2f}")
print(f"Time cost: {best_schedule.timeCost:.2f}")

# 6. 可视化
best_schedule.plotTimetable()
best_schedule.costBar()
best_schedule.chargingPowerPlot()
```

---

## 迁移步骤

### 第1周：准备工作

**任务1.1**：复制RL组件
```bash
cd alns-framework-for-evsp/
mkdir RL_Components
cp path/to/GNN.py RL_Components/
cp path/to/actor.py RL_Components/
cp path/to/critic.py RL_Components/
```

**任务1.2**：测试现有ALNS
```python
# 确保原ALNS能正常运行
alns = ALNS(evsp)
alns.solve()
```

### 第2周：实现图转换

**任务2.1**：实现GraphConverter（上面的代码）

**任务2.2**：测试图转换
```python
# 测试：Schedule → Graph → Schedule 一致性
converter = EVSPGraphConverter(evsp)

schedule1 = initialize(evsp)
graph = converter.schedule_to_graph(schedule1)
schedule2 = graph.state

# 验证
assert schedule1.cost == schedule2.cost
assert len(schedule1.schedule) == len(schedule2.schedule)
```

**任务2.3**：可视化图结构
```python
import networkx as nx
import matplotlib.pyplot as plt

# 将PyG图转换为NetworkX图
G = nx.Graph()
edge_index = graph.edge_index.numpy()
G.add_edges_from(edge_index.T)

# 绘制
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.savefig('evsp_graph_structure.png')
```

### 第3周：集成RL

**任务3.1**：实现ALNS_RL类（上面的代码）

**任务3.2**：小规模测试
```python
# 使用T20.xlsx小数据集测试
alns_rl = ALNS_RL(evsp_small, use_rl=True)
alns_rl.iterMax = 1000  # 减少迭代次数
best = alns_rl.solve()
```

**任务3.3**：对比传统方法
```python
# 传统ALNS
alns_traditional = ALNS_RL(evsp, use_rl=False)
cost_traditional = alns_traditional.solve().cost

# RL-ALNS
alns_rl = ALNS_RL(evsp, use_rl=True)
cost_rl = alns_rl.solve().cost

print(f"Traditional: {cost_traditional:.2f}")
print(f"RL: {cost_rl:.2f}")
print(f"Improvement: {(cost_traditional - cost_rl) / cost_traditional * 100:.2f}%")
```

### 第4周：训练与调优

**任务4.1**：多epoch训练
```python
# 运行train_evsp_rl.py
# 监控训练过程
```

**任务4.2**：超参数调优
```python
# 尝试不同的：
# - 学习率：1e-4, 1e-5, 1e-6
# - GNN层数：2, 3, 4
# - 嵌入维度：64, 128, 256
```

**任务4.3**：评估泛化能力
```python
# 在不同数据集上测试
for data_file in ['T20.xlsx', 'T40.xlsx', 'T80.xlsx', 'T100.xlsx']:
    # 测试并记录结果
```

---

## 预期效果

### 训练收敛曲线

预期在100-200次epoch后看到：
- 平均成本下降
- 操作符选择变得更有针对性
- 解的质量稳定在较优水平

### 性能对比

| 方法 | T20 | T40 | T80 | T100 |
|------|-----|-----|-----|------|
| 传统ALNS | 基准 | 基准 | 基准 | 基准 |
| RL-ALNS | 预期改进5-15% | 预期改进5-15% | 预期改进5-15% | 预期改进5-15% |

### 操作符选择分布

RL应该学会：
- 初期：更多使用randomRemoval（探索）
- 中期：更多使用timeRelatedRemoval和neighborRemoval（局部优化）
- 充电紧急时：优先选择greedyInsert（保证可行性）

---

## 关键注意事项

### ✅ DO

1. ✅ **保留所有原有操作符**：不要修改RemoveOperators和InsertOperators的实现
2. ✅ **渐进式测试**：先小数据集，再大数据集
3. ✅ **监控训练**：记录每个epoch的成本、操作符分布
4. ✅ **对比基准**：始终与传统ALNS对比
5. ✅ **归一化特征**：确保所有特征在相似的尺度

### ❌ DON'T

1. ❌ **不要修改操作符逻辑**：RL只学习选择，不改变实现
2. ❌ **不要过度设计特征**：12维足够，太多反而难训练
3. ❌ **不要期望立即改进**：RL需要充分训练（几小时到几天）
4. ❌ **不要忽略模拟退火**：可以同时保留模拟退火接受准则
5. ❌ **不要在所有实例上训练**：使用一部分训练，一部分测试

---

## 常见问题

### Q1: RL训练很慢怎么办？

**答**：
- 减小问题规模（先用T20、T40）
- 减少迭代次数（iterMax=5000）
- 使用GPU加速（如果可用）

### Q2: RL效果不如传统方法怎么办？

**答**：
- 检查特征设计是否合理
- 增加训练epoch数
- 调整学习率
- 确保图转换正确（单元测试）

### Q3: 如何可视化RL学到的策略？

**答**：
```python
# 记录操作符选择分布
from collections import Counter

selections = []
for iteration in range(1000):
    removeIdx, insertIdx, _, _, _ = alns_rl._select_operators_rl()
    selections.append((removeIdx, insertIdx))

remove_counts = Counter([s[0] for s in selections])
insert_counts = Counter([s[1] for s in selections])

print("Remove distribution:", remove_counts)
print("Insert distribution:", insert_counts)
```

---

## 总结

### 核心思路

```
你的EVSP-ALNS
    ↓
保留：RemoveOperators, InsertOperators, Schedule, Duty（完全不变）
    ↓
替换：WeightsManagement → Actor网络（RL选择）
    ↓
新增：GraphConverter（Schedule → Graph）
    ↓
结果：RL-ALNS
```

### 工作量估算

| 任务 | 时间 |
|------|------|
| 复制RL组件 | 1天 |
| 实现GraphConverter | 3-5天 |
| 实现ALNS_RL | 2-3天 |
| 测试与调试 | 5-7天 |
| 训练与优化 | 7-14天 |
| **总计** | **3-4周** |

### 最终收获

完成迁移后，你将获得：
1. ✅ 一个智能的操作符选择器（Actor网络）
2. ✅ 可能比传统方法好5-15%的解质量
3. ✅ 深入理解RL在组合优化中的应用
4. ✅ 可扩展的框架（易于添加新操作符）

祝你迁移顺利！🚀
