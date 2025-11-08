# AGV-ALNS-RL: AGV Scheduling with Reinforcement Learning

Python implementation of AGV (Automated Guided Vehicle) charging scheduling problem using ALNS (Adaptive Large Neighborhood Search) enhanced with Reinforcement Learning.

## Overview

This project integrates three codebases:
1. **C++ AGV-ALNS** - Original C++ implementation of AGV charging problem
2. **EVSP Framework** - Python ALNS framework for electric vehicle scheduling
3. **VRP-ALNS-RL** - RL-enhanced ALNS for vehicle routing

## Key Features

- **15 ALNS Operators**: 6 destroy + 9 repair operators
- **RL-Based Operator Selection**: Actor-Critic networks replace traditional adaptive weights
- **Graph Neural Networks**: GNN encodes AGV scheduling states (15-dimensional features)
- **Charging Constraints**: Handles SOC (State of Charge) and station capacity
- **Python-Idiomatic Design**: Uses dataclasses, type hints, and clean architecture

## Project Structure

```
AGV-ALNS-RL-Python/
├── agv_model/              # Data structures
│   ├── Event.py            # Task/charging events
│   ├── AGVPath.py          # Single AGV schedule
│   ├── Solution.py         # Overall solution
│   └── ProblemData.py      # Problem instance
│
├── agv_alns/               # ALNS algorithm
│   ├── DestroyOperators.py # 6 destroy operators
│   ├── RepairOperators.py  # 9 repair operators
│   ├── GraphConverter.py   # Solution → Graph conversion
│   └── ALNS_RL.py          # Main solver
│
├── rl_components/          # RL components (from VRP project)
│   ├── GNN.py              # Graph Isomorphism Network
│   ├── actor.py            # Policy network
│   └── critic.py           # Value network
│
├── scripts/
│   ├── data_generator.py   # Generate test instances
│   ├── train.py            # Training script
│   └── test.py             # Testing script
│
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- NumPy
- tqdm

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Train the RL agent on randomly generated instances:

```bash
python scripts/train.py \
    --episodes 100 \
    --tasks 20 \
    --agvs 5 \
    --stations 3 \
    --iterations 500 \
    --save ./checkpoints/agv_alns_rl.pth
```

**Arguments:**
- `--episodes`: Number of training episodes (default: 100)
- `--tasks`: Number of tasks per instance (default: 20)
- `--agvs`: Number of AGVs (default: 5)
- `--stations`: Number of charging stations (default: 3)
- `--iterations`: ALNS iterations per episode (default: 500)
- `--seed`: Random seed (default: 42)
- `--save`: Model save path

### 2. Testing

Test trained model and compare with baseline (random operator selection):

```bash
python scripts/test.py \
    --model ./checkpoints/agv_alns_rl.pth \
    --instances 20 \
    --tasks 20 \
    --agvs 5 \
    --stations 3 \
    --iterations 500
```

**Output:**
- RL performance statistics
- Baseline performance statistics
- Comparison: RL improvement over baseline

### 3. Using as Library

```python
from agv_alns.ALNS_RL import AGV_ALNS_RL
from scripts.data_generator import generate_agv_instance

# Generate problem
problem = generate_agv_instance(num_tasks=20, num_agvs=5, num_stations=3)

# Create solver
solver = AGV_ALNS_RL(problem, use_rl=True, max_iterations=500)

# Solve
solution = solver.solve()

print(f"Best cost: {solution.total_cost}")
print(f"Number of AGVs used: {len(solution.agv_paths)}")
print(f"Makespan: {solution.get_makespan()}")
```

## Algorithm Details

### ALNS Operators

**Destroy Operators (6):**
1. `RandomTaskRemoval` - Randomly remove tasks
2. `CriticalTaskRemoval` - Remove tasks on critical path
3. `WorstTaskRemoval` - Remove highest-cost tasks
4. `BusiestStationRemoval` - Remove tasks from AGVs using busiest station
5. `RandomStationRemoval` - Remove tasks from AGVs using random station
6. `StationRelatedRemoval` - Remove spatially close tasks to a station

**Repair Operators (9) - 3×3 Matrix:**

| Selection | Optimal | Random | Time |
|-----------|---------|--------|------|
| **Greedy** | GreedyOptimalInsert | GreedyRandomInsert | GreedyTimeInsert |
| **Random** | RandomOptimalInsert | RandomRandomInsert | RandomTimeInsert |
| **Adaptive** | AdaptiveOptimalInsert | AdaptiveRandomInsert | AdaptiveTimeInsert |

### RL Mechanism

- **Actor Network**: Selects destroy/repair operators based on graph state
- **Critic Network**: Evaluates state value for advantage calculation
- **GNN Encoder**: Encodes solution as graph with 15-dimensional node features
- **Training**: REINFORCE algorithm with rewards-to-go

### Graph Representation (15 Features)

**Event Nodes:**
1-3: Structural (position, AGV ID, event type)
4-6: Temporal (start time, duration, waiting)
7-9: Energy (SOC before, after, change)
10-12: Spatial (X, Y coords, distance)
13-15: Charging (station usage, is_charging, urgency)

**Center Node:** Global aggregation of solution state

## Design Principles

### "保持理智与清醒" (Stay Rational and Clear-Headed)

1. **Python-First Design**: Don't directly translate C++, use Python idioms
2. **Type Safety**: Use dataclasses, type hints, and enums
3. **Modularity**: Separate concerns (model, ALNS, RL, data)
4. **Clarity**: Clear naming, comprehensive docstrings
5. **Testability**: Independent operators, easy to unit test

### Key Improvements over C++

| Aspect | C++ | Python | Benefit |
|--------|-----|--------|---------|
| Data structures | Pointers, manual memory | dataclass | Cleaner, safer |
| Event handling | Flags | Enum (EventType) | Type-safe |
| Validation | Embedded | Separate methods | Testable |
| RL integration | N/A | Native PyTorch | Easy training |

## Expected Performance

| Metric | Baseline (Random) | RL-ALNS | Improvement |
|--------|-------------------|---------|-------------|
| Solution cost | Baseline | -10~20% | ✅ |
| Convergence speed | Baseline | +30~50% | ✅ |
| Charging efficiency | Baseline | -15~25% | ✅ |
| Stability | Medium | High | ✅ |

## Migration from C++ AGV-ALNS

This project successfully migrates C++ AGV-ALNS to Python with RL enhancement:

- ✅ All 15 operators implemented
- ✅ Charging constraints preserved
- ✅ Graph representation designed
- ✅ RL integration complete
- ✅ Training/testing scripts ready

**Time to implement:** ~2 weeks (actual: following 6-week plan from migration guide)

## References

- **Original C++ AGV-ALNS**: [Your thesis paper]
- **EVSP Framework**: alns-framework-for-evsp
- **VRP-ALNS-RL**: Solving-CVRP-by-ALNS-and-RL

## License

MIT License

## Citation

If you use this code, please cite:

```bibtex
@software{agv_alns_rl,
  title={AGV-ALNS-RL: Reinforcement Learning Enhanced ALNS for AGV Scheduling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AGV-ALNS-RL-Python}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
