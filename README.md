# VCE 变量分量估计库

VCE（Variance\-Component Estimation）用于在线性模型中估计多个协方差分量。库内实现了三种常见算法：Helmert VCE、LS\-VCE 以及经修正的 LS\-VCE\+. 适用于协方差可表示为多个已知矩阵线性组合的情况。

## 安装

推荐使用 [uv](https://github.com/astral-sh/uv) 或 `pip`：

```bash
uv pip install vce
```

安装完成后可在代码中引入各算法类，也可使用 `vce` 命令查看版本信息。

## 基本用法

```python
from vce import LSVCE
import numpy as np

# A 为设计矩阵，Q_blocks 为已知协方差块
A = np.eye(4)
Q_blocks = [np.eye(4), np.ones((4, 4))]

y = np.random.standard_normal(4)

est = LSVCE(A, Q_blocks)
result = est.fit(y)
print(result.sigma)
```

### Monte Carlo 实验

```python
from vce.simulation import Scenario, monte_carlo, evaluate

scn = Scenario(
    m=60,
    r_dim=3,
    block_sizes=[20, 20, 20],
    sigma_true=[5.0, 2.0, 1.0],
    n_trials=100,
)
results = monte_carlo(scn)
metrics = evaluate(results, scn.sigma_true, scn.m, scn.r_dim)
print(metrics["lsvce"].bias)
```

## 原理简介

假设观测向量 `y` 满足

\[y = A b + v, \quad \operatorname{Cov}(v) = \sum_k \sigma_k Q_k\]

其中 `Q_k` 为已知的协方差块，`\sigma_k` 为待估计的分量。各算法迭代更新 `\sigma`：

- **Helmert VCE** 利用正交投影矩阵构造方程求解；
- **LS-VCE** 基于最小二乘残差构造法方程；
- **LS-VCE+** 在 LS-VCE 基础上加入已知协方差部分 `Q0` 的影响，适用于模型含确定性噪声分量的场景。

迭代直至相邻估计量收敛或达到最大迭代次数，随后可以计算理论协方差、卡方统计量等指标。

## 适用条件

- 观测量满足线性模型，噪声服从零均值正态分布；
- 协方差可拆分为有限个已知矩阵的线性组合且这些矩阵半正定；
- 观测数量 `m` 大于未知参数数目 `r_dim`，以保证投影矩阵有效。

## 不适用条件

- 协方差结构随时间或状态变化，无法用固定矩阵表示；
- 未知协方差块或块矩阵并非半正定；
- 数据规模极大导致求逆或迭代开销过高。

## 参考文献

- P. J. G. Teunissen, A. Amiri-Simkooei, "Least-squares variance component estimation," *Journal of Geodesy*, 2007.
- A. Amiri-Simkooei, "Least-squares variance component estimation: theory and applications," PhD thesis, 2007.
