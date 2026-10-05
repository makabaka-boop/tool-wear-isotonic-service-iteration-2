"""加权最小二乘保序回归（Pool Adjacent Violators Algorithm）。

输入点按加工顺序排列，输出唯一的非递减拟合曲线。整个计算只使用
``int`` 与分数（整数分子/整数分母），不引入 ``float``：

* 块均值的比较通过交叉相乘完成；
* 每个块的加权均值与块内平方误差在合并时即时约分；
* 总误差由各块分数以公分母精确累加后再约分。

当相邻块均值相等时也进行合并，因此不存在"在何处断开并列块"的人为
选择空间——算法始终返回所有最优解中最粗（块数最少）的规范分块，
相同输入必然得到相同分块与逐点 fitted。

固定观测（锚点）校正：经人工复测的观测必须原值保留，锚点的 fitted
以**等式约束**固定在其 reading 上（不是用极大有限权重近似），其余点
仍在整条非递减曲线上最小化加权平方误差。锚点把序列切成独立段——
首锚点左侧只有上界、相邻锚点之间同时有上下界、末锚点右侧只有下界。
链式全序加常数界下，有界解恰好等于无界保序解逐点截断到界内（截断后
KKT 的前缀/后缀条件依然成立），因此段内复用无界 PAVA 再截断即可，
全程仍然只用整数与分数。分块取整条曲线上的最大连续等值段：等值跨越
锚点时也合并为同一块。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Block:
    """一个连续合并块。"""

    start: int
    end: int
    mean_num: int
    """块加权均值的分子（不可约）。"""
    mean_den: int
    """块加权均值的分母（不可约，恒为正）。"""
    error_num: int
    """块内总误差 sum(w*(fitted-reading)^2) 的分子（不可约）。"""
    error_den: int
    """块内总误差的分母（不可约，恒为正）。"""


@dataclass(frozen=True)
class FitResult:
    """保序回归结果，分块索引均为含端点的 0 基索引。"""

    blocks: Tuple[Block, ...]
    fitted_num: Tuple[int, ...]
    """逐点 fitted 分子（与 observations 等长、同序）。"""
    fitted_den: Tuple[int, ...]
    """逐点 fitted 分母（恒为正）。"""
    total_error_num: int
    """总误差分子（不可约）。"""
    total_error_den: int
    """总误差分母（不可约，恒为正）。"""


@dataclass
class _Pool:
    """PAVA 运行时的中间合并块（可变）。"""

    start: int
    end: int
    sum_wr: int  # sum(weight * reading)
    sum_w: int  # sum(weight)
    sum_wr2: int  # sum(weight * reading^2)
    error_num: int
    error_den: int


def _merge(left: _Pool, right: _Pool) -> _Pool:
    """合并两个相邻块，并精确重算合并块的平方误差。

    合并块均值 m = sum(w*r)/sum(w)，块误差为::

        sum(w*r^2) - (sum(w*r))^2 / sum(w)
                  = (sum_w*sum_wr2 - sum_wr^2) / sum_w
    """

    sum_w = left.sum_w + right.sum_w
    sum_wr = left.sum_wr + right.sum_wr
    sum_wr2 = left.sum_wr2 + right.sum_wr2
    error_num = sum_w * sum_wr2 - sum_wr * sum_wr
    error = Fraction(error_num, sum_w)
    return _Pool(
        start=left.start,
        end=right.end,
        sum_wr=sum_wr,
        sum_w=sum_w,
        sum_wr2=sum_wr2,
        error_num=error.numerator,
        error_den=error.denominator,
    )


def fit_isotonic(
    readings: Sequence[int], weights: Sequence[int]
) -> FitResult:
    """计算加权最小二乘保序回归。

    ``readings[i]`` / ``weights[i]`` 为非负整数（权重为正整数）。
    返回规范（最粗）分块、逐点 fitted 与总误差，全部以不可约分数表示。
    """

    n = len(readings)
    if n != len(weights):
        raise ValueError("readings and weights must have equal length")
    if n == 0:
        return FitResult((), (), (), 0, 1)

    stacks: List[_Pool] = []
    for i, (reading, weight) in enumerate(zip(readings, weights)):
        pool = _Pool(
            start=i,
            end=i,
            sum_wr=weight * reading,
            sum_w=weight,
            sum_wr2=weight * reading * reading,
            error_num=0,
            error_den=1,
        )
        # 栈顶块均值 <= 新块均值 时违反非递减约束（== 也合并，
        # 以消除并列处断开位置的任意性），比较用交叉相乘，无浮点。
        while stacks:
            top = stacks[-1]
            if top.sum_wr * pool.sum_w < pool.sum_wr * top.sum_w:
                break
            pool = _merge(stacks.pop(), pool)
        stacks.append(pool)

    blocks: List[Block] = []
    fitted_num: List[int] = [0] * n
    fitted_den: List[int] = [1] * n
    total = Fraction(0, 1)

    for pool in stacks:
        mean = Fraction(pool.sum_wr, pool.sum_w)
        error = Fraction(pool.error_num, pool.error_den)
        blocks.append(
            Block(
                start=pool.start,
                end=pool.end,
                mean_num=mean.numerator,
                mean_den=mean.denominator,
                error_num=error.numerator,
                error_den=error.denominator,
            )
        )
        for idx in range(pool.start, pool.end + 1):
            fitted_num[idx] = mean.numerator
            fitted_den[idx] = mean.denominator
        total += error

    total = Fraction(total.numerator, total.denominator)
    return FitResult(
        blocks=tuple(blocks),
        fitted_num=tuple(fitted_num),
        fitted_den=tuple(fitted_den),
        total_error_num=total.numerator,
        total_error_den=total.denominator,
    )


class InfeasibleAnchorsError(ValueError):
    """锚点读数按观测顺序出现下降：等式约束与非递减曲线不可兼得。

    携带最早冲突的一对相邻锚点（按观测顺序相邻，0 基位置），
    异常抛出时不产生任何部分拟合结果。
    """

    def __init__(self, left: int, right: int) -> None:
        super().__init__(
            f"anchor readings decrease between positions {left} and {right}"
        )
        self.left = left
        """冲突对中靠前的锚点位置。"""
        self.right = right
        """冲突对中靠后的锚点位置。"""


def _clip(
    value: Fraction, lower: Optional[Fraction], upper: Optional[Fraction]
) -> Fraction:
    """把分数截断到 [lower, upper]（None 表示该侧无界）。"""

    if lower is not None and value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


def _fit_bounded_segment(
    readings: Sequence[int],
    weights: Sequence[int],
    lower: Optional[Fraction],
    upper: Optional[Fraction],
) -> List[Fraction]:
    """段内有界保序拟合：lower <= fitted <= upper，精确、无浮点。

    链式全序加常数上下界时，有界解等于无界保序解逐点截断到
    [lower, upper]：截断只把越界块推到界上，KKT 前缀/后缀条件在
    截断后仍成立。因此直接复用无界 PAVA 再截断，不引入任何
    "极大权重"式的近似固定。
    """

    result = fit_isotonic(readings, weights)
    return [
        _clip(Fraction(result.fitted_num[i], result.fitted_den[i]), lower, upper)
        for i in range(len(readings))
    ]


def fit_isotonic_anchored(
    readings: Sequence[int],
    weights: Sequence[int],
    anchor_positions: Sequence[int],
) -> FitResult:
    """固定观测的加权最小二乘保序回归。

    ``anchor_positions`` 为必须原值保留的观测位置（0 基，可乱序传入，
    不得重复）；这些点的 fitted 以等式约束恒等于其 reading，其余点在
    整条非递减曲线上最小化加权平方误差。锚点读数按观测顺序必须非递减，
    否则抛出 :class:`InfeasibleAnchorsError`（携带最早冲突的相邻锚点
    位置），不返回部分拟合。

    锚点把序列切成独立段（两端单侧有界、锚点之间双侧有界），段内做
    有界精确拟合；分块为整条曲线上的最大连续等值段——等值跨越锚点
    时也合并为同一块。``anchor_positions`` 为空时退化为普通保序回归，
    结果与 :func:`fit_isotonic` 逐项一致。
    """

    n = len(readings)
    if n != len(weights):
        raise ValueError("readings and weights must have equal length")

    positions = sorted(anchor_positions)
    if not positions:
        return fit_isotonic(readings, weights)
    if len(set(positions)) != len(positions):
        raise ValueError("anchor positions must be distinct")
    if positions[0] < 0 or positions[-1] >= n:
        raise ValueError("anchor position out of range")

    # 可行性：锚点读数按观测顺序非递减；发现下降立即报告最早冲突对
    for prev, curr in zip(positions, positions[1:]):
        if readings[prev] > readings[curr]:
            raise InfeasibleAnchorsError(prev, curr)

    fitted: List[Fraction] = [Fraction(0, 1)] * n
    for pos in positions:
        fitted[pos] = Fraction(readings[pos], 1)

    # 段边界：-1（序列起点左侧）与 n（序列终点右侧）表示该侧无界
    bounds = [-1] + positions + [n]
    for lo_bound, hi_bound in zip(bounds, bounds[1:]):
        start = lo_bound + 1
        end = hi_bound - 1
        if start > end:
            continue
        lower = None if lo_bound < 0 else Fraction(readings[lo_bound], 1)
        upper = None if hi_bound >= n else Fraction(readings[hi_bound], 1)
        segment = _fit_bounded_segment(
            readings[start : end + 1],
            weights[start : end + 1],
            lower,
            upper,
        )
        for offset, value in enumerate(segment):
            fitted[start + offset] = value

    # 最大连续等值分块（等值跨锚点也合并为一块）。截断块与含锚点的块
    # 取值未必等于块内加权均值，块误差必须逐点精确累加，不能用均值公式。
    blocks: List[Block] = []
    total = Fraction(0, 1)
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and fitted[stop + 1] == fitted[index]:
            stop += 1
        value = fitted[index]
        error = Fraction(0, 1)
        for k in range(index, stop + 1):
            diff = value - readings[k]
            error += weights[k] * diff * diff
        blocks.append(
            Block(
                start=index,
                end=stop,
                mean_num=value.numerator,
                mean_den=value.denominator,
                error_num=error.numerator,
                error_den=error.denominator,
            )
        )
        total += error
        index = stop + 1

    return FitResult(
        blocks=tuple(blocks),
        fitted_num=tuple(value.numerator for value in fitted),
        fitted_den=tuple(value.denominator for value in fitted),
        total_error_num=total.numerator,
        total_error_den=total.denominator,
    )
