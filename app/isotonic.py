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

平台段校正：一段经复测确认处于同一磨损平台的连续工位（至少两点）必须
取完全相等的校正值，该共同值本身是待求的精确有理数。把平台段坍缩为
单个"超观测"（权重 = 段内权重和，读数 = 段内加权均值，均为精确分数）：
段内目标 sum(w*(v-r)^2) 与超观测目标 W*(v-mean)^2 只差一个与 v 无关的
常数，段内等值约束又恰好替代了平台两侧的保序约束，因此归约后的问题
与原问题同解——平台值由整条曲线联合解出，而不是先跑普通 PAVA 再强行
覆盖。平台可同时携带锚点（平台内锚点把共同值钉死在其 reading 上；
平台内锚点读数不一致则等式约束互相矛盾，属不可行），归约后与平台外
锚点一起按等式约束联合求解。分块仍取整条曲线上的最大连续等值段：
等值跨越平台边界时也合并为同一块。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union


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

    fixed = [(pos, Fraction(readings[pos], 1)) for pos in positions]
    fitted = _fit_with_fixed_values(readings, weights, fixed)
    return _result_from_fitted(fitted, readings, weights)


def _fit_with_fixed_values(
    readings: Sequence[Union[int, Fraction]],
    weights: Sequence[Union[int, Fraction]],
    fixed: Sequence[Tuple[int, Fraction]],
) -> List[Fraction]:
    """等式约束下的精确保序拟合（内部共用件）。

    ``fixed`` 为（位置, 固定值）对，按位置升序、互不重复，且固定值已
    保证非递减（调用方负责可行性检查）。固定点把序列切成独立段（两端
    单侧有界、固定点之间双侧有界），段内做有界精确拟合。``readings``
    允许混入分数（平台归约后的超观测读数），全程无浮点。
    """

    n = len(readings)
    fitted: List[Fraction] = [Fraction(0, 1)] * n
    positions: List[int] = []
    for pos, value in fixed:
        fitted[pos] = value
        positions.append(pos)

    # 段边界：-1（序列起点左侧）与 n（序列终点右侧）表示该侧无界
    bounds = [-1] + positions + [n]
    for lo_bound, hi_bound in zip(bounds, bounds[1:]):
        start = lo_bound + 1
        end = hi_bound - 1
        if start > end:
            continue
        lower = None if lo_bound < 0 else fitted[lo_bound]
        upper = None if hi_bound >= n else fitted[hi_bound]
        segment = _fit_bounded_segment(
            readings[start : end + 1],
            weights[start : end + 1],
            lower,
            upper,
        )
        for offset, value in enumerate(segment):
            fitted[start + offset] = value
    return fitted


def _result_from_fitted(
    fitted: Sequence[Fraction],
    readings: Sequence[Union[int, Fraction]],
    weights: Sequence[Union[int, Fraction]],
) -> FitResult:
    """由逐点 fitted 构造结果：最大连续等值分块 + 逐点精确累加的误差。

    等值跨越锚点/平台边界时也合并为同一块。截断块、含锚点的块与平台块
    的取值未必等于块内加权均值，块误差必须逐点精确累加，不能用均值公式。
    """

    n = len(readings)
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


class InfeasiblePlateauError(ValueError):
    """平台段内的锚点读数不一致：平台要求各点等值，等式约束不可兼得。

    携带最早冲突的一对平台内锚点（按观测顺序，0 基位置），异常抛出时
    不产生任何部分拟合结果。
    """

    def __init__(self, left: int, right: int) -> None:
        super().__init__(
            f"anchors inside plateau disagree between positions {left} and {right}"
        )
        self.left = left
        """冲突对中靠前的锚点位置。"""
        self.right = right
        """冲突对中靠后的锚点位置。"""


def fit_isotonic_plateau(
    readings: Sequence[int],
    weights: Sequence[int],
    plateau_positions: Sequence[int],
    anchor_positions: Sequence[int] = (),
) -> FitResult:
    """平台段校正：平台各点共享一个待求的精确有理数校正值。

    ``plateau_positions`` 为处于同一磨损平台的连续观测位置（0 基，至少
    两点，可乱序传入，不得重复，必须恰好覆盖一段连续区间）；
    ``anchor_positions`` 为可同时传入的固定观测（规则同
    :func:`fit_isotonic_anchored`，可为空，允许落在平台内）。平台内各点
    的 fitted 严格相等，共同值与平台两侧的保序约束、全部锚点等式约束
    **联合**解出（不是先跑普通 PAVA 再覆盖输出）。

    不可行情形（抛出时不产生任何部分拟合）：

    * 平台内锚点读数不一致——:class:`InfeasiblePlateauError`，携带按
      观测顺序最早冲突的一对平台内锚点位置；
    * 锚点读数按观测顺序出现下降（含平台内锚点与平台外锚点之间）——
      :class:`InfeasibleAnchorsError`，携带最早冲突的相邻锚点对。

    归约：平台段坍缩为单个超观测（权重 = 段内权重和，读数 = 段内加权
    均值的精确分数），段内目标与超观测目标只差一个与共同值无关的常数，
    段内等值约束恰好替代平台两侧的保序约束，故归约后的问题与原问题
    同解；平台内锚点（读数必一致）转化为超观测上的等式约束。归约后的
    序列与锚点值仍只用整数与分数求解，最后把超观测的解展回平台各点，
    分块取整条曲线上的最大连续等值段（等值跨平台边界也合并）。
    """

    n = len(readings)
    if n != len(weights):
        raise ValueError("readings and weights must have equal length")

    plateau = sorted(plateau_positions)
    if len(plateau) < 2:
        raise ValueError("plateau must contain at least two points")
    if len(set(plateau)) != len(plateau):
        raise ValueError("plateau positions must be distinct")
    if plateau[0] < 0 or plateau[-1] >= n:
        raise ValueError("plateau position out of range")
    if plateau[-1] - plateau[0] != len(plateau) - 1:
        raise ValueError("plateau positions must be contiguous")
    first, last = plateau[0], plateau[-1]

    anchors = sorted(anchor_positions)
    if len(set(anchors)) != len(anchors):
        raise ValueError("anchor positions must be distinct")
    if anchors and (anchors[0] < 0 or anchors[-1] >= n):
        raise ValueError("anchor position out of range")

    plateau_set = set(plateau)
    inner = [pos for pos in anchors if pos in plateau_set]

    # 可行性 1：平台内锚点必须同值（平台等值约束与锚点等式约束的交集）
    if inner:
        base = inner[0]
        for other in inner[1:]:
            if readings[other] != readings[base]:
                raise InfeasiblePlateauError(base, other)

    # 可行性 2：锚点读数按观测顺序非递减；发现下降立即报告最早冲突对
    for prev, curr in zip(anchors, anchors[1:]):
        if readings[prev] > readings[curr]:
            raise InfeasibleAnchorsError(prev, curr)

    # 归约：平台段坍缩为单个超观测（精确分数读数，无浮点）
    sum_w = 0
    sum_wr = 0
    for i in range(first, last + 1):
        sum_w += weights[i]
        sum_wr += weights[i] * readings[i]
    reduced_readings: List[Union[int, Fraction]] = (
        list(readings[:first])
        + [Fraction(sum_wr, sum_w)]
        + list(readings[last + 1 :])
    )
    reduced_weights = (
        list(weights[:first]) + [sum_w] + list(weights[last + 1 :])
    )

    # 位置映射：平台左侧不变，超观测位于 first，平台右侧左移段长-1
    shift = last - first
    fixed: List[Tuple[int, Fraction]] = []
    for pos in anchors:
        if pos < first:
            fixed.append((pos, Fraction(readings[pos], 1)))
        elif pos > last:
            fixed.append((pos - shift, Fraction(readings[pos], 1)))
    if inner:
        # 平台内锚点读数已验证一致：共同值钉死在该读数上
        fixed.append((first, Fraction(readings[inner[0]], 1)))
    fixed.sort()

    reduced_fitted = _fit_with_fixed_values(
        reduced_readings, reduced_weights, fixed
    )

    # 展回：平台各点共享超观测的解
    value = reduced_fitted[first]
    fitted = (
        reduced_fitted[:first]
        + [value] * (last - first + 1)
        + reduced_fitted[first + 1 :]
    )
    return _result_from_fitted(fitted, readings, weights)
