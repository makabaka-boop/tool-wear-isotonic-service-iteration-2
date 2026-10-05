"""保序回归的"神谕"实现：枚举全部连续分区，用 Fraction 精确计算。

用于与 PAVA 对拍：n 个点的连续分区共有 2^(n-1) 种（每个相邻间隙
选择断开或合并）。对每个分区，块内 fitted 取加权均值，总误差为各块
误差之和。我们取总误差最小的分区；最优分块唯一（PAVA 的最粗分块），
故直接断言枚举选出的分块与 PAVA 完全一致。
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple


def _cut_masks(n: int):
    """生成 n 个点的全部 2^(n-1) 个连续分区（用切点位掩码表示）。"""

    for mask in range(1 << (n - 1)):
        yield mask


def _block_error(
    readings: Sequence[int], weights: Sequence[int], start: int, end: int
) -> Fraction:
    sw = sum(weights[i] for i in range(start, end + 1))
    swr = sum(weights[i] * readings[i] for i in range(start, end + 1))
    swr2 = sum(
        weights[i] * readings[i] * readings[i]
        for i in range(start, end + 1)
    )
    return Fraction(sw * swr2 - swr * swr, sw)


def _partition_blocks(
    readings: Sequence[int], weights: Sequence[int], mask: int
) -> Tuple[List[Tuple[int, int]], Fraction, bool]:
    """按切分掩码构造分块，返回（区间列表, 总误差, 是否可行）。

    可行当且仅当各块加权均值非递减——单点分块对降序输入误差为 0，
    但拟合曲线下降，不属于保序回归的可行域。
    """

    n = len(readings)
    ranges: List[Tuple[int, int]] = []
    error = Fraction(0, 1)
    prev_mean: Fraction | None = None
    feasible = True
    start = 0
    for i in range(n - 1):
        if mask & (1 << i):
            ranges.append((start, i))
            sw = sum(weights[start : i + 1])
            swr = sum(
                weights[j] * readings[j] for j in range(start, i + 1)
            )
            mean = Fraction(swr, sw)
            if prev_mean is not None and mean < prev_mean:
                feasible = False
            prev_mean = mean
            error += _block_error(readings, weights, start, i)
            start = i + 1
    ranges.append((start, n - 1))
    sw = sum(weights[start:n])
    swr = sum(weights[j] * readings[j] for j in range(start, n))
    mean = Fraction(swr, sw)
    if prev_mean is not None and mean < prev_mean:
        feasible = False
    error += _block_error(readings, weights, start, n - 1)
    return ranges, error, feasible


def oracle_best_partition(
    readings: Sequence[int], weights: Sequence[int]
) -> Tuple[Tuple[Tuple[int, int], ...], Fraction]:
    """枚举全部连续分区，返回（最优可行分块区间序列，最小总误差）。"""

    n = len(readings)
    best_ranges: Tuple[Tuple[int, int], ...] | None = None
    best_error = Fraction(10**100, 1)

    for mask in _cut_masks(n):
        ranges, error, feasible = _partition_blocks(readings, weights, mask)
        if not feasible:
            continue
        if error < best_error:
            best_error = error
            best_ranges = tuple(ranges)

    assert best_ranges is not None
    return best_ranges, best_error


def oracle_bounded_segment(
    readings: Sequence[int],
    weights: Sequence[int],
    lower: Fraction | None,
    upper: Fraction | None,
) -> Tuple[List[Fraction], Fraction]:
    """段内有界保序的独立枚举神谕（lower/upper 为 None 表示该侧无界）。

    最优逐点取值必为某连续区间的加权均值或其向 [lower, upper] 的截断，
    因此枚举全部连续分区、每块候选值取 clip(块加权均值, lower, upper)，
    即穷尽了全部可能最优的有理数候选曲线。过滤块值非递减的可行候选，
    返回（最优逐点 fitted，最小误差）；目标严格凸，最优向量唯一。
    """

    n = len(readings)
    if n == 0:
        return [], Fraction(0, 1)

    best_values: List[Fraction] | None = None
    best_error: Fraction | None = None
    for mask in _cut_masks(n):
        ranges: List[Tuple[int, int]] = []
        start = 0
        for i in range(n - 1):
            if mask & (1 << i):
                ranges.append((start, i))
                start = i + 1
        ranges.append((start, n - 1))

        values: List[Fraction] = []
        feasible = True
        prev: Fraction | None = None
        for block_start, block_end in ranges:
            sw = sum(weights[block_start : block_end + 1])
            swr = sum(
                weights[j] * readings[j]
                for j in range(block_start, block_end + 1)
            )
            value = Fraction(swr, sw)
            if lower is not None and value < lower:
                value = Fraction(lower)
            if upper is not None and value > upper:
                value = Fraction(upper)
            if prev is not None and value < prev:
                feasible = False
                break
            prev = value
            values.extend([value] * (block_end - block_start + 1))
        if not feasible:
            continue

        error = sum(
            (weights[i] * (values[i] - readings[i]) ** 2 for i in range(n)),
            Fraction(0, 1),
        )
        if best_error is None or error < best_error:
            best_error = error
            best_values = values

    assert best_values is not None and best_error is not None
    return best_values, best_error


def oracle_anchored_fit(
    readings: Sequence[int], weights: Sequence[int], anchor_positions: Sequence[int]
) -> Tuple[List[Fraction], Fraction]:
    """固定观测校正的独立神谕：锚点原值固定，各段独立枚举后拼接。

    返回（逐点 fitted，总误差）。调用前需保证锚点读数按观测顺序非递减。
    """

    n = len(readings)
    positions = sorted(anchor_positions)
    fitted: List[Fraction | None] = [None] * n
    for pos in positions:
        fitted[pos] = Fraction(readings[pos])

    total = Fraction(0, 1)
    bounds = [-1] + positions + [n]
    for lo_bound, hi_bound in zip(bounds, bounds[1:]):
        start = lo_bound + 1
        end = hi_bound - 1
        if start > end:
            continue
        lower = None if lo_bound < 0 else Fraction(readings[lo_bound])
        upper = None if hi_bound >= n else Fraction(readings[hi_bound])
        values, error = oracle_bounded_segment(
            readings[start : end + 1],
            weights[start : end + 1],
            lower,
            upper,
        )
        for offset, value in enumerate(values):
            fitted[start + offset] = value
        total += error

    assert all(value is not None for value in fitted)
    return fitted, total  # type: ignore[return-value]


def oracle_anchored_fit_whole(
    readings: Sequence[int], weights: Sequence[int], anchor_positions: Sequence[int]
) -> Tuple[List[Fraction], Fraction]:
    """整序列暴力枚举神谕：不做段分解、不用截断性质。

    枚举整条序列的全部连续分区：含锚点的块强制取锚点读数（同块含两个
    不同读数的锚点则该分区不可行），其余块取块内加权均值；过滤块值
    非递减的可行候选，取误差最小者。与求解器不共享任何结构假设，
    用于交叉核对段分解 + 截断实现的正确性。
    """

    n = len(readings)
    anchor_set = set(anchor_positions)
    best_values: List[Fraction] | None = None
    best_error: Fraction | None = None
    for mask in _cut_masks(n):
        ranges: List[Tuple[int, int]] = []
        start = 0
        for i in range(n - 1):
            if mask & (1 << i):
                ranges.append((start, i))
                start = i + 1
        ranges.append((start, n - 1))

        values: List[Fraction] = []
        feasible = True
        prev: Fraction | None = None
        for block_start, block_end in ranges:
            block_anchors = [
                i for i in range(block_start, block_end + 1) if i in anchor_set
            ]
            if block_anchors:
                value = Fraction(readings[block_anchors[0]])
                if any(
                    readings[i] != readings[block_anchors[0]]
                    for i in block_anchors
                ):
                    feasible = False
                    break
            else:
                sw = sum(weights[block_start : block_end + 1])
                swr = sum(
                    weights[j] * readings[j]
                    for j in range(block_start, block_end + 1)
                )
                value = Fraction(swr, sw)
            if prev is not None and value < prev:
                feasible = False
                break
            prev = value
            values.extend([value] * (block_end - block_start + 1))
        if not feasible:
            continue

        error = sum(
            (weights[i] * (values[i] - readings[i]) ** 2 for i in range(n)),
            Fraction(0, 1),
        )
        if best_error is None or error < best_error:
            best_error = error
            best_values = values

    assert best_values is not None and best_error is not None
    return best_values, best_error
