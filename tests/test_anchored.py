"""固定观测校正（锚点等式约束）的领域算法测试。

最优性用独立神谕核对：短序列上枚举全部连续分区、块值取
clip(加权均值, 下界, 上界) 得到的全部可行有理数候选曲线，取误差
最小者（目标严格凸 ⇒ 最优向量唯一），与求解器的逐点 fitted 及
总误差精确对拍。全程不允许 float。
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from math import gcd

import pytest

from app.isotonic import (
    InfeasibleAnchorsError,
    fit_isotonic,
    fit_isotonic_anchored,
)

from .oracle import oracle_anchored_fit, oracle_anchored_fit_whole


def _fitted(result):
    return [
        Fraction(num, den)
        for num, den in zip(result.fitted_num, result.fitted_den)
    ]


def _total(result):
    return Fraction(result.total_error_num, result.total_error_den)


def _block_ranges(result):
    return tuple((b.start, b.end) for b in result.blocks)


def _maximal_equal_runs(values):
    runs = []
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and values[stop + 1] == values[index]:
            stop += 1
        runs.append((index, stop))
        index = stop + 1
    return tuple(runs)


def _non_decreasing(values):
    return all(a <= b for a, b in zip(values, values[1:]))


# ---------------------------------------------------------------------------
# 独立有理数候选枚举对拍（短序列）
# ---------------------------------------------------------------------------


def test_anchored_matches_oracle_exhaustive_short():
    # n=2..4 全枚举：readings ∈ {0,1,2}，weights ∈ {1,2}，锚点子集 1～2 个
    for n in range(2, 5):
        for readings in itertools.product(range(3), repeat=n):
            for weights in itertools.product([1, 2], repeat=n):
                subsets = []
                for k in (1, 2):
                    subsets.extend(itertools.combinations(range(n), k))
                for anchors in subsets:
                    anchor_readings = [readings[p] for p in anchors]
                    if not _non_decreasing(anchor_readings):
                        continue  # 不可行情形由专门测试覆盖
                    result = fit_isotonic_anchored(
                        list(readings), list(weights), anchors
                    )
                    expected_fitted, expected_error = oracle_anchored_fit(
                        list(readings), list(weights), anchors
                    )
                    assert _fitted(result) == expected_fitted, (
                        readings,
                        weights,
                        anchors,
                    )
                    assert _total(result) == expected_error
                    # 分块 = 最优曲线的最大连续等值段
                    assert _block_ranges(result) == _maximal_equal_runs(
                        expected_fitted
                    )


def test_anchored_matches_oracle_random_longer():
    rng = random.Random(20260925)
    sampled = 0
    while sampled < 300:
        n = rng.randint(2, 8)
        readings = [rng.randint(0, 6) for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        k = rng.randint(1, min(4, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        sampled += 1
        result = fit_isotonic_anchored(readings, weights, anchors)
        expected_fitted, expected_error = oracle_anchored_fit(
            readings, weights, anchors
        )
        assert _fitted(result) == expected_fitted, (readings, weights, anchors)
        assert _total(result) == expected_error
        assert _block_ranges(result) == _maximal_equal_runs(expected_fitted)


def test_anchored_matches_whole_sequence_brute_force():
    # 与不做段分解、不用截断性质的整序列暴力枚举交叉核对
    rng = random.Random(123456)
    sampled = 0
    while sampled < 250:
        n = rng.randint(2, 7)
        readings = [rng.randint(0, 5) for _ in range(n)]
        weights = [rng.randint(1, 3) for _ in range(n)]
        k = rng.randint(1, min(3, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        sampled += 1
        result = fit_isotonic_anchored(readings, weights, anchors)
        expected_fitted, expected_error = oracle_anchored_fit_whole(
            readings, weights, anchors
        )
        assert _fitted(result) == expected_fitted, (readings, weights, anchors)
        assert _total(result) == expected_error


# ---------------------------------------------------------------------------
# 规定场景：相等锚点、端点锚点、全固定、跨锚点等值分块
# ---------------------------------------------------------------------------


def test_equal_anchors_force_plateau_between_them():
    # 两个等值锚点：之间所有点被夹到同一值，并与锚点合成一个分块
    result = fit_isotonic_anchored([0, 10, 0, 10], [1, 1, 1, 1], [1, 3])
    assert _fitted(result) == [Fraction(0), Fraction(10), Fraction(10), Fraction(10)]
    assert _block_ranges(result) == ((0, 0), (1, 3))
    assert _total(result) == Fraction(100)

    # 相邻的等值锚点
    result = fit_isotonic_anchored([4, 4, 9], [2, 3, 1], [0, 1])
    assert _fitted(result) == [Fraction(4), Fraction(4), Fraction(9)]
    assert _block_ranges(result) == ((0, 1), (2, 2))
    assert _total(result) == 0


def test_endpoint_anchors():
    # 仅首点锚定：右侧段只有下界
    result = fit_isotonic_anchored([3, 1, 2], [1, 1, 1], [0])
    assert _fitted(result) == [Fraction(3)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total(result) == Fraction(5)

    # 仅末点锚定：左侧段只有上界
    result = fit_isotonic_anchored([3, 1, 2], [1, 1, 1], [2])
    assert _fitted(result) == [Fraction(2)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total(result) == Fraction(2)

    # 首尾同时锚定：中段双侧有界
    result = fit_isotonic_anchored([5, 1, 9], [1, 1, 1], [0, 2])
    assert _fitted(result) == [Fraction(5), Fraction(5), Fraction(9)]
    assert _block_ranges(result) == ((0, 1), (2, 2))
    assert _total(result) == Fraction(16)


def test_all_points_fixed():
    result = fit_isotonic_anchored([1, 2, 2, 5], [3, 1, 4, 1], [0, 1, 2, 3])
    assert _fitted(result) == [Fraction(1), Fraction(2), Fraction(2), Fraction(5)]
    assert _total(result) == 0
    # 等值的相邻锚点合并为同一块
    assert _block_ranges(result) == ((0, 0), (1, 2), (3, 3))


def test_equal_values_merge_across_anchor_into_single_block():
    # 段拟合值与锚点值相等时，跨锚点也只是一个分块
    result = fit_isotonic_anchored([9, 5, 1], [1, 1, 1], [1])
    assert _fitted(result) == [Fraction(5)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total(result) == Fraction(32)


def test_anchor_is_exact_equality_not_heavy_weight():
    # 若用极大有限权重"假装固定"，锚点会被邻点拉动而偏离原值；
    # 等式约束下锚点必须分毫不差
    result = fit_isotonic_anchored([100, 0], [1, 1], [0])
    assert _fitted(result) == [Fraction(100), Fraction(100)]
    assert _total(result) == Fraction(10000)

    # 锚点自身权重再大，fitted 仍恒等于 reading
    result = fit_isotonic_anchored([3, 1, 2], [1, 10**6, 1], [1])
    assert _fitted(result)[1] == Fraction(1)
    assert _fitted(result) == [Fraction(1), Fraction(1), Fraction(2)]


def test_empty_anchor_list_matches_plain_fit_exactly():
    rng = random.Random(99)
    for _ in range(50):
        n = rng.randint(2, 12)
        readings = [rng.randint(0, 50) for _ in range(n)]
        weights = [rng.randint(1, 9) for _ in range(n)]
        assert fit_isotonic_anchored(readings, weights, []) == fit_isotonic(
            readings, weights
        )


# ---------------------------------------------------------------------------
# 不可行锚点：最早冲突的相邻锚点对
# ---------------------------------------------------------------------------


def test_infeasible_anchors_raise_earliest_conflict():
    # 多对冲突时报告最早的一对
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([7, 1, 9, 2, 5], [1] * 5, [0, 1, 3, 4])
    assert (excinfo.value.left, excinfo.value.right) == (0, 1)

    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([1, 5, 4, 0], [1] * 4, [1, 2, 3])
    assert (excinfo.value.left, excinfo.value.right) == (1, 2)

    # "相邻锚点"指锚点序列中相邻，观测位置上不必相邻
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([7, 1, 9, 2, 5], [1] * 5, [0, 3])
    assert (excinfo.value.left, excinfo.value.right) == (0, 3)

    # 乱序传入按观测顺序判断
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_anchored([5, 8, 1], [1] * 3, [2, 0])
    assert (excinfo.value.left, excinfo.value.right) == (0, 2)

    # 等值不算下降
    result = fit_isotonic_anchored([5, 8, 5], [1] * 3, [0, 2])
    assert _fitted(result)[0] == _fitted(result)[2] == Fraction(5)


def test_anchor_position_validation():
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [0, 0])  # 重复位置
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [2])  # 越界
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1, 1], [-1])
    with pytest.raises(ValueError):
        fit_isotonic_anchored([1, 2], [1], [0])  # 长度不齐


# ---------------------------------------------------------------------------
# 结构不变量：单调、锚点原值、块覆盖、不可约、误差自洽
# ---------------------------------------------------------------------------


def test_anchored_invariants_on_random_instances():
    rng = random.Random(31337)
    checked = 0
    while checked < 200:
        n = rng.randint(2, 30)
        readings = [rng.randint(0, 100) for _ in range(n)]
        weights = [rng.randint(1, 10) for _ in range(n)]
        k = rng.randint(1, min(12, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        checked += 1
        result = fit_isotonic_anchored(readings, weights, anchors)
        fitted = _fitted(result)

        # 保留全部点、非递减、锚点原值保留
        assert len(fitted) == n
        assert _non_decreasing(fitted)
        for pos in anchors:
            assert fitted[pos] == readings[pos]

        # 块连续覆盖全部点且无重叠
        assert result.blocks[0].start == 0
        assert result.blocks[-1].end == n - 1
        for prev, curr in zip(result.blocks, result.blocks[1:]):
            assert curr.start == prev.end + 1

        total = Fraction(0, 1)
        for block in result.blocks:
            mean = Fraction(block.mean_num, block.mean_den)
            assert block.mean_den > 0
            assert gcd(abs(block.mean_num), block.mean_den) == 1
            error = Fraction(0, 1)
            for i in range(block.start, block.end + 1):
                assert fitted[i] == mean
                error += weights[i] * (mean - readings[i]) ** 2
            assert Fraction(block.error_num, block.error_den) == error
            assert block.error_den > 0
            assert gcd(abs(block.error_num), block.error_den) == 1
            total += error

        assert _total(result) == total
        assert result.total_error_den > 0
        assert gcd(abs(result.total_error_num), result.total_error_den) == 1
        # 逐点误差之和等于总误差
        pointwise = sum(
            (weights[i] * (fitted[i] - readings[i]) ** 2 for i in range(n)),
            Fraction(0, 1),
        )
        assert pointwise == total


def test_anchored_deterministic_for_identical_input():
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(2, 20)
        readings = [rng.randint(0, 30) for _ in range(n)]
        weights = [rng.randint(1, 9) for _ in range(n)]
        k = rng.randint(1, min(12, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        first = fit_isotonic_anchored(readings, weights, anchors)
        for _ in range(3):
            assert fit_isotonic_anchored(readings, weights, anchors) == first
