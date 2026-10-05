"""平台段校正（等值平台 + 锚点等式约束）的领域算法测试。

最优性用独立神谕核对：整序列枚举全部连续分区中平台段内无切点的候选
（块值取块内加权均值或锚点读数），取误差最小者；目标严格凸 ⇒ 最优
曲线唯一，与求解器的逐点 fitted 及总误差精确对拍。不可行情形由神谕
"无可行分区"交叉确认。全程不允许 float。
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from math import gcd

import pytest

from app.isotonic import (
    InfeasibleAnchorsError,
    InfeasiblePlateauError,
    fit_isotonic_plateau,
)

from .oracle import oracle_plateau_fit_whole


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


def _plateau_ranges(n):
    """n 个点中全部至少两点的连续区间（含端点 0 基位置列表）。"""

    return [
        list(range(first, last + 1))
        for first in range(n - 1)
        for last in range(first + 1, n)
    ]


def _assert_matches_oracle(readings, weights, plateau, anchors):
    expected_fitted, expected_error = oracle_plateau_fit_whole(
        readings, weights, plateau, anchors
    )
    if expected_fitted is None:
        # 神谕无可行分区：求解器必须给出不可行证据而非部分拟合
        with pytest.raises((InfeasibleAnchorsError, InfeasiblePlateauError)):
            fit_isotonic_plateau(readings, weights, plateau, anchors)
        return
    result = fit_isotonic_plateau(readings, weights, plateau, anchors)
    assert _fitted(result) == expected_fitted, (
        readings,
        weights,
        plateau,
        anchors,
    )
    assert _total(result) == expected_error
    # 分块 = 最优曲线的最大连续等值段（最粗口径）
    assert _block_ranges(result) == _maximal_equal_runs(expected_fitted)


# ---------------------------------------------------------------------------
# 独立整序列枚举神谕对拍（短序列全枚举 + 长序列随机抽样）
# ---------------------------------------------------------------------------


def test_plateau_matches_oracle_exhaustive_short():
    # n=2..3 全枚举：readings ∈ {0,1,2}，weights ∈ {1,2}，
    # 全部平台区间 × 0～2 个锚点的全部子集
    for n in range(2, 4):
        for readings in itertools.product(range(3), repeat=n):
            for weights in itertools.product([1, 2], repeat=n):
                for plateau in _plateau_ranges(n):
                    subsets = [()]
                    for k in (1, 2):
                        subsets.extend(itertools.combinations(range(n), k))
                    for anchors in subsets:
                        _assert_matches_oracle(
                            list(readings), list(weights), plateau, list(anchors)
                        )


def test_plateau_matches_oracle_exhaustive_length_4():
    # n=4：readings ∈ {0,1,2}，weights ∈ {1,2}，全部平台区间 × 0～1 个锚点
    n = 4
    for readings in itertools.product(range(3), repeat=n):
        for weights in itertools.product([1, 2], repeat=n):
            for plateau in _plateau_ranges(n):
                for anchors in [()] + [(i,) for i in range(n)]:
                    _assert_matches_oracle(
                        list(readings), list(weights), plateau, list(anchors)
                    )


def test_plateau_matches_oracle_random_longer():
    rng = random.Random(20261005)
    for _ in range(400):
        n = rng.randint(2, 8)
        readings = [rng.randint(0, 6) for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        first = rng.randrange(n - 1)
        last = rng.randrange(first + 1, n)
        plateau = list(range(first, last + 1))
        k = rng.randint(0, min(4, n))
        anchors = sorted(rng.sample(range(n), k))
        _assert_matches_oracle(readings, weights, plateau, anchors)


# ---------------------------------------------------------------------------
# 规定场景：联合求解、边界位置、平台内锚点、最粗分块
# ---------------------------------------------------------------------------


def test_plateau_value_solved_jointly_not_overwritten():
    # 平台 [1,2] 自身均值 3/2，但左侧 10 通过保序约束把平台共同值拉高：
    # 若先跑普通 PAVA 再强行覆盖平台为 3/2，曲线 10, 3/2, 3/2 不再非递减。
    # 联合求解下平台与左邻合并，全段共同值 13/3。
    result = fit_isotonic_plateau([10, 1, 2], [1, 1, 1], [1, 2])
    assert _fitted(result) == [Fraction(13, 3)] * 3
    assert _block_ranges(result) == ((0, 2),)
    assert _total(result) == Fraction(146, 3)

    # 平台被两侧同时拉动：[1, 10, 2, 3] 平台 [1,2] → 超观测 (w=2, r=6)，
    # 与右侧 3 合并为 (12+3)/3 = 5，左侧 1 保持独立
    result = fit_isotonic_plateau([1, 10, 2, 3], [1, 1, 1, 1], [1, 2])
    assert _fitted(result) == [Fraction(1), Fraction(5), Fraction(5), Fraction(5)]
    assert _block_ranges(result) == ((0, 0), (1, 3))


def test_plateau_at_sequence_boundaries():
    # 平台在序列开头：右侧只有保序下界
    result = fit_isotonic_plateau([1, 2, 10], [1, 1, 1], [0, 1])
    assert _fitted(result) == [Fraction(3, 2), Fraction(3, 2), Fraction(10)]
    assert _block_ranges(result) == ((0, 1), (2, 2))
    assert _total(result) == Fraction(1, 2)

    # 平台在序列末尾：左侧只有保序上界
    result = fit_isotonic_plateau([10, 1, 2], [1, 1, 1], [1, 2])
    assert _fitted(result) == [Fraction(13, 3)] * 3

    # 平台覆盖全部点：共同值即整体加权均值，单块
    result = fit_isotonic_plateau([3, 1, 2], [2, 1, 3], [0, 1, 2])
    assert _fitted(result) == [Fraction(13, 6)] * 3
    assert _block_ranges(result) == ((0, 2),)
    # 2*(5/6)^2 + (7/6)^2 + 3*(1/6)^2 = 102/36
    assert _total(result) == Fraction(17, 6)


def test_plateau_minimum_size_two_points():
    result = fit_isotonic_plateau([5, 1], [1, 1], [0, 1])
    assert _fitted(result) == [Fraction(3), Fraction(3)]
    assert _total(result) == Fraction(8)


def test_plateau_with_internal_anchor_pins_value():
    # 平台内锚点把共同值钉死在其 reading 上，两侧被夹到同一值
    result = fit_isotonic_plateau([9, 1, 2, 0], [1, 1, 1, 1], [1, 2], [2])
    assert _fitted(result) == [Fraction(2)] * 4
    assert _block_ranges(result) == ((0, 3),)
    assert _total(result) == Fraction(54)

    # 多个平台内锚点读数一致：共同值同样被钉死
    result = fit_isotonic_plateau([0, 10, 0, 10], [1, 1, 1, 1], [1, 2], [1, 3])
    assert _fitted(result) == [Fraction(0), Fraction(10), Fraction(10), Fraction(10)]
    assert _block_ranges(result) == ((0, 0), (1, 3))
    assert _total(result) == Fraction(100)

    # 平台内锚点权重再大也只是等式约束，不影响共同值的取值来源
    result = fit_isotonic_plateau([8, 0, 0], [1, 10**6, 1], [0, 1, 2], [1])
    assert _fitted(result) == [Fraction(0)] * 3


def test_plateau_anchors_on_both_sides():
    # 平台外两侧锚点等值：整段被夹平，平台与锚点合成一个分块
    result = fit_isotonic_plateau([5, 9, 1, 9, 5], [1] * 5, [1, 2], [0, 4])
    assert _fitted(result) == [Fraction(5)] * 5
    assert _block_ranges(result) == ((0, 4),)
    assert _total(result) == Fraction(48)


def test_equal_values_merge_across_plateau_boundary():
    # 平台共同值与右邻拟合值相等：跨平台边界合并为一个最粗分块
    result = fit_isotonic_plateau([1, 5, 5, 1], [1, 1, 1, 1], [1, 2])
    assert _fitted(result) == [Fraction(1), Fraction(11, 3), Fraction(11, 3), Fraction(11, 3)]
    assert _block_ranges(result) == ((0, 0), (1, 3))
    assert _total(result) == Fraction(32, 3)


def test_plateau_without_anchors_matches_reduced_pava():
    # 无锚点时等价于把平台段替换为单点（权重和、加权均值）后的普通 PAVA
    rng = random.Random(777)
    for _ in range(100):
        n = rng.randint(2, 10)
        readings = [rng.randint(0, 30) for _ in range(n)]
        weights = [rng.randint(1, 9) for _ in range(n)]
        first = rng.randrange(n - 1)
        last = rng.randrange(first + 1, n)
        plateau = list(range(first, last + 1))
        result = fit_isotonic_plateau(readings, weights, plateau)
        expected_fitted, expected_error = oracle_plateau_fit_whole(
            readings, weights, plateau
        )
        assert _fitted(result) == expected_fitted
        assert _total(result) == expected_error
        # 平台各点等值且曲线整体非递减
        assert len({expected_fitted[i] for i in plateau}) == 1
        assert _non_decreasing(expected_fitted)


# ---------------------------------------------------------------------------
# 不可行：平台内锚点矛盾、锚点顺序下降（均不返回部分拟合）
# ---------------------------------------------------------------------------


def test_infeasible_plateau_internal_anchors_disagree():
    # 平台内两个锚点读数不同：等值约束与等式约束矛盾
    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([1, 5, 3, 1], [1] * 4, [1, 2], [1, 2])
    assert (excinfo.value.left, excinfo.value.right) == (1, 2)

    # 读数下降的平台内锚点同样报平台矛盾（更具体的证据）
    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([1, 5, 3, 1], [1] * 4, [1, 2], [2, 1])
    assert (excinfo.value.left, excinfo.value.right) == (1, 2)

    # 三个平台内锚点：报告与首个锚点最早不一致的一对
    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([0, 7, 7, 2, 9], [1] * 5, [1, 2, 3, 4], [1, 2, 3])
    assert (excinfo.value.left, excinfo.value.right) == (1, 3)

    # 平台内锚点一致但平台外锚点与之矛盾时属顺序下降，不是平台矛盾
    with pytest.raises(InfeasibleAnchorsError):
        fit_isotonic_plateau([9, 5, 5, 0], [1] * 4, [1, 2], [0, 1])


def test_infeasible_anchor_decrease_with_plateau():
    # 平台外锚点下降：沿用锚点冲突契约（最早冲突的相邻锚点对）
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([7, 1, 2, 0], [1] * 4, [1, 2], [0, 3])
    assert (excinfo.value.left, excinfo.value.right) == (0, 3)

    # 平台内锚点（值被钉死）与左侧锚点下降
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([9, 5, 5, 8], [1] * 4, [1, 2], [0, 1])
    assert (excinfo.value.left, excinfo.value.right) == (0, 1)

    # 平台内锚点与右侧锚点下降
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([0, 5, 5, 2], [1] * 4, [1, 2], [2, 3])
    assert (excinfo.value.left, excinfo.value.right) == (2, 3)


def test_infeasible_cross_checked_with_oracle():
    # 随机实例：求解器报不可行 ⟺ 神谕无可行分区
    rng = random.Random(555)
    feasible = infeasible = 0
    for _ in range(600):
        n = rng.randint(2, 7)
        readings = [rng.randint(0, 4) for _ in range(n)]
        weights = [rng.randint(1, 3) for _ in range(n)]
        first = rng.randrange(n - 1)
        last = rng.randrange(first + 1, n)
        plateau = list(range(first, last + 1))
        k = rng.randint(1, min(3, n))
        anchors = sorted(rng.sample(range(n), k))
        expected_fitted, _ = oracle_plateau_fit_whole(
            readings, weights, plateau, anchors
        )
        try:
            fit_isotonic_plateau(readings, weights, plateau, anchors)
        except (InfeasibleAnchorsError, InfeasiblePlateauError):
            assert expected_fitted is None
            infeasible += 1
        else:
            assert expected_fitted is not None
            feasible += 1
    assert feasible > 0 and infeasible > 0


# ---------------------------------------------------------------------------
# 输入校验：非法位置、非连续范围、重复锚点
# ---------------------------------------------------------------------------


def test_plateau_position_validation():
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], [0])  # 平台少于两点
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], [])  # 空平台
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], [0, 0])  # 重复位置
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], [0, 2])  # 非连续范围
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3, 4], [1] * 4, [0, 1, 3])  # 非连续
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], [0, 2])  # 越界
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], [-1, 0])  # 越界
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1], [0, 1])  # 长度不齐
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], [0, 1], [1, 1])  # 重复锚点
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], [0, 1], [3])  # 锚点越界

    # 乱序传入平台位置与锚点均可接受
    result = fit_isotonic_plateau([3, 1, 2], [1, 1, 1], [2, 1, 0])
    assert _fitted(result) == [Fraction(2)] * 3


# ---------------------------------------------------------------------------
# 结构不变量：单调、平台等值、锚点原值、块覆盖、不可约、误差自洽、确定性
# ---------------------------------------------------------------------------


def test_plateau_invariants_on_random_instances():
    rng = random.Random(31337)
    checked = 0
    while checked < 200:
        n = rng.randint(2, 30)
        readings = [rng.randint(0, 100) for _ in range(n)]
        weights = [rng.randint(1, 10) for _ in range(n)]
        first = rng.randrange(n - 1)
        last = rng.randrange(first + 1, n)
        plateau = list(range(first, last + 1))
        k = rng.randint(0, min(12, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        inner = [p for p in anchors if p in set(plateau)]
        if len({readings[p] for p in inner}) > 1:
            continue
        checked += 1
        result = fit_isotonic_plateau(readings, weights, plateau, anchors)
        fitted = _fitted(result)

        # 保留全部点、非递减、平台各点等值、锚点原值保留
        assert len(fitted) == n
        assert _non_decreasing(fitted)
        assert len({fitted[i] for i in plateau}) == 1
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


def test_plateau_deterministic_for_identical_input():
    rng = random.Random(7)
    for _ in range(30):
        n = rng.randint(2, 20)
        readings = [rng.randint(0, 30) for _ in range(n)]
        weights = [rng.randint(1, 9) for _ in range(n)]
        first = rng.randrange(n - 1)
        last = rng.randrange(first + 1, n)
        plateau = list(range(first, last + 1))
        k = rng.randint(0, min(12, n))
        anchors = sorted(rng.sample(range(n), k))
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        inner = [p for p in anchors if p in set(plateau)]
        if len({readings[p] for p in inner}) > 1:
            continue
        first_result = fit_isotonic_plateau(readings, weights, plateau, anchors)
        for _ in range(3):
            assert (
                fit_isotonic_plateau(readings, weights, plateau, anchors)
                == first_result
            )
