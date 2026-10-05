"""平台段校正（共享平台值 + 锚点等式约束）的领域算法测试。

最优性用独立神谕核对：直接在原序列上枚举全部连续分区（平台必须完整
落在单一段内，段值取加权均值或锚点读数），用 Fraction 精确计算误差，
取最小者（目标严格凸 ⇒ 最优向量唯一），与求解器的逐点 fitted、总误差
及最大连续等值分块精确对拍。神谕不做超点归约、不做段分解、不用截断
性质。全程不允许 float。
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
    fit_isotonic,
    fit_isotonic_anchored,
    fit_isotonic_plateau,
)

from .oracle import oracle_plateau_fit


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


def _platform_ranges(n):
    return [
        (start, end)
        for start in range(n)
        for end in range(start + 1, n)
    ]


# ---------------------------------------------------------------------------
# 独立分区枚举神谕对拍（短序列全枚举 + 长序列随机）
# ---------------------------------------------------------------------------


def test_plateau_matches_oracle_exhaustive_short():
    # n=2..3 全枚举：readings ∈ {0,1,2}，weights ∈ {1,2}，全部平台范围
    # 与全部锚点子集；n=4 枚举至多 2 个锚点的子集（不可行冲突都是成对
    # 出现的，更大子集在 n=2..3 已覆盖）。神谕无可行分区 ⟺ 求解器抛出
    # 不可行异常。
    for n in (2, 3, 4):
        max_anchors = n if n <= 3 else 2
        for readings in itertools.product(range(3), repeat=n):
            for weights in itertools.product([1, 2], repeat=n):
                for platform_start, platform_end in _platform_ranges(n):
                    for k in range(max_anchors + 1):
                        for anchors in itertools.combinations(range(n), k):
                            _check_against_oracle(
                                list(readings),
                                list(weights),
                                platform_start,
                                platform_end,
                                list(anchors),
                            )


def test_plateau_matches_oracle_random_longer():
    rng = random.Random(20261005)
    for _ in range(400):
        n = rng.randint(2, 8)
        readings = [rng.randint(0, 6) for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        platform_start = rng.randrange(n - 1)
        platform_end = rng.randrange(platform_start + 1, n)
        k = rng.randint(0, min(4, n))
        anchors = sorted(rng.sample(range(n), k))
        _check_against_oracle(
            readings, weights, platform_start, platform_end, anchors
        )


def _check_against_oracle(
    readings, weights, platform_start, platform_end, anchors
):
    expected = oracle_plateau_fit(
        readings, weights, platform_start, platform_end, anchors
    )
    if expected is None:
        # 不可行：必须抛出携带冲突对的异常，不返回部分拟合
        with pytest.raises((InfeasibleAnchorsError, InfeasiblePlateauError)):
            fit_isotonic_plateau(
                readings, weights, platform_start, platform_end, anchors
            )
        return
    expected_fitted, expected_error = expected
    result = fit_isotonic_plateau(
        readings, weights, platform_start, platform_end, anchors
    )
    context = (readings, weights, platform_start, platform_end, anchors)
    assert _fitted(result) == expected_fitted, context
    assert _total(result) == expected_error, context
    # 分块 = 最优曲线的最大连续等值段（最粗口径）
    assert _block_ranges(result) == _maximal_equal_runs(expected_fitted), context


# ---------------------------------------------------------------------------
# 规定场景：联合求解（非 PAVA 后覆盖）、边界、锚点交互、跨平台分块合并
# ---------------------------------------------------------------------------


def test_plateau_is_joint_solve_not_pava_then_overwrite():
    # 普通 PAVA 得 [1,2,10]（误差 0），平台约束把前两点联合拉到 3/2
    result = fit_isotonic_plateau([1, 2, 10], [1, 1, 1], 0, 1)
    assert _fitted(result) == [Fraction(3, 2), Fraction(3, 2), Fraction(10)]
    assert _total(result) == Fraction(1, 2)
    assert _block_ranges(result) == ((0, 1), (2, 2))
    assert _fitted(result) != _fitted(fit_isotonic([1, 2, 10], [1, 1, 1]))

    # 先 PAVA 再把平台改写成平台均值，会得到非单调且非最优的曲线；
    # 联合求解把左侧邻点一并拉动：[8/3, 8/3, 8/3, 4]
    result = fit_isotonic_plateau([5, 1, 2, 4], [1, 1, 1, 1], 1, 2)
    assert _fitted(result) == [Fraction(8, 3)] * 3 + [Fraction(4)]
    assert _total(result) == Fraction(26, 3)
    assert _block_ranges(result) == ((0, 2), (3, 3))
    naive = _fitted(fit_isotonic([5, 1, 2, 4], [1, 1, 1, 1]))
    naive[1] = naive[2] = Fraction(3, 2)  # 强行覆盖平台输出
    assert _fitted(result) != naive


def test_plateau_at_sequence_boundaries():
    # 平台覆盖全部点：共享值 = 整体加权均值
    result = fit_isotonic_plateau([3, 1, 2], [1, 1, 1], 0, 2)
    assert _fitted(result) == [Fraction(2)] * 3
    assert _total(result) == Fraction(2)
    assert _block_ranges(result) == ((0, 2),)

    # 平台在序列末尾：左侧无约束
    result = fit_isotonic_plateau([1, 5, 2], [1, 1, 1], 1, 2)
    assert _fitted(result) == [Fraction(1), Fraction(7, 2), Fraction(7, 2)]
    assert _total(result) == Fraction(9, 2)
    assert _block_ranges(result) == ((0, 0), (1, 2))

    # 平台在序列开头：右侧无约束
    result = fit_isotonic_plateau([5, 2, 8], [1, 1, 1], 0, 1)
    assert _fitted(result) == [Fraction(7, 2), Fraction(7, 2), Fraction(8)]
    assert _total(result) == Fraction(9, 2)
    assert _block_ranges(result) == ((0, 1), (2, 2))


def test_plateau_weighted_shared_value():
    # 平台共享值是平台内加权均值，不是算术均值
    result = fit_isotonic_plateau([0, 6], [3, 1], 0, 1)
    assert _fitted(result) == [Fraction(3, 2)] * 2
    assert _total(result) == Fraction(27)


def test_plateau_clipped_by_neighboring_anchors():
    # 左锚点把平台值顶上去，并与平台合并为一个分块
    result = fit_isotonic_plateau([10, 1, 2], [1, 1, 1], 1, 2, [0])
    assert _fitted(result) == [Fraction(10)] * 3
    assert _total(result) == Fraction(145)
    assert _block_ranges(result) == ((0, 2),)

    # 右锚点把平台值压下来
    result = fit_isotonic_plateau([8, 9, 1], [1, 1, 1], 0, 1, [2])
    assert _fitted(result) == [Fraction(1)] * 3
    assert _total(result) == Fraction(113)
    assert _block_ranges(result) == ((0, 2),)

    # 两侧锚点夹住平台
    result = fit_isotonic_plateau([10, 1, 2, 20], [1, 1, 1, 1], 1, 2, [0, 3])
    assert _fitted(result) == [Fraction(10), Fraction(10), Fraction(10), Fraction(20)]
    assert _total(result) == Fraction(145)
    assert _block_ranges(result) == ((0, 2), (3, 3))


def test_plateau_anchor_inside_platform_forces_shared_value():
    # 平台内单个锚点：平台值被等式约束到该锚点读数
    result = fit_isotonic_plateau([1, 5, 9], [1, 1, 1], 0, 2, [1])
    assert _fitted(result) == [Fraction(5)] * 3
    assert _total(result) == Fraction(32)
    assert _block_ranges(result) == ((0, 2),)

    # 平台内多个等值锚点
    result = fit_isotonic_plateau([4, 4, 9], [2, 3, 1], 0, 1, [0, 1])
    assert _fitted(result) == [Fraction(4), Fraction(4), Fraction(9)]
    assert _total(result) == 0
    assert _block_ranges(result) == ((0, 1), (2, 2))

    # 平台内锚点权重再大，平台值仍恒等于锚点读数（等式约束）
    result = fit_isotonic_plateau([1, 5, 9], [1, 10**6, 1], 0, 2, [1])
    assert _fitted(result) == [Fraction(5)] * 3


def test_plateau_blocks_merge_across_platform_boundary():
    # 平台共享值与左侧邻点 fitted 相等：跨平台边界合并为一个分块
    result = fit_isotonic_plateau([2, 1, 3, 10], [1, 1, 1, 1], 1, 2)
    assert _fitted(result) == [Fraction(2), Fraction(2), Fraction(2), Fraction(10)]
    assert _block_ranges(result) == ((0, 2), (3, 3))
    assert _total(result) == Fraction(2)


def test_plateau_all_anchored_inside_platform_matches_anchored_fit():
    # 平台内读数全等且全部锚定：平台段校正退化为锚点校正，逐项一致
    rng = random.Random(555)
    checked = 0
    while checked < 150:
        n = rng.randint(2, 9)
        readings = [rng.randint(0, 6) for _ in range(n)]
        weights = [rng.randint(1, 4) for _ in range(n)]
        platform_start = rng.randrange(n - 1)
        platform_end = rng.randrange(platform_start + 1, n)
        common = readings[platform_start]
        for i in range(platform_start, platform_end + 1):
            readings[i] = common
        outside = [i for i in range(n) if not platform_start <= i <= platform_end]
        k = rng.randint(0, len(outside))
        anchors = sorted(
            rng.sample(outside, k) + list(range(platform_start, platform_end + 1))
        )
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        checked += 1
        assert fit_isotonic_plateau(
            readings, weights, platform_start, platform_end, anchors
        ) == fit_isotonic_anchored(readings, weights, anchors)


# ---------------------------------------------------------------------------
# 不可行：平台内锚点读数不一致 / 锚点读数下降，报告最早冲突对
# ---------------------------------------------------------------------------


def test_infeasible_plateau_anchors_disagree():
    # 平台内锚点读数不同（递增）：平台共享值不可满足
    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 2, [0, 2])
    assert (excinfo.value.left, excinfo.value.right) == (0, 2)

    # 多对不一致时报告按观测顺序最早的一对相邻平台内锚点
    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([5, 5, 9], [1, 1, 1], 0, 2, [0, 1, 2])
    assert (excinfo.value.left, excinfo.value.right) == (1, 2)

    with pytest.raises(InfeasiblePlateauError) as excinfo:
        fit_isotonic_plateau([1, 5, 5, 9], [1, 1, 1, 1], 0, 3, [0, 1, 2, 3])
    assert (excinfo.value.left, excinfo.value.right) == (0, 1)

    # 平台外锚点不影响平台一致性判定
    result = fit_isotonic_plateau([0, 5, 5, 9], [1, 1, 1, 1], 1, 2, [0, 1, 2, 3])
    assert _fitted(result) == [Fraction(0), Fraction(5), Fraction(5), Fraction(9)]


def test_infeasible_anchors_decrease_inside_or_around_platform():
    # 平台内锚点读数下降：属于锚点顺序冲突（先于平台一致性报告）
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([3, 2, 1], [1, 1, 1], 0, 2, [0, 2])
    assert (excinfo.value.left, excinfo.value.right) == (0, 2)

    # 平台外锚点下降：报告最早冲突的相邻锚点对
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([7, 1, 9, 2, 5], [1] * 5, 1, 2, [0, 3, 4])
    assert (excinfo.value.left, excinfo.value.right) == (0, 3)

    # 锚点与平台内锚点之间下降
    with pytest.raises(InfeasibleAnchorsError) as excinfo:
        fit_isotonic_plateau([5, 1, 2], [1, 1, 1], 1, 2, [0, 1])
    assert (excinfo.value.left, excinfo.value.right) == (0, 1)


def test_infeasible_means_no_partial_fit():
    # 不可行时只抛异常，不存在任何部分拟合结果可返回
    with pytest.raises(InfeasiblePlateauError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 2, [0, 1])
    # 同一求解器修正输入后立即可用
    result = fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 1, [1])
    assert _fitted(result) == [Fraction(2), Fraction(2), Fraction(3)]


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def test_plateau_range_validation():
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], 0, 0)  # 单点平台
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], 1, 0)  # 起止颠倒
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], -1, 0)  # 起点越界
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1, 1], 0, 2)  # 终点越界
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2], [1], 0, 1)  # 长度不齐
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 1, [2, 2])  # 重复锚点
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 1, [3])  # 锚点越界
    with pytest.raises(ValueError):
        fit_isotonic_plateau([1, 2, 3], [1, 1, 1], 0, 1, [-1])


# ---------------------------------------------------------------------------
# 结构不变量：单调、平台共享、锚点原值、块覆盖、不可约、误差自洽、确定性
# ---------------------------------------------------------------------------


def test_plateau_invariants_on_random_instances():
    rng = random.Random(20260701)
    checked = 0
    while checked < 250:
        n = rng.randint(2, 30)
        readings = [rng.randint(0, 100) for _ in range(n)]
        weights = [rng.randint(1, 10) for _ in range(n)]
        platform_start = rng.randrange(n - 1)
        platform_end = rng.randrange(platform_start + 1, n)
        k = rng.randint(0, min(12, n))
        anchors = sorted(rng.sample(range(n), k))
        in_platform = [p for p in anchors if platform_start <= p <= platform_end]
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        if any(readings[p] != readings[in_platform[0]] for p in in_platform):
            continue
        checked += 1
        result = fit_isotonic_plateau(
            readings, weights, platform_start, platform_end, anchors
        )
        fitted = _fitted(result)

        # 保留全部点、非递减、平台各点共享同一值、锚点原值保留
        assert len(fitted) == n
        assert _non_decreasing(fitted)
        plateau_values = fitted[platform_start : platform_end + 1]
        assert len(set(plateau_values)) == 1
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
    rng = random.Random(77)
    for _ in range(30):
        n = rng.randint(2, 15)
        readings = [rng.randint(0, 30) for _ in range(n)]
        weights = [rng.randint(1, 9) for _ in range(n)]
        platform_start = rng.randrange(n - 1)
        platform_end = rng.randrange(platform_start + 1, n)
        k = rng.randint(0, min(4, n))
        anchors = sorted(rng.sample(range(n), k))
        in_platform = [p for p in anchors if platform_start <= p <= platform_end]
        if not _non_decreasing([readings[p] for p in anchors]):
            continue
        if any(readings[p] != readings[in_platform[0]] for p in in_platform):
            continue
        first = fit_isotonic_plateau(
            readings, weights, platform_start, platform_end, anchors
        )
        for _ in range(3):
            assert (
                fit_isotonic_plateau(
                    readings, weights, platform_start, platform_end, anchors
                )
                == first
            )
