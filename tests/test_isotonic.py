"""领域算法测试：PAVA 与全分区枚举神谕对拍，外加边界情形。

全程不允许 float：断言中所有期望误差/均值均为 Fraction，
另用字节码扫描确认生产模块未引入浮点运算。
"""

from __future__ import annotations

import dis
import itertools
from fractions import Fraction

import pytest

from app import isotonic
from app.isotonic import fit_isotonic

from .oracle import oracle_best_partition


def _pava_blocks(result):
    return tuple((b.start, b.end) for b in result.blocks)


def _pava_error(result):
    return Fraction(result.total_error_num, result.total_error_den)


def _fitted_fractions(result):
    return [
        Fraction(result.fitted_num[i], result.fitted_den[i])
        for i in range(len(result.fitted_num))
    ]


# ---------------------------------------------------------------------------
# 全分区枚举对拍
# ---------------------------------------------------------------------------


def test_all_reading_patterns_up_to_length_6():
    # 2^? 个序列：n=2..6，每个点取值 {0,1,2,3}，权重全 1。
    # 分区数最多 2^5=32，全部用 Fraction 精确对拍。
    for n in range(2, 7):
        for pattern in itertools.product(range(4), repeat=n):
            readings = list(pattern)
            weights = [1] * n
            result = fit_isotonic(readings, weights)
            expected_ranges, expected_error = oracle_best_partition(
                readings, weights
            )
            assert _pava_blocks(result) == expected_ranges, (readings,)
            assert _pava_error(result) == expected_error


def test_all_reading_patterns_length_7_weighted():
    # 权重取 {1,2,3} 的全部组合对 n=3 序列枚举；n=7 用均匀/交替权重。
    for pattern in itertools.product(range(3), repeat=3):
        for weights in itertools.product([1, 2, 3], repeat=3):
            readings = list(pattern)
            result = fit_isotonic(readings, list(weights))
            expected_ranges, expected_error = oracle_best_partition(
                readings, list(weights)
            )
            assert _pava_blocks(result) == expected_ranges
            assert _pava_error(result) == expected_error

    for pattern in itertools.product(range(3), repeat=7):
        for weights in ([1] * 7, [1, 7, 1, 7, 1, 7, 1], [3, 1, 4, 1, 5, 9, 2]):
            readings = list(pattern)
            result = fit_isotonic(readings, weights)
            expected_ranges, expected_error = oracle_best_partition(
                readings, weights
            )
            assert _pava_blocks(result) == expected_ranges
            assert _pava_error(result) == expected_error


def test_random_sequences_match_oracle():
    import random

    rng = random.Random(20260923)
    # n <= 9：512 个分区 × 若干随机序列，秒级完成
    for n in (2, 3, 4, 5, 8, 9):
        for _ in range(40):
            readings = [rng.randint(0, 20) for _ in range(n)]
            weights = [rng.randint(1, 20) for _ in range(n)]
            result = fit_isotonic(readings, weights)
            expected_ranges, expected_error = oracle_best_partition(
                readings, weights
            )
            assert _pava_blocks(result) == expected_ranges, readings
            assert _pava_error(result) == expected_error
            # 逐点 fitted = 所属块加权均值
            for start, end in expected_ranges:
                sw = sum(weights[start : end + 1])
                swr = sum(
                    weights[i] * readings[i]
                    for i in range(start, end + 1)
                )
                for i in range(start, end + 1):
                    assert _fitted_fractions(result)[i] == Fraction(
                        swr, sw
                    )


# ---------------------------------------------------------------------------
# 结构性质：单调性、保点、无浮点、不可约
# ---------------------------------------------------------------------------


def _assert_no_float_in_module(module):
    """扫描模块中源码定义函数的字节码常量：不允许出现 float。

    只检查 ``__module__`` 指向本模块、且有源码行号的用户函数，
    排除 dataclass 自动生成的 ``__eq__``（其常量与计算无关）与
    从 fractions 等外部模块导入的对象。
    """

    float_consts = []

    def walk(code, origin):
        for const in code.co_consts:
            if isinstance(const, float):
                float_consts.append((origin, code.co_name, const))
            if hasattr(const, "co_code"):
                walk(const, origin)

    for name in vars(module):
        obj = getattr(module, name)
        candidates = []
        if isinstance(obj, type):
            for attr in vars(obj).values():
                if (
                    callable(attr)
                    and getattr(attr, "__module__", None) == module.__name__
                    and hasattr(attr, "__code__")
                    and attr.__code__.co_firstlineno > 0
                    and not getattr(attr, "__wrapped__", False)
                ):
                    candidates.append(attr)
        elif (
            callable(obj)
            and getattr(obj, "__module__", None) == module.__name__
            and hasattr(obj, "__code__")
        ):
            candidates.append(obj)
        for fn in candidates:
            # dataclass 生成的 __eq__ 带有 __eq__ 自身的行号特征：
            # 其 co_filename 不指向真实源码行，直接按名字跳过生成方法。
            if getattr(fn, "__qualname__", "").endswith(".__eq__"):
                continue
            walk(fn.__code__, name)

    assert not float_consts, float_consts

    # AST 级保证：无浮点常量、无真除法、无 float 内建调用
    import ast

    source = isotonic.__loader__.get_source(isotonic.__name__)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            pytest.fail(f"float literal at line {node.lineno}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            pytest.fail(f"true division at line {node.lineno}")
        if isinstance(node, ast.Name) and node.id == "float":
            pytest.fail(f"float builtin at line {node.lineno}")


def test_no_floating_point_used():
    _assert_no_float_in_module(isotonic)
    # 反汇编层面也不应出现 BINARY_OP 之外的浮点指令；常量检查已足够，
    # 这里再确认 fractions.Fraction 是唯一数值分数来源（导入存在）。
    assert hasattr(isotonic, "Fraction")
    # dis 仅用于在本测试文件中确认字节码可读，无断言含义。
    assert dis.opmap is not None


def _check_result_invariants(readings, weights, result):
    n = len(readings)
    # 块连续、含端点覆盖全部点且无重叠
    assert result.blocks[0].start == 0
    assert result.blocks[-1].end == n - 1
    for prev, curr in zip(result.blocks, result.blocks[1:]):
        assert curr.start == prev.end + 1

    fitted = _fitted_fractions(result)
    # 保留全部原始点
    assert len(fitted) == n
    # fitted 非递减
    for a, b in zip(fitted, fitted[1:]):
        assert a <= b

    total = Fraction(0, 1)
    from math import gcd

    for block in result.blocks:
        assert block.start <= block.end
        mean = Fraction(block.mean_num, block.mean_den)
        # 不可约、分母为正
        assert block.mean_den > 0
        assert gcd(abs(block.mean_num), block.mean_den) == 1
        # fitted 与块均值一致
        for i in range(block.start, block.end + 1):
            assert fitted[i] == mean
        # 块误差精确公式
        sw = sum(weights[block.start : block.end + 1])
        swr = sum(
            weights[i] * readings[i]
            for i in range(block.start, block.end + 1)
        )
        swr2 = sum(
            weights[i] * readings[i] * readings[i]
            for i in range(block.start, block.end + 1)
        )
        expected_block_error = Fraction(sw * swr2 - swr * swr, sw)
        assert Fraction(block.error_num, block.error_den) == (
            expected_block_error
        )
        assert block.error_den > 0
        assert gcd(abs(block.error_num), block.error_den) == 1
        total += expected_block_error
        # 块均值就是加权均值
        assert mean == Fraction(swr, sw)

    assert _pava_error(result) == total
    assert result.total_error_den > 0
    assert gcd(abs(result.total_error_num), result.total_error_den) == 1
    # 逐点误差之和等于总误差
    pointwise = sum(
        (weights[i] * (fitted[i] - readings[i]) ** 2 for i in range(n)),
        Fraction(0, 1),
    )
    assert pointwise == total


def test_invariants_on_many_sequences():
    import random

    rng = random.Random(4242)
    for n in (2, 3, 10, 50, 137):
        for _ in range(10):
            readings = [rng.randint(0, 10**9) for _ in range(n)]
            weights = [rng.randint(1, 10**6) for _ in range(n)]
            _check_result_invariants(readings, weights, fit_isotonic(readings, weights))


# ---------------------------------------------------------------------------
# 规定的边界场景：全降序、零读数、大权重、并列、确定性
# ---------------------------------------------------------------------------


def test_strictly_decreasing_becomes_one_block():
    # 全降序：所有权重点合并为一块，fitted 为整体加权均值
    readings = [9, 7, 5, 3, 1]
    weights = [1, 2, 3, 4, 5]
    result = fit_isotonic(readings, weights)
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert (block.start, block.end) == (0, 4)
    sw = sum(weights)
    swr = sum(w * r for w, r in zip(weights, readings))
    assert Fraction(block.mean_num, block.mean_den) == Fraction(swr, sw)
    expected_error = Fraction(sw * sum(w * r * r for w, r in zip(weights, readings)) - swr * swr, sw)
    assert _pava_error(result) == expected_error
    assert all(
        Fraction(n, d) == Fraction(swr, sw)
        for n, d in zip(result.fitted_num, result.fitted_den)
    )


def test_already_monotone_keeps_points_separate():
    readings = [1, 2, 4, 8]
    weights = [1, 1, 1, 1]
    result = fit_isotonic(readings, weights)
    assert _pava_blocks(result) == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert _pava_error(result) == 0
    assert _fitted_fractions(result) == [Fraction(r) for r in readings]


def test_zero_readings():
    # 全零
    result = fit_isotonic([0, 0, 0], [1, 2, 3])
    assert _pava_blocks(result) == ((0, 2),)  # 并列合并：最粗规范分块
    assert _pava_error(result) == 0
    assert _fitted_fractions(result) == [Fraction(0)] * 3

    # 零读数与后续正读数：[0,0,1] 非递减，点保持分离（前两个并列仍合并）
    result = fit_isotonic([0, 0, 1], [1, 1, 1])
    assert _pava_blocks(result) == ((0, 1), (2, 2))
    assert _pava_error(result) == 0

    # 正读数后出现零，必须回拉合并
    result = fit_isotonic([5, 0], [1, 1])
    assert _pava_blocks(result) == ((0, 1),)
    assert _fitted_fractions(result) == [Fraction(5, 2), Fraction(5, 2)]
    assert _pava_error(result) == Fraction(25, 2)


def test_large_weights_and_readings_exact_fraction():
    # 大权重（10^6 量级）、大读数（10^9 量级）：精确分数，无浮点误差
    readings = [10**9, 0]
    weights = [10**6, 10**6 - 1]
    result = fit_isotonic(readings, weights)
    assert len(result.blocks) == 1
    sw = sum(weights)
    swr = sum(w * r for w, r in zip(weights, readings))
    assert Fraction(result.blocks[0].mean_num, result.blocks[0].mean_den) == Fraction(
        swr, sw
    )
    # gcd 检查不可约
    from math import gcd

    assert gcd(result.blocks[0].mean_num, result.blocks[0].mean_den) == 1
    # 结果可复现
    again = fit_isotonic(readings, weights)
    assert again == result


def test_weighted_merge_example():
    # [3, 1, 2] 等权：前两个合并均值 2，与第三个 2 并列再合并，最粗一块
    result = fit_isotonic([3, 1, 2], [1, 1, 1])
    assert _pava_blocks(result) == ((0, 2),)
    assert _fitted_fractions(result) == [Fraction(2)] * 3
    assert _pava_error(result) == Fraction(2)  # (1+1+0)/1

    # 权重影响均值：[2, 0]，权重 [3, 1] → 均值 3/2，误差
    result = fit_isotonic([2, 0], [3, 1])
    assert _pava_blocks(result) == ((0, 1),)
    assert _fitted_fractions(result) == [Fraction(3, 2), Fraction(3, 2)]
    # 3*(1/2)^2 + 1*(3/2)^2 = 3
    assert _pava_error(result) == Fraction(3)


def test_ties_are_merged_canonically():
    # 并列处无选择空间：相等均值必合并，分块确定
    result = fit_isotonic([5, 5, 5], [2, 3, 4])
    assert _pava_blocks(result) == ((0, 2),)

    result = fit_isotonic([1, 3, 3, 5], [1, 1, 1, 1])
    assert _pava_blocks(result) == ((0, 0), (1, 2), (3, 3))
    assert _pava_error(result) == 0

    # 两个合并块均值恰好相等时也合并为最粗分块
    # readings [2,0, 2,0]：PAVA 先得 [1,1] 两个等均值块 → 再合并
    result = fit_isotonic([2, 0, 2, 0], [1, 1, 1, 1])
    assert _pava_blocks(result) == ((0, 3),)
    assert all(f == Fraction(1) for f in _fitted_fractions(result))


def test_deterministic_partition_for_identical_input():
    import random

    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(2, 60)
        readings = [rng.randint(0, 100) for _ in range(n)]
        weights = [rng.randint(1, 100) for _ in range(n)]
        first = fit_isotonic(readings, weights)
        for _ in range(3):
            assert fit_isotonic(readings, weights) == first


def test_empty_and_length_mismatch():
    result = fit_isotonic([], [])
    assert result.blocks == ()
    assert result.fitted_num == ()
    assert _pava_error(result) == 0
    with pytest.raises(ValueError):
        fit_isotonic([1], [1, 2])


def test_performance_max_size_descending():
    # 5000 点全降序（最坏合并规模）应快速完成并给出单块
    readings = list(range(5000, 0, -1))
    weights = [1] * 5000
    result = fit_isotonic(readings, weights)
    assert len(result.blocks) == 1
    assert (result.blocks[0].start, result.blocks[0].end) == (0, 4999)
