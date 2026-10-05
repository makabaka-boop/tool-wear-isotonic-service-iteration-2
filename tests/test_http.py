"""HTTP 层测试：校验 422 规则、响应结构与端到端精确数值。"""

from __future__ import annotations

from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_payload(items):
    return {"observations": items}


def obs(id_, reading, weight=1, **extra):
    item = {"id": id_, "reading": reading, "weight": weight}
    item.update(extra)
    return item


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simple_correction_response():
    payload = make_payload(
        [
            obs("a", 3),
            obs("b", 1),
            obs("c", 2),
        ]
    )
    response = client.post("/correct", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # 3,1,2 等权 → 合并为一块，fitted 全部 2/1，总误差 2/1
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}}
    ]
    assert data["fitted"] == [
        {"id": "a", "fitted": {"numerator": 2, "denominator": 1}},
        {"id": "b", "fitted": {"numerator": 2, "denominator": 1}},
        {"id": "c", "fitted": {"numerator": 2, "denominator": 1}},
    ]
    assert data["total_error"] == {"numerator": 2, "denominator": 1}


def test_fraction_mean_is_reduced_and_order_preserved():
    # 权重 [3,1]：均值 (6+0)/4 = 3/2；逐点 id 与输入顺序一致，全部保留
    payload = make_payload([obs("x", 2, 3), obs("y", 0, 1)])
    response = client.post("/correct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 1, "mean": {"numerator": 3, "denominator": 2}}
    ]
    assert [p["id"] for p in data["fitted"]] == ["x", "y"]
    assert all(p["fitted"] == {"numerator": 3, "denominator": 2} for p in data["fitted"])
    assert data["total_error"] == {"numerator": 3, "denominator": 1}


def test_zero_readings_accepted():
    payload = make_payload([obs("a", 0, 1), obs("b", 0, 10**6)])
    response = client.post("/correct", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["blocks"][0]["mean"] == {"numerator": 0, "denominator": 1}
    assert data["total_error"] == {"numerator": 0, "denominator": 1}


def test_boundary_sizes_accepted():
    # 恰好 2 个
    r = client.post("/correct", json=make_payload([obs("a", 1), obs("b", 2)]))
    assert r.status_code == 200
    # 恰好 5000 个
    items = [obs(f"id-{i}", i % 7) for i in range(5000)]
    r = client.post("/correct", json=make_payload(items))
    assert r.status_code == 200
    data = r.json()
    # fitted 非递减，点数保留
    values = [Fraction(p["fitted"]["numerator"], p["fitted"]["denominator"]) for p in data["fitted"]]
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert len(values) == 5000
    # 块区间连续覆盖
    assert data["blocks"][0]["start_index"] == 0
    assert data["blocks"][-1]["end_index"] == 4999
    for prev, curr in zip(data["blocks"], data["blocks"][1:]):
        assert curr["start_index"] == prev["end_index"] + 1


def test_extreme_values_exact():
    payload = make_payload(
        [obs("hi", 10**9, 10**6), obs("lo", 0, 10**6)]
    )
    r = client.post("/correct", json=payload)
    assert r.status_code == 200
    mean = r.json()["blocks"][0]["mean"]
    swr = 10**9 * 10**6
    sw = 2 * 10**6
    assert Fraction(mean["numerator"], mean["denominator"]) == Fraction(swr, sw)


# ---------------------------------------------------------------------------
# 422：未知字段、重复 id、越界值、数量越界、类型错误
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # 观测过少 / 过多
        make_payload([obs("a", 1)]),
        make_payload([]),
        make_payload([obs(f"i{i}", 1) for i in range(5001)]),
        # reading 越界
        make_payload([obs("a", -1), obs("b", 1)]),
        make_payload([obs("a", 10**9 + 1), obs("b", 1)]),
        # weight 越界
        make_payload([obs("a", 1, 0), obs("b", 1)]),
        make_payload([obs("a", 1, 10**6 + 1), obs("b", 1)]),
        make_payload([obs("a", 1, -5), obs("b", 1)]),
        # 重复 id
        make_payload([obs("dup", 1), obs("dup", 2)]),
        make_payload(
            [obs("x", 1), obs("y", 2), obs("x", 3), obs("z", 4)]
        ),
        # 观测内未知字段
        make_payload([obs("a", 1, extra=99), obs("b", 2)]),
        # 顶层未知字段
        {**make_payload([obs("a", 1), obs("b", 2)]), "mode": "aggressive"},
        # 缺字段
        {"observations": [{"id": "a", "reading": 1}, {"id": "b", "reading": 2}]},
        {"observations": [{"reading": 1, "weight": 1}, {"reading": 2, "weight": 1}]},
        {"observations": [{"id": "a", "weight": 1}, {"id": "b", "weight": 1}]},
        # 空 id / 非 ASCII id
        make_payload([obs("", 1), obs("b", 2)]),
        make_payload([obs("刀具", 1), obs("b", 2)]),
        # 类型错误（strict：浮点冒充整数、布尔、字符串）
        make_payload([obs("a", 1.5), obs("b", 2)]),
        make_payload([obs("a", True), obs("b", 2)]),
        make_payload([obs("a", "1"), obs("b", 2)]),
        make_payload([obs("a", 1, 1.0), obs("b", 2, 1)]),
        make_payload([obs(1, 1), obs(2, 2)]),
        {"observations": {}},
        {"observations": "nope"},
        # 非 JSON 媒体类型由调用方在另一个测试中覆盖
    ],
)
def test_invalid_payloads_return_422(payload):
    response = client.post("/correct", json=payload)
    assert response.status_code == 422, response.text


def test_duplicate_id_check_is_order_independent_422():
    # 即使重复点本身"单调"，重复 id 仍拒绝
    payload = make_payload([obs("k", 1), obs("k", 1)])
    assert client.post("/correct", json=payload).status_code == 422


def test_non_dict_observation_422():
    r = client.post("/correct", json={"observations": [1, 2]})
    assert r.status_code == 422
    r = client.post("/correct", json={"observations": [["a", 1], ["b", 2]]})
    assert r.status_code == 422


def test_malformed_json_422():
    r = client.post(
        "/correct",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_wrong_media_type_422():
    r = client.post(
        "/correct",
        content="observations=something",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 端到端数值：与领域实现独立交叉核对（HTTP 层直接用 Fraction 重算）
# ---------------------------------------------------------------------------


def test_end_to_end_matches_independent_recomputation():
    readings = [10, 4, 8, 2, 9, 3]
    weights = [3, 1, 2, 4, 1, 5]
    payload = make_payload(
        [obs(f"p{i}", readings[i], weights[i]) for i in range(len(readings))]
    )
    r = client.post("/correct", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # 用最简 PAVA 参考（仅做均值比较）独立重算逐点曲线
    fitted_ref = _reference_pava(readings, weights)

    http_values = [
        Fraction(p["fitted"]["numerator"], p["fitted"]["denominator"])
        for p in data["fitted"]
    ]
    assert http_values == fitted_ref
    assert [p["id"] for p in data["fitted"]] == [f"p{i}" for i in range(6)]

    # 块结构与逐点曲线一致
    for block in data["blocks"]:
        m = Fraction(block["mean"]["numerator"], block["mean"]["denominator"])
        for i in range(block["start_index"], block["end_index"] + 1):
            assert http_values[i] == m

    # 总误差独立重算
    total = sum(
        weights[i] * (http_values[i] - readings[i]) ** 2
        for i in range(len(readings))
    )
    assert Fraction(
        data["total_error"]["numerator"], data["total_error"]["denominator"]
    ) == total


def _reference_pava(readings, weights):
    pools = []
    for i, (r, w) in enumerate(zip(readings, weights)):
        pools.append([i, i, r * w, w])
        while len(pools) >= 2:
            left, right = pools[-2], pools[-1]
            if Fraction(left[2], left[3]) < Fraction(right[2], right[3]):
                break
            pools.pop()
            pools[-1][1] = right[1]
            pools[-1][2] += right[2]
            pools[-1][3] += right[3]
    out = [Fraction(0)] * len(readings)
    for start, end, swr, sw in pools:
        for i in range(start, end + 1):
            out[i] = Fraction(swr, sw)
    return out


# ---------------------------------------------------------------------------
# 固定观测校正 /correct/anchored：正常路径
# ---------------------------------------------------------------------------


def anchored_payload(items, anchors):
    return {"observations": items, "anchors": anchors}


def fitted_values(data):
    return [
        Fraction(p["fitted"]["numerator"], p["fitted"]["denominator"])
        for p in data["fitted"]
    ]


def test_anchored_basic_response_marks_fixed_points():
    payload = anchored_payload([obs("a", 3), obs("b", 1), obs("c", 2)], ["b"])
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # 锚点 b 原值保留为 1；a 被上界 1 压回，c 保持 2
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 1, "mean": {"numerator": 1, "denominator": 1}},
        {"start_index": 2, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}},
    ]
    assert data["fitted"] == [
        {"id": "a", "fitted": {"numerator": 1, "denominator": 1}, "fixed": False},
        {"id": "b", "fitted": {"numerator": 1, "denominator": 1}, "fixed": True},
        {"id": "c", "fitted": {"numerator": 2, "denominator": 1}, "fixed": False},
    ]
    assert data["total_error"] == {"numerator": 4, "denominator": 1}


def test_anchored_equal_values_merge_across_anchor():
    # 两侧段拟合值都等于锚点值：跨锚点只作为一个分块
    payload = anchored_payload([obs("a", 9), obs("b", 5), obs("c", 1)], ["b"])
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 2, "mean": {"numerator": 5, "denominator": 1}}
    ]
    assert fitted_values(data) == [Fraction(5)] * 3
    assert [p["fixed"] for p in data["fitted"]] == [False, True, False]
    assert data["total_error"] == {"numerator": 32, "denominator": 1}


def test_anchored_equal_anchor_readings():
    payload = anchored_payload(
        [obs("a", 0), obs("b", 10), obs("c", 0), obs("d", 10)],
        ["b", "d"],
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert fitted_values(data) == [Fraction(0), Fraction(10), Fraction(10), Fraction(10)]
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 0, "mean": {"numerator": 0, "denominator": 1}},
        {"start_index": 1, "end_index": 3, "mean": {"numerator": 10, "denominator": 1}},
    ]
    assert data["total_error"] == {"numerator": 100, "denominator": 1}


def test_anchored_endpoint_anchors():
    payload = anchored_payload([obs("a", 5), obs("b", 1), obs("c", 9)], ["a", "c"])
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert fitted_values(data) == [Fraction(5), Fraction(5), Fraction(9)]
    assert [p["fixed"] for p in data["fitted"]] == [True, False, True]
    assert data["total_error"] == {"numerator": 16, "denominator": 1}


def test_anchored_all_fixed():
    payload = anchored_payload(
        [obs("a", 1), obs("b", 2), obs("c", 2), obs("d", 5)],
        ["a", "b", "c", "d"],
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert all(p["fixed"] for p in data["fitted"])
    assert fitted_values(data) == [Fraction(1), Fraction(2), Fraction(2), Fraction(5)]
    assert data["total_error"] == {"numerator": 0, "denominator": 1}
    # 等值的相邻锚点合并为同一块
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 0, "mean": {"numerator": 1, "denominator": 1}},
        {"start_index": 1, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}},
        {"start_index": 3, "end_index": 3, "mean": {"numerator": 5, "denominator": 1}},
    ]


def test_anchored_max_anchor_count_accepted():
    items = [obs(f"p{i}", i) for i in range(12)]
    r = client.post(
        "/correct/anchored",
        json=anchored_payload(items, [f"p{i}" for i in range(12)]),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert all(p["fixed"] for p in data["fitted"])
    assert fitted_values(data) == [Fraction(i) for i in range(12)]


def test_anchored_anchor_weight_does_not_move_anchor():
    # 锚点权重再大，fitted 仍恒等于其 reading（等式约束而非大权重近似）
    payload = anchored_payload(
        [obs("a", 3), obs("b", 1, 10**6), obs("c", 2)], ["b"]
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    assert fitted_values(r.json()) == [Fraction(1), Fraction(1), Fraction(2)]


def test_anchored_end_to_end_matches_independent_recomputation():
    # 与测试内独立的有界枚举神谕交叉核对（HTTP 层）
    from .oracle import oracle_anchored_fit

    readings = [10, 4, 8, 2, 9, 3]
    weights = [3, 1, 2, 4, 1, 5]
    anchors = ["p1", "p4"]  # 位置 1（值 4）与位置 4（值 9）
    payload = anchored_payload(
        [obs(f"p{i}", readings[i], weights[i]) for i in range(6)], anchors
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    expected_fitted, expected_error = oracle_anchored_fit(
        readings, weights, [1, 4]
    )
    assert fitted_values(data) == expected_fitted
    assert Fraction(
        data["total_error"]["numerator"], data["total_error"]["denominator"]
    ) == expected_error
    assert [p["id"] for p in data["fitted"]] == [f"p{i}" for i in range(6)]
    assert [p["fixed"] for p in data["fitted"]] == [
        False, True, False, False, True, False,
    ]
    # 块结构与逐点曲线一致
    for block in data["blocks"]:
        m = Fraction(block["mean"]["numerator"], block["mean"]["denominator"])
        for i in range(block["start_index"], block["end_index"] + 1):
            assert fitted_values(data)[i] == m


# ---------------------------------------------------------------------------
# 固定观测校正：409 INFEASIBLE_ANCHORS
# ---------------------------------------------------------------------------


def test_anchored_infeasible_returns_409_with_earliest_conflict():
    payload = anchored_payload(
        [obs("a", 1), obs("b", 5), obs("c", 4), obs("d", 0)],
        ["a", "b", "c", "d"],
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 409, r.text
    body = r.json()
    # 最早冲突的相邻锚点对是 (b, c)：(5,4) 先于 (4,0)
    assert body["detail"]["code"] == "INFEASIBLE_ANCHORS"
    assert body["detail"]["conflict"] == ["b", "c"]
    # 不发布部分拟合
    assert "blocks" not in body
    assert "fitted" not in body
    assert "total_error" not in body


def test_anchored_infeasible_order_follows_observation_order():
    # 锚点数组乱序给出，冲突判定按观测顺序
    payload = anchored_payload(
        [obs("a", 5), obs("b", 8), obs("c", 1)], ["c", "a"]
    )
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 409
    assert r.json()["detail"]["conflict"] == ["a", "c"]


def test_anchored_all_fixed_decreasing_is_infeasible():
    payload = anchored_payload([obs("a", 3), obs("b", 1)], ["a", "b"])
    r = client.post("/correct/anchored", json=payload)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "INFEASIBLE_ANCHORS"
    assert r.json()["detail"]["conflict"] == ["a", "b"]


def test_service_usable_after_infeasible_request():
    bad = anchored_payload([obs("a", 3), obs("b", 1)], ["a", "b"])
    assert client.post("/correct/anchored", json=bad).status_code == 409

    # 原 /correct 不受影响，结果逐项不变
    r = client.post("/correct", json=make_payload([obs("a", 3), obs("b", 1)]))
    assert r.status_code == 200
    assert r.json() == {
        "blocks": [
            {
                "start_index": 0,
                "end_index": 1,
                "mean": {"numerator": 2, "denominator": 1},
            }
        ],
        "fitted": [
            {"id": "a", "fitted": {"numerator": 2, "denominator": 1}},
            {"id": "b", "fitted": {"numerator": 2, "denominator": 1}},
        ],
        "total_error": {"numerator": 2, "denominator": 1},
    }

    # 修正锚点后同端点正常
    ok = anchored_payload([obs("a", 3), obs("b", 1)], ["b"])
    r = client.post("/correct/anchored", json=ok)
    assert r.status_code == 200, r.text
    assert fitted_values(r.json()) == [Fraction(1), Fraction(1)]


# ---------------------------------------------------------------------------
# 固定观测校正：422（未知/重复锚点、数量越界、类型错误、未知字段）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # 未知锚点 id
        anchored_payload([obs("a", 1), obs("b", 2)], ["zzz"]),
        anchored_payload([obs("a", 1), obs("b", 2)], ["a", "zzz"]),
        anchored_payload([obs("a", 1), obs("b", 2)], [""]),
        # 重复锚点 id
        anchored_payload([obs("a", 1), obs("b", 2)], ["a", "a"]),
        anchored_payload([obs("a", 1), obs("b", 2)], ["b", "a", "b"]),
        # 锚点数量越界：0 个 / 13 个
        anchored_payload([obs("a", 1), obs("b", 2)], []),
        anchored_payload(
            [obs(f"p{i}", i) for i in range(13)], [f"p{i}" for i in range(13)]
        ),
        # 缺 anchors 字段
        {"observations": [obs("a", 1), obs("b", 2)]},
        # 顶层未知字段
        {**anchored_payload([obs("a", 1), obs("b", 2)], ["a"]), "mode": "x"},
        # 锚点类型错误
        anchored_payload([obs("a", 1), obs("b", 2)], "a"),
        anchored_payload([obs("a", 1), obs("b", 2)], [1]),
        anchored_payload([obs("a", 1), obs("b", 2)], [["a"]]),
        anchored_payload([obs("a", 1), obs("b", 2)], [None]),
        # 观测侧违规依旧 422
        anchored_payload([obs("a", 1), obs("a", 2)], ["a"]),
        anchored_payload([obs("a", -1), obs("b", 2)], ["a"]),
        anchored_payload([obs("a", 1, extra=1), obs("b", 2)], ["a"]),
    ],
)
def test_anchored_invalid_payloads_return_422(payload):
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 422, response.text


def test_anchored_malformed_json_422():
    r = client.post(
        "/correct/anchored",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_correct_unchanged_by_anchored_feature():
    # 原 /correct 行为逐项不变（同一批观测，带不带锚点入口互不影响）
    payload = make_payload([obs("a", 3), obs("b", 1), obs("c", 2)])
    r = client.post("/correct", json=payload)
    assert r.status_code == 200
    assert r.json() == {
        "blocks": [
            {
                "start_index": 0,
                "end_index": 2,
                "mean": {"numerator": 2, "denominator": 1},
            }
        ],
        "fitted": [
            {"id": "a", "fitted": {"numerator": 2, "denominator": 1}},
            {"id": "b", "fitted": {"numerator": 2, "denominator": 1}},
            {"id": "c", "fitted": {"numerator": 2, "denominator": 1}},
        ],
        "total_error": {"numerator": 2, "denominator": 1},
    }
