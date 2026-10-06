import os
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    os.environ["DB_PATH"] = path
    from app import db
    db.reset()
    from app.main import app
    with TestClient(app) as c:
        yield c
    db.reset()
    Path(path).unlink(missing_ok=True)


def add_cell(client, cell_id, chemistry="LFP", capacity=100.0, resistance=2.0, version="T1"):
    r = client.post("/cells", json={
        "cell_id": cell_id, "chemistry": chemistry, "capacity": capacity,
        "resistance": resistance, "test_version": version,
    })
    assert r.status_code == 201, r.text
    return r


def seed_six(client):
    # 6 节 LFP：容量两两差 <= 5，内阻差 <= 0.5
    data = [
        ("C001", 100.0, 2.0), ("C002", 99.0, 2.1), ("C003", 98.0, 2.2),
        ("C004", 97.0, 2.3), ("C005", 96.0, 2.4), ("C006", 95.5, 2.5),
    ]
    for cid, cap, res in data:
        add_cell(client, cid, capacity=cap, resistance=res)


def quarantine(client, cell_id, reason="检出漏液"):
    r = client.post(f"/cells/{cell_id}/quarantine", json={"reason": reason})
    assert r.status_code == 201, r.text
    return r


def confirm_payload(client, series_count=4, chemistry=None):
    req = {
        "series_count": series_count, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }
    if chemistry is not None:
        req["chemistry"] = chemistry
    pre = client.post("/preselect", json=req).json()
    assert pre["feasible"] is True
    return {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                      for c in pre["cells"]]}, pre


def test_register_limit_12(client):
    for i in range(12):
        add_cell(client, f"C{i:03d}")
    r = client.post("/cells", json={
        "cell_id": "C999", "chemistry": "LFP", "capacity": 1.0,
        "resistance": 1.0, "test_version": "T1",
    })
    assert r.status_code == 409
    assert "12" in r.json()["detail"]


def test_duplicate_cell_rejected(client):
    add_cell(client, "C001")
    r = client.post("/cells", json={
        "cell_id": "C001", "chemistry": "LFP", "capacity": 1.0,
        "resistance": 1.0, "test_version": "T1",
    })
    assert r.status_code == 409


def test_preselect_maximizes_min_capacity(client):
    seed_six(client)
    r = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    body = r.json()
    assert body["feasible"] is True
    # 最小容量最大的 4 节组合是 C001..C004（min=97）
    assert body["cell_ids"] == ["C001", "C002", "C003", "C004"]
    assert body["limited_capacity"] == 97.0
    assert all(c["test_version"] == "T1" for c in body["cells"])


def test_preselect_tie_break_lexicographic(client):
    # 4 节中任取 3 节最小容量都是 99：同值时按 ID 排序序列字典序取最小
    add_cell(client, "B001", capacity=100.0, resistance=2.0)
    add_cell(client, "B002", capacity=99.0, resistance=2.0)
    add_cell(client, "A001", capacity=100.0, resistance=2.0)
    add_cell(client, "A002", capacity=99.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 1.0, "max_resistance_diff": 0.5,
    })
    body = r.json()
    assert body["feasible"] is True
    assert body["limited_capacity"] == 99.0
    assert body["cell_ids"] == ["A001", "A002", "B001"]


def test_preselect_chemistry_isolated(client):
    seed_six(client)
    add_cell(client, "N001", chemistry="NCM", capacity=100.0, resistance=2.0)
    add_cell(client, "N002", chemistry="NCM", capacity=99.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5, "chemistry": "NCM",
    })
    body = r.json()
    assert body["feasible"] is False
    assert "不足 3 节" in body["reason"]


def test_preselect_infeasible_reasons(client):
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.json()["feasible"] is False
    assert "没有可用的未占用电芯" in r.json()["reason"]

    # 容量全部不达标
    add_cell(client, "C001", capacity=50.0)
    add_cell(client, "C002", capacity=50.0)
    add_cell(client, "C003", capacity=50.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert "最低单芯容量" in r.json()["reason"]

    # 数量够但差值约束不满足
    add_cell(client, "C004", capacity=200.0, resistance=9.0)
    add_cell(client, "C005", capacity=100.0, resistance=2.0)
    add_cell(client, "C006", capacity=100.0, resistance=2.0)
    r = client.post("/preselect", json={
        "series_count": 3, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert "容量差上限" in r.json()["reason"] or "内阻差上限" in r.json()["reason"]


def test_confirm_success_and_snapshot(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 201, r.text
    group = r.json()
    assert group["group_id"] == 1
    assert group["limited_capacity"] == 97.0
    assert group["series_count"] == 4

    # 电芯被占用
    cells = client.get("/cells").json()["cells"]
    used = {c["cell_id"]: c for c in cells if c["occupied"]}
    assert set(used) == set(pre["cell_ids"])
    assert all(c["group_id"] == 1 for c in used.values())

    # 快照可按组编号查询
    snap = client.get("/groups/1").json()
    assert snap["cells"] == group["cells"]


def test_confirm_stale_version_rejected_atomically(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    # 预选后复测其中一节 -> 候选过期
    client.put("/cells/C002/test", json={
        "capacity": 99.0, "resistance": 2.1, "test_version": "T2"})
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "C002" in r.json()["detail"]
    # 整组拒绝：无任何部分占用，无组产生
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])
    assert client.get("/groups").json()["count"] == 0


def test_confirm_occupied_by_other_group_rejected(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    assert client.post("/groups/confirm", json=payload).status_code == 201
    # 别组已抢先占用 -> 再次确认同一候选被拒
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "占用" in r.json()["detail"]
    assert client.get("/groups").json()["count"] == 1


def test_revision_after_confirm_does_not_change_snapshot(client):
    seed_six(client)
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    payload = {"cells": [{"cell_id": c["cell_id"], "test_version": c["test_version"]}
                         for c in pre["cells"]]}
    group = client.post("/groups/confirm", json=payload).json()
    # 确认后复测已入组电芯：允许修订档案，但快照不变
    client.put("/cells/C001/test", json={
        "capacity": 88.0, "resistance": 3.0, "test_version": "T9"})
    snap = client.get(f"/groups/{group['group_id']}").json()
    c001 = next(c for c in snap["cells"] if c["cell_id"] == "C001")
    assert c001["capacity"] == 100.0
    assert c001["test_version"] == "T1"
    assert snap["limited_capacity"] == 97.0


def test_confirm_mixed_chemistry_rejected(client):
    add_cell(client, "C001", chemistry="LFP")
    add_cell(client, "C002", chemistry="LFP")
    add_cell(client, "C003", chemistry="NCM")
    r = client.post("/groups/confirm", json={"cells": [
        {"cell_id": "C001", "test_version": "T1"},
        {"cell_id": "C002", "test_version": "T1"},
        {"cell_id": "C003", "test_version": "T1"},
    ]})
    assert r.status_code == 409
    assert "化学体系" in r.json()["detail"]
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def test_series_count_validation(client):
    r = client.post("/preselect", json={
        "series_count": 2, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.status_code == 422
    r = client.post("/preselect", json={
        "series_count": 7, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    })
    assert r.status_code == 422


def _begin_uncommitted_group(conn):
    """在当前线程的连接上模拟一个尚未提交的确认事务。"""
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO groups (chemistry, series_count, limited_capacity, created_at)"
        " VALUES ('LFP', 4, 97.0, '2026-10-05T00:00:00+00:00')")
    gid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    for cid, cap, res in [("C001", 100.0, 2.0), ("C002", 99.0, 2.1),
                          ("C003", 98.0, 2.2), ("C004", 97.0, 2.3)]:
        conn.execute(
            "INSERT INTO group_cells (group_id, cell_id, capacity, resistance, test_version)"
            " VALUES (?, ?, ?, ?, 'T1')", (gid, cid, cap, res))
        conn.execute(
            "UPDATE cells SET occupied = 1, group_id = ? WHERE cell_id = ?", (gid, cid))
    return gid


def test_uncommitted_group_invisible_to_other_requests(client):
    """确认事务未提交时，其他请求看不到组记录、快照与临时占用；提交后整组一次性可见。"""
    seed_six(client)
    from app import db
    gid = _begin_uncommitted_group(db.get_conn())  # 本线程连接持有未提交事务

    # 未提交：组列表 / 按编号查询 / 电芯占用 / 预选都看不到事务的中间状态
    assert client.get("/groups").json()["count"] == 0
    assert client.get(f"/groups/{gid}").status_code == 404
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])
    pre = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    assert pre["feasible"] is True
    assert pre["cell_ids"] == ["C001", "C002", "C003", "C004"]

    # 提交后：整组（组记录 + 快照 + 占用）一次性可见
    db.get_conn().commit()
    assert client.get("/groups").json()["count"] == 1
    snap = client.get(f"/groups/{gid}").json()
    assert [c["cell_id"] for c in snap["cells"]] == ["C001", "C002", "C003", "C004"]
    used = {c["cell_id"] for c in client.get("/cells").json()["cells"] if c["occupied"]}
    assert used == {"C001", "C002", "C003", "C004"}


def test_rolled_back_group_leaves_no_trace(client):
    """确认回滚后，组记录、快照与占用对外毫无痕迹。"""
    seed_six(client)
    from app import db
    conn = db.get_conn()
    gid = _begin_uncommitted_group(conn)
    conn.rollback()
    assert client.get("/groups").json()["count"] == 0
    assert client.get(f"/groups/{gid}").status_code == 404
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def test_concurrent_confirm_and_reads_do_not_interfere(client):
    """并发确认同一候选：只有一个成功；并发查询只看到完整快照，互不干扰。"""
    seed_six(client)
    from app.main import app
    payload = {"cells": [{"cell_id": f"C00{i}", "test_version": "T1"} for i in range(1, 5)]}
    statuses = []
    broken = []

    def worker():
        with TestClient(app) as c:
            r = c.post("/groups/confirm", json=payload)
            statuses.append(r.status_code)
            for _ in range(10):
                # 组列表里出现的任何组都必须带完整快照
                for g in c.get("/groups").json()["groups"]:
                    if len(g["cells"]) != g["series_count"]:
                        broken.append(g)
                # 已提交的占用必须整组可见，不允许只看到部分占用
                members = [cell["cell_id"] for cell in c.get("/cells").json()["cells"]
                           if cell["occupied"]]
                if members and len(members) != 4:
                    broken.append(members)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses.count(201) == 1
    assert statuses.count(409) == 3
    assert not broken
    assert client.get("/groups").json()["count"] == 1


# ---------- 安全隔离 ----------


def test_quarantine_records_and_visible_on_cell(client):
    add_cell(client, "C001")
    body = quarantine(client, "C001", "检出漏液").json()
    assert body["cell_id"] == "C001"
    assert body["reason"] == "检出漏液"
    assert body["created_at"]

    cell = client.get("/cells/C001").json()
    assert cell["quarantined"] is True
    assert cell["quarantine_reason"] == "检出漏液"
    listed = {c["cell_id"]: c for c in client.get("/cells").json()["cells"]}
    assert listed["C001"]["quarantined"] is True


def test_quarantine_missing_cell_404(client):
    r = client.post("/cells/NOPE/quarantine", json={"reason": "漏液"})
    assert r.status_code == 404


def test_quarantine_duplicate_returns_original_record(client):
    add_cell(client, "C001")
    first = quarantine(client, "C001", "首次原因").json()
    r = client.post("/cells/C001/quarantine", json={"reason": "不同的原因"})
    assert r.status_code == 200
    assert r.json() == first  # 原记录不被覆盖


def test_revision_does_not_lift_quarantine(client):
    add_cell(client, "C001")
    quarantine(client, "C001", "电压异常")
    r = client.put("/cells/C001/test", json={
        "capacity": 90.0, "resistance": 3.0, "test_version": "T2"})
    assert r.status_code == 200
    cell = client.get("/cells/C001").json()
    assert cell["test_version"] == "T2"
    assert cell["quarantined"] is True  # 复测不能解除隔离
    assert cell["quarantine_reason"] == "电压异常"


def test_quarantine_irrevocable_at_db_level(client):
    add_cell(client, "C001")
    quarantine(client, "C001")
    import sqlite3
    from app import db
    conn = db.get_conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE quarantines SET reason = 'x' WHERE cell_id = 'C001'")
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM quarantines WHERE cell_id = 'C001'")
    conn.rollback()


def test_preselect_excludes_quarantined_cells(client):
    seed_six(client)
    quarantine(client, "C001")
    body = client.post("/preselect", json={
        "series_count": 4, "min_capacity": 90.0,
        "max_capacity_diff": 5.0, "max_resistance_diff": 0.5,
    }).json()
    assert body["feasible"] is True
    assert body["cell_ids"] == ["C002", "C003", "C004", "C005"]
    assert body["limited_capacity"] == 96.0


def test_confirm_rejects_quarantined_candidate_atomically(client):
    seed_six(client)
    payload, _ = confirm_payload(client)
    quarantine(client, "C003", "电压异常")  # 预选后、确认前隔离候选中的一节
    r = client.post("/groups/confirm", json=payload)
    assert r.status_code == 409
    assert "C003" in r.json()["detail"]
    # 整组拒绝：不产生组，不留任何部分占用
    assert client.get("/groups").json()["count"] == 0
    assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])


def test_quarantine_after_confirm_blocks_group_but_keeps_snapshot(client):
    seed_six(client)
    payload, pre = confirm_payload(client)
    group = client.post("/groups/confirm", json=payload).json()
    gid = group["group_id"]

    quarantine(client, "C002", "复检发现壳体破损")

    # 组列表与按编号查询都显示已阻断、涉及电芯与原因，原快照保留
    listed = client.get("/groups").json()["groups"][0]
    detail = client.get(f"/groups/{gid}").json()
    for g in (listed, detail):
        assert g["blocked"] is True
        assert g["blocked_cells"] == [
            {"cell_id": "C002", "reason": "复检发现壳体破损"}]
        assert g["cells"] == group["cells"]
        assert g["limited_capacity"] == 97.0
    # 组占用保留
    used = {c["cell_id"] for c in client.get("/cells").json()["cells"] if c["occupied"]}
    assert used == set(pre["cell_ids"])


def test_unaffected_group_stays_releasable(client):
    seed_six(client)
    for i in range(1, 5):  # 另一体系成第二组
        add_cell(client, f"N00{i}", chemistry="NCM", capacity=100.0, resistance=2.0)
    payload, _ = confirm_payload(client, chemistry="LFP")
    g1 = client.post("/groups/confirm", json=payload).json()
    r = client.post("/groups/confirm", json={"cells": [
        {"cell_id": f"N00{i}", "test_version": "T1"} for i in range(1, 5)]})
    assert r.status_code == 201, r.text
    g2 = r.json()

    quarantine(client, "C001")  # 只影响第一组

    groups = {g["group_id"]: g for g in client.get("/groups").json()["groups"]}
    assert groups[g1["group_id"]]["blocked"] is True
    assert groups[g2["group_id"]]["blocked"] is False
    assert groups[g2["group_id"]]["blocked_cells"] == []
    detail = client.get(f"/groups/{g2['group_id']}").json()
    assert detail["blocked"] is False


def test_concurrent_quarantine_and_confirm_consistent(client):
    """隔离与确认并发：按提交先后，要么确认被拒、要么新组被阻断，查询不见半成品。"""
    seed_six(client)
    from app.main import app
    payload = {"cells": [{"cell_id": f"C00{i}", "test_version": "T1"} for i in range(1, 5)]}
    results = {}
    broken = []

    def do_confirm():
        with TestClient(app) as c:
            results["confirm"] = c.post("/groups/confirm", json=payload).status_code

    def do_quarantine():
        with TestClient(app) as c:
            results["quarantine"] = c.post(
                "/cells/C002/quarantine", json={"reason": "并发隔离"}).status_code

    def do_reads():
        with TestClient(app) as c:
            for _ in range(10):
                for g in c.get("/groups").json()["groups"]:
                    # 任何可见的组都必须快照完整、阻断状态与成员一致
                    if len(g["cells"]) != g["series_count"]:
                        broken.append(g)
                    if g["blocked"] and not g["blocked_cells"]:
                        broken.append(g)

    threads = [threading.Thread(target=do_confirm),
               threading.Thread(target=do_quarantine),
               threading.Thread(target=do_reads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["quarantine"] == 201
    assert not broken
    groups = client.get("/groups").json()
    if results["confirm"] == 201:
        # 确认先提交：组产生，随后被隔离阻断
        assert groups["count"] == 1
        g = groups["groups"][0]
        assert g["blocked"] is True
        assert [b["cell_id"] for b in g["blocked_cells"]] == ["C002"]
        assert len(g["cells"]) == 4
    else:
        # 隔离先提交：候选被整组拒绝，不产生组或部分占用
        assert results["confirm"] == 409
        assert groups["count"] == 0
        assert all(not c["occupied"] for c in client.get("/cells").json()["cells"])
