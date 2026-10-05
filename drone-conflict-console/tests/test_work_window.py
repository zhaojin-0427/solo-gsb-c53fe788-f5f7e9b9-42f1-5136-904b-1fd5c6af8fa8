"""Temporary work-zone start-window validation.

Covers (pure logic, no HTTP layer):

* continuous-time cylinder intersection of a piecewise-linear track,
  including lateral/vertical boundary touches, waypoint-instant occupancy and
  a degenerate ``[T, T]`` interval, plus merging across waypoints;
* lifting closed occupancy intervals onto integer work starts: the closed
  work interval ``[s, s+d]`` intersecting occupancy in even one instant makes
  ``s`` unsafe (cylinder surface, waypoint, work start/end instants all count),
  without per-second sampling;
* candidate window ``[t, T-d]`` clipping, empty windows, overdue =>
  unconfirmable (no safe starts), early-time rejection when ``t`` precedes a
  reported actual take-off;
* reported sorties fly at the reported time, unreported ones with planned
  take-off >= t at the planned time;
* revision + reload consistency: amending a report and re-querying, and
  reloading the batch from SQLite, give identical results;
* an independent exact oracle (a second, direct quadratic-root implementation
  per integer start) agrees on random scenarios.

Run:  python tests/test_work_window.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.batch import (  # noqa: E402
    Report,
    WorkWindowReject,
    evaluate_work_window,
)
from app.geometry import cylinder_intersection_intervals  # noqa: E402
from app.model import Waypoint, make_sortie  # noqa: E402
from app.storage import Store  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def eastbound(z0=20, z1=20, y=0.0):
    """10 m/s east over [0, 20] at fixed y (linear z)."""
    return make_sortie("A", 1, 30, [
        Waypoint(0, y, z0, 0), Waypoint(100, y, 0.5 * (z0 + z1), 10),
        Waypoint(200, y, z1, 20)])


def evaluate(sorties, plan, reports, t, T, d, zc=(100.0, 0.0), r=10.0,
             z_lo=15.0, z_hi=25.0):
    return evaluate_work_window(
        sorties, plan, reports, t, T, d, zc[0], zc[1], r, z_lo, z_hi)


# ---------------------------------------------------------------------------
# Continuous cylinder geometry
# ---------------------------------------------------------------------------


def test_basic_cylinder_interval():
    a = eastbound()
    # disk r=10 centred at (100, 0): x in [90, 110] -> global q in [9, 11].
    check(cylinder_intersection_intervals(a, 0, 100, 0, 10, 15, 25)
          == [(9.0, 11.0)], "level pass-through occupies [9, 11]")
    # Take-off shift moves the whole interval rigidly.
    check(cylinder_intersection_intervals(a, 5, 100, 0, 10, 15, 25)
          == [(14.0, 16.0)], "take-off shift translates occupancy")
    # Well above / below the slab: nothing.
    check(cylinder_intersection_intervals(a, 0, 100, 0, 10, 21, 30) == [],
          "trajectory entirely below the cylinder slab")
    check(cylinder_intersection_intervals(a, 0, 100, 0, 10, 5, 19) == [],
          "slab strictly below the trajectory: no occupancy")


def test_boundary_and_point_occupancy():
    # Horizontal tangent: path along y=10, disk r=10 touches at one instant only.
    a = eastbound(y=10.0)
    ivs = cylinder_intersection_intervals(a, 0, 100, 0, 10, 15, 25)
    check(ivs == [(10.0, 10.0)],
          f"lateral tangency is a degenerate [10,10] interval, got {ivs}")

    # Cap boundary z_lo == z_hi == flight altitude: touching the cap counts.
    a = eastbound()
    check(cylinder_intersection_intervals(a, 0, 100, 0, 10, 20, 20)
          == [(9.0, 11.0)], "lying exactly on a cap plane is occupied")

    # Climbing track meets a single cap height exactly at the disk centre.
    climb = make_sortie("C", 1, 30, [
        Waypoint(100, 0, 0, 0), Waypoint(100, 0, 40, 20)])
    ivs = cylinder_intersection_intervals(climb, 0, 100, 0, 5, 20, 20)
    check(ivs == [(10.0, 10.0)],
          f"single cap crossing inside the disk is [10,10], got {ivs}")

    # Cap clipping: disk window q in [9, 11] (r=10), slab z <= 20 with
    # z(q)=2q reaches 20 at q=10 -> intersection is the head [9, 10].
    climb2 = make_sortie("C", 1, 30, [
        Waypoint(0, 0, 0, 0), Waypoint(200, 0, 40, 20)])
    ivs = cylinder_intersection_intervals(climb2, 0, 100, 0, 10, -100, 20)
    check(ivs == [(9.0, 10.0)],
          f"disk window clipped by a cap gives its head, got {ivs}")

    # True corner instant: a diagonal path reaches the disk boundary at the
    # same instant it meets the top cap (q=10: y=0, z=20, x=100).
    climb3 = make_sortie("C", 1, 30, [
        Waypoint(100, 20, 0, 0), Waypoint(100, 0, 40, 20)])
    ivs = cylinder_intersection_intervals(climb3, 0, 100, 0, 10, -100, 20)
    check(ivs == [(10.0, 10.0)],
          f"lateral boundary x cap corner is one instant, got {ivs}")


def test_merges_across_waypoint():
    a = eastbound()
    # r=50 spans q in [5, 15] across the interior waypoint at q=10: one row.
    ivs = cylinder_intersection_intervals(a, 0, 100, 0, 50, 15, 25)
    check(ivs == [(5.0, 15.0)],
          f"occupancy crossing a waypoint merges into one interval, got {ivs}")


def test_hovering_sortie():
    # Stationary in xyz for 10 s inside the cylinder: occupied the whole time.
    hov = make_sortie("H", 1, 30, [
        Waypoint(100, 0, 20, 0), Waypoint(100, 0, 20, 10)])
    check(cylinder_intersection_intervals(hov, 0, 100, 0, 10, 20, 20)
          == [(0.0, 10.0)], "hovering inside occupies the full flight window")
    check(cylinder_intersection_intervals(hov, 3, 100, 0, 10, 20, 20)
          == [(3.0, 13.0)], "hovering shifts with take-off")
    # Just outside the disk: nothing.
    check(cylinder_intersection_intervals(hov, 0, 100, 11, 10, 20, 20) == [],
          "hovering outside the disk never occupies")


# ---------------------------------------------------------------------------
# Integer start lifting / candidate window / classification
# ---------------------------------------------------------------------------


def test_integer_starts_and_boundary_touch():
    a = eastbound()
    r = evaluate([a], {"A": 0}, {"A": Report(0, 1)}, 0, 30, 4)
    # Occupancy [9, 11]: closed work window [s, s+4] intersects it for
    # s in [5, 11] (s=5 touches at 9, s=11 touches at 11). s=4 and s=12 safe.
    check(r["confirmable"] and not r["incomplete"], "fully predictable")
    check(r["forbidden_intervals"] == [[5, 11]],
          f"forbidden starts 5..11, got {r['forbidden_intervals']}")
    check(r["safe_intervals"] == [[0, 4], [12, 26]],
          f"safe starts [0,4] U [12,26], got {r['safe_intervals']}")
    check(r["earliest_safe_start"] == 0, "earliest safe start is 0")
    check(r["occupancy"][0]["intervals"]
          == [{"time_start": 9.0, "time_end": 11.0}],
          "occupancy row shows the closed interval")


def test_point_occupancy_blocks_integers():
    # Tangency flight at y=10: only instant q=10 occupied. d=4 => a start s
    # whose [s, s+4] covers 10 is unsafe: s in [6, 10]. s=5 ends at 9 (safe),
    # s=11 starts at 11 (safe). Touching endpoints s=6 ([6,10]) and s=10 are
    # both forbidden.
    a = eastbound(y=10.0)
    r = evaluate([a], {"A": 0}, {"A": Report(0, 1)}, 0, 30, 4)
    check(r["forbidden_intervals"] == [[6, 10]],
          f"point occupancy forbids starts 6..10, got {r['forbidden_intervals']}")
    check(r["safe_intervals"][0] == [0, 5]
          and r["safe_intervals"][1][0] == 11,
          "work ending one instant before / starting one after the point is safe")


def test_candidate_window_clipping_and_empty():
    a = eastbound()
    reports = {"A": Report(0, 1)}
    # t=8: candidates [8, 26]; occupancy [9,11] d=4 blocks [5,11] -> [8,11].
    r = evaluate([a], {"A": 0}, reports, 8, 30, 4)
    check(r["candidate_window"] == [8, 26]
          and r["forbidden_intervals"] == [[8, 11]]
          and r["safe_intervals"] == [[12, 26]]
          and r["earliest_safe_start"] == 12,
          f"candidate clipping wrong: {r['safe_intervals']}")
    # Occupancy row is clipped to the horizon [t, T] too.
    check(r["occupancy"][0]["intervals"]
          == [{"time_start": 9.0, "time_end": 11.0}],
          "occupancy clipped to horizon only when beyond it")

    # d = T - t + 1: no candidate second at all -> empty window, still
    # complete, earliest start empty.
    r = evaluate([a], {"A": 0}, reports, 0, 3, 4)
    check(r["confirmable"] and not r["incomplete"]
          and r["candidate_window"] == [0, -1]
          and r["safe_intervals"] == [] and r["earliest_safe_start"] is None
          and r["forbidden_intervals"] == [],
          "no candidate second -> empty safe interval, no earliest start")

    # T == t + d: exactly one candidate s = t.
    r = evaluate([a], {"A": 0}, reports, 0, 4, 4)
    check(r["candidate_window"] == [0, 0]
          and r["safe_intervals"] == [[0, 0]]
          and r["earliest_safe_start"] == 0,
          "single candidate s=t with work ending at T is evaluated")


def test_no_safe_start_at_all():
    # A hovering drone sits inside the cylinder for the whole horizon: no
    # candidate start is safe, yet the prediction is complete.
    hov = make_sortie("H", 1, 30, [
        Waypoint(100, 0, 20, 0), Waypoint(100, 0, 20, 30)])
    r = evaluate([hov], {"H": 0}, {"H": Report(0, 1)}, 0, 30, 4)
    check(r["confirmable"] and r["forbidden_intervals"] == [[0, 26]]
          and r["safe_intervals"] == [] and r["earliest_safe_start"] is None,
          "confirmable but genuinely no safe start -> empty interval/start")


def test_empty_fleet():
    r = evaluate([], {}, {}, 0, 30, 4)
    check(r["confirmable"] and r["safe_intervals"] == [[0, 26]]
          and r["earliest_safe_start"] == 0 and r["occupancy"] == [],
          "no sorties: every candidate start is safe")


def test_overdue_unconfirmable():
    a = eastbound()
    # t=12, A unreported with planned take-off 0 < t: overdue and excluded.
    r = evaluate([a], {"A": 0}, {}, 12, 30, 4)
    check(r["confirmable"] is False and r["incomplete"] is True
          and r["overdue_sorties"] == ["A"]
          and r["safe_intervals"] == [] and r["earliest_safe_start"] is None
          and r["forbidden_intervals"] == [] and r["occupancy"] == [],
          "overdue => unconfirmable and no safe starts provided")
    # Planned take-off exactly at t stays predictable.
    r2 = evaluate([a], {"A": 12}, {}, 12, 30, 4)
    check(r2["confirmable"] and r2["overdue_sorties"] == [],
          "planned take-off == t is still predictable")


def test_reported_actual_takeoff_used():
    a = eastbound()
    # Planned 0 but reported actual 5: querying at t=5, occupancy is [14,16],
    # so forbidden starts (d=4) are [10, 16] clipped to the candidate window.
    r = evaluate([a], {"A": 0}, {"A": Report(5, 1)}, 5, 30, 4)
    check(r["forbidden_intervals"] == [[10, 16]]
          and r["safe_intervals"] == [[5, 9], [17, 26]]
          and r["earliest_safe_start"] == 5
          and r["sorties"][0]["status"] == "reported"
          and r["sorties"][0]["takeoff_offset"] == 5,
          "reported actual take-off drives occupancy, not the plan")


def test_early_time_rejection():
    a = eastbound()
    try:
        evaluate([a], {"A": 0}, {"A": Report(9, 1)}, 3, 30, 4)
        check(False, "t earlier than a reported actual take-off must reject")
    except WorkWindowReject as exc:
        check([v["sortie_id"] for v in exc.violations] == ["A"]
              and exc.t == 3, "rejection names the violating sortie")
    # t equal to the reported take-off is allowed (instant still observable).
    r = evaluate([a], {"A": 0}, {"A": Report(3, 1)}, 3, 30, 4)
    check(r["confirmable"], "t == reported take-off is allowed")


# ---------------------------------------------------------------------------
# Persistence: amendments and reload give identical results
# ---------------------------------------------------------------------------


def test_revision_and_reload_consistency():
    a = eastbound()
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "t.db"))
        sid = store.create_scenario({"sorties": []}, 30, 10)
        bid = store.create_batch(sid, "b", {"sorties": [{"id": "A"}]},
                                 30, 10, {"offsets": {"A": 0}})

        def restored():
            row = store.get_batch(bid)
            return {r["sortie_id"]: Report(r["actual_offset"], r["revision"])
                    for r in row["reports"]}

        store.submit_report(bid, "A", 0, 1)
        r1 = evaluate([a], {"A": 0}, restored(), 0, 30, 4)
        r1b = evaluate([a], {"A": 0}, restored(), 0, 30, 4)
        check(r1 == r1b, "repeated queries are identical")
        check(r1["forbidden_intervals"] == [[5, 11]], "rev 1 occupancy")

        # Amendment (higher revision) shifts the track and the window moves.
        store.submit_report(bid, "A", 5, 2)
        r2 = evaluate([a], {"A": 0}, restored(), 5, 30, 4)
        check(r2["forbidden_intervals"] == [[10, 16]],
              "amended report re-queries with the new actual take-off")
        # Reload through a fresh Store and confirm byte-identical result.
        store2 = Store(os.path.join(d, "t.db"))
        row = store2.get_batch(bid)
        reports = {r["sortie_id"]: Report(r["actual_offset"], r["revision"])
                   for r in row["reports"]}
        r2b = evaluate([a], {"A": 0}, reports, 5, 30, 4)
        check(r2 == r2b, "reloaded batch gives the identical work window")
        # The batch plan/thresholds are frozen and the zone is query-scoped;
        # the stale revision 1 row must not influence the result.
        check(reports["A"].revision == 2 and reports["A"].actual_offset == 5,
              "only the highest revision participates")


# ---------------------------------------------------------------------------
# Independent exact oracle on random scenarios
# ---------------------------------------------------------------------------


def _oracle_unsafe(sorties, takeoffs, zone, s, duration):
    """True when [s, s+duration] intersects the closed cylinder occupancy.

    Independent direct implementation: per segment, intersect the global-time
    work window with (a) the horizontal disk time interval from the quadratic
    |P_h(t)|^2 <= r^2 and (b) the vertical slab z_lo <= z(t) <= z_hi, both
    closed (boundary instants included).  Stationary/constant cases included.
    """
    cx, cy, radius, z_lo, z_hi = zone
    ws, we = float(s), float(s + duration)
    for sortie, takeoff in zip(sorties, takeoffs):
        for seg in sortie.segments:
            g0 = seg.t0 + takeoff
            g1 = seg.t1 + takeoff
            lo, hi = max(ws, g0), min(we, g1)
            if lo > hi + 1e-12:
                continue
            # Vertical slab over global t: z(t) = p0z + vz*(t - g0).
            z0, vz = seg.p0[2], seg.v[2]
            if abs(vz) <= 1e-14:
                if not (z_lo - 1e-10 <= z0 <= z_hi + 1e-10):
                    continue
                vlo, vhi = lo, hi
            else:
                ql = (z_lo - z0) / vz
                qh = (z_hi - z0) / vz
                enter, leave = sorted((g0 + ql, g0 + qh))
                vlo, vhi = max(lo, enter), min(hi, leave)
                if vlo > vhi + 1e-12:
                    continue
            # Horizontal disk over global t.
            dx0 = seg.p0[0] - cx
            dy0 = seg.p0[1] - cy
            A = seg.v[0] ** 2 + seg.v[1] ** 2
            B = 2 * (dx0 * seg.v[0] + dy0 * seg.v[1])
            C = dx0 * dx0 + dy0 * dy0 - radius ** 2
            if A <= 1e-14:
                if dx0 * dx0 + dy0 * dy0 <= radius ** 2 + 1e-9:
                    return True
                continue
            disc = B * B - 4 * A * C
            if disc < -1e-9:
                continue
            disc = max(disc, 0.0)
            sq = math.sqrt(disc)
            hlo = (-B - sq) / (2 * A) + g0
            hhi = (-B + sq) / (2 * A) + g0
            if max(vlo, hlo) <= min(vhi, hhi) + 1e-12:
                return True
    return False


def test_random_oracle_crosscheck():
    rng = random.Random(20261005)
    trials = 400
    for _ in range(trials):
        n = rng.randint(1, 3)
        sorties = []
        plan = {}
        reports = {}
        t = rng.randint(0, 8)
        T = t + rng.randint(0, 40)
        d = rng.randint(1, 8)
        for i in range(n):
            sid = f"S{i}"
            wps = []
            x = rng.uniform(-40, 40)
            y = rng.uniform(-40, 40)
            z = rng.uniform(0, 40)
            nseg = rng.randint(1, 3)
            for k in range(nseg + 1):
                wps.append(Waypoint(x, y, z, 5 * k))
                x += rng.uniform(-60, 60)
                y += rng.uniform(-60, 60)
                z += rng.uniform(-15, 15)
            sorties.append(make_sortie(sid, 1, 40, wps))
            planned = rng.randint(0, 12)
            plan[sid] = planned
            # Half the time a report exists; else the sortie may be planned or
            # overdue - oracle comparison only runs on fully predictable runs.
            if rng.random() < 0.5:
                reports[sid] = Report(rng.randint(0, 12), 1)
        try:
            res = evaluate_work_window(
                sorties, plan, reports, t, T, d,
                rng.uniform(-40, 40), rng.uniform(-40, 40),
                rng.uniform(2, 30), rng.uniform(-5, 20), rng.uniform(20, 45))
        except WorkWindowReject:
            # t earlier than a randomly generated future take-off is a legal
            # rejection, not an evaluation to compare.
            continue
        if not res["confirmable"]:
            continue
        takeoffs = {row["id"]: row["takeoff_offset"] for row in res["sorties"]}
        ordered = sorties
        toffs = [takeoffs[s.id] for s in ordered]
        zone = (res["zone"]["cx"], res["zone"]["cy"], res["zone"]["radius"],
                res["zone"]["z_lo"], res["zone"]["z_hi"])
        lo, hi = res["candidate_window"]
        safe_set = set()
        for a, b in res["safe_intervals"]:
            safe_set.update(range(a, b + 1))
        blocked_set = set()
        for a, b in res["forbidden_intervals"]:
            blocked_set.update(range(a, b + 1))
        for s in range(lo, hi + 1):
            oracle_bad = _oracle_unsafe(ordered, toffs, zone, s, d)
            if oracle_bad:
                check(s in blocked_set and s not in safe_set,
                      f"oracle: start {s} unsafe but reported safe, "
                      f"safe={res['safe_intervals']}")
            else:
                check(s in safe_set and s not in blocked_set,
                      f"oracle: start {s} safe but reported blocked, "
                      f"blocked={res['forbidden_intervals']}")
        if safe_set:
            check(res["earliest_safe_start"] == min(safe_set),
                  "earliest safe start equals the minimum safe integer")
        else:
            check(res["earliest_safe_start"] is None,
                  "no safe start => earliest is null")


def main():
    test_basic_cylinder_interval()
    test_boundary_and_point_occupancy()
    test_merges_across_waypoint()
    test_hovering_sortie()
    test_integer_starts_and_boundary_touch()
    test_point_occupancy_blocks_integers()
    test_candidate_window_clipping_and_empty()
    test_no_safe_start_at_all()
    test_empty_fleet()
    test_overdue_unconfirmable()
    test_reported_actual_takeoff_used()
    test_early_time_rejection()
    test_revision_and_reload_consistency()
    test_random_oracle_crosscheck()
    print("failures:", FAILS)
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
