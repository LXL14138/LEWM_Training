"""Synthetic numpy-only tests for the measured-strata danger evaluation.

No real data, simulator, torch, GPU or network. Runs under the Windows
numpy-only interpreter and under WSL.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wm_training.danger_strata import (
    DIRECTIONS, GRIP_EDGES, GRIP_NAMES, SINCE_EDGES, SINCE_NAMES, TILT_EDGES, TILT_NAMES,
    band_labels, body_up, connected_alarm_start, episode_features, evaluate_set, fall_direction,
    native, onset_records_to_predictions, tilt_from_up,          # release: native was reached through evaluate_danger_strata (not shipped)
)


def quat_wxyz(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = np.deg2rad(deg) / 2
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def make_episode(left_deg, right_deg=None, left_axis=(0, 1, 0), right_axis=(0, 1, 0), pinch=None,
                 contact=None, grasped=None, green=None, in_frame=None, dt=0.05):
    """Guards rotated about horizontal axes; +Y rotation leans the top toward +X (camera)."""
    left = np.asarray(left_deg, dtype=float)
    n = len(left)
    right = np.zeros(n) if right_deg is None else np.asarray(right_deg, dtype=float)
    quat = np.zeros((n, 3, 4))
    quat[:, 0, 0] = 1
    for t in range(n):
        quat[t, 1] = quat_wxyz(left_axis, left[t])
        quat[t, 2] = quat_wxyz(right_axis, right[t])
    tilt = tilt_from_up(body_up(quat[:, 1:]))
    com = np.tile(np.array([[0.425, 0, 0.05], [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]]), (n, 1, 1))
    if green is not None:
        com[:, 0] = green
    return {"time": np.arange(n) * dt, "object_quat_wxyz": quat, "object_com_xyz": com, "guard_tilt_deg": tilt,
            "block_in_frame": np.ones((n, 3), dtype=bool) if in_frame is None else np.asarray(in_frame, dtype=bool),
            "pinch_xyz": np.tile([0.375, 0, 0.16], (n, 1)) if pinch is None else np.asarray(pinch, dtype=float),
            "guard_contact_peak_last_step_n": np.zeros((n, 2)) if contact is None else np.asarray(contact, dtype=float),
            "unsafe": (tilt > 30 + 1e-7).any(1), "grasped": np.zeros(n, dtype=bool) if grasped is None else np.asarray(grasped, dtype=bool)}


def profile(n, onset, angle):
    out = np.zeros(n)
    out[onset:] = angle
    return out


def dense(features, score_fn):
    return {eid: {"frame": np.arange(f["n"]), "score": score_fn(eid, f)} for eid, f in features.items()}


class GeometryTests(unittest.TestCase):
    def test_body_up_direction_and_tilt_from_known_rotations(self):
        up = body_up(quat_wxyz((0, 1, 0), 0))
        np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
        self.assertEqual(fall_direction(up, 0), "upright")
        cases = [((0, 1, 0), 30.5, 0, "toward_camera"), ((0, 1, 0), -30.5, 0, "away_camera"),
                 ((1, 0, 0), 30.5, 0, "lateral_outward"), ((1, 0, 0), 30.5, 1, "lateral_inward"),
                 ((1, 0, 0), -30.5, 0, "lateral_inward"), ((1, 0, 0), -30.5, 1, "lateral_outward"),
                 ((1, 1, 0), 40, 0, "lateral_outward"), ((1, 1, 0), 40, 1, "lateral_inward")]  # exact tie -> lateral
        for axis, deg, guard, expected in cases:
            up = body_up(quat_wxyz(axis, deg))
            self.assertEqual(fall_direction(up, guard), expected, (axis, deg, guard))
            self.assertAlmostEqual(float(tilt_from_up(up)), abs(deg), places=9)
        with self.assertRaises(ValueError):
            body_up([2, 0, 0, 0])
        with self.assertRaises(ValueError):
            fall_direction(np.zeros((2, 3)), [0, 2])

    def test_tilt_matches_classifier_audit_formula(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(50, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        expected = np.rad2deg(np.arccos(np.clip(1 - 2 * (q[:, 1]**2 + q[:, 2]**2), -1, 1)))
        np.testing.assert_allclose(tilt_from_up(body_up(q)), expected, atol=1e-9)

    def test_band_edges_are_half_open(self):
        got = band_labels([29.999, 30, 39.999, 40, 60, 85, 179, np.nan], TILT_EDGES, TILT_NAMES, "<30")
        self.assertEqual(got.tolist(), ["<30", "30-40", "30-40", "40-60", "60-85", "85+", "85+", "n/a"])
        self.assertEqual(band_labels([0.049, 0.05, 0.1], GRIP_EDGES, GRIP_NAMES, "n/a").tolist(), ["<5cm", "5-10cm", ">=10cm"])
        self.assertEqual(band_labels([-1, 0, 2, 3, 6, 11], SINCE_EDGES, SINCE_NAMES, "before").tolist(),
                         ["before", "0-2", "0-2", "3-5", "6-10", "11+"])
        with self.assertRaises(ValueError):
            band_labels([1], (2, 1), ("a", "b"), "x")


class EpisodeFeatureTests(unittest.TestCase):
    def test_unsafe_boundary_and_recorded_consistency_checks(self):
        ep = make_episode([0, 30, 30 + 1e-6, 31, 31])
        f = episode_features(ep, 0.05)
        np.testing.assert_array_equal(f["unsafe"], [False, False, True, True, True])
        self.assertEqual(f["event"]["onset_frame"], 2)
        bad = dict(ep, unsafe=~ep["unsafe"])
        with self.assertRaises(ValueError):
            episode_features(bad, 0.05)
        bad = dict(ep, guard_tilt_deg=ep["guard_tilt_deg"] + 1e-3)
        with self.assertRaises(ValueError):
            episode_features(bad, 0.05)
        with self.assertRaises(ValueError):
            episode_features({k: v for k, v in ep.items() if k != "grasped"}, 0.05)
        with self.assertRaises(ValueError):
            episode_features(dict(ep, grasped=ep["grasped"].astype(int)), 0.05)
        with self.assertRaises(ValueError):
            episode_features(ep, 0.1)

    def test_event_strata_leaning_flat_and_brief(self):
        n = 60
        angles = np.zeros(n)
        angles[30:33] = [31, 33, 36]
        angles[33:] = 45
        f = episode_features(make_episode(angles), 0.05)
        e = f["event"]
        self.assertEqual((e["onset_frame"], e["peak_band"], e["dominant_direction"], e["onset_direction"]),
                         (30, "40-60", "toward_camera", "toward_camera"))
        self.assertTrue(e["leaning"] and f["leaning"])
        self.assertEqual(e["grip_band_at_onset"], "5-10cm")
        self.assertEqual(e["frame_band_at_onset"], "in_frame")
        self.assertEqual(f["since_band"][[29, 30, 32, 33, 36, 41]].tolist(), ["before_onset", "0-2", "0-2", "3-5", "6-10", "11+"])
        self.assertEqual(f["tilt_band"][[30, 33]].tolist(), ["30-40", "40-60"])
        np.testing.assert_array_equal(f["clean_safe"], np.arange(n) < 10)
        flat = episode_features(make_episode(profile(n, 30, 90), left_axis=(1, 0, 0)), 0.05)
        self.assertEqual((flat["event"]["peak_band"], flat["event"]["dominant_direction"], flat["leaning"]),
                         ("85+", "lateral_outward", False))
        brief = np.zeros(n)
        brief[30:33] = 45
        self.assertFalse(episode_features(make_episode(brief), 0.05)["leaning"])
        self.assertIsNone(episode_features(make_episode(np.zeros(n)), 0.05)["event"])

    def test_hard_negative_flags(self):
        n = 50
        angles = np.zeros(n)
        angles[10:15] = 25
        contact = np.zeros((n, 2))
        contact[5:7, 0] = 0.5
        contact[22, 1] = 0.5
        pinch = np.tile([0.375, 0, 0.16], (n, 1))
        pinch[20] = pinch[22] = [0.425, -0.09, 0.08]      # beside the outer Y face: |dy| = 4 cm
        pinch[21] = [0.425, -0.09, 0.20]                  # same xy but above the height limit
        pinch[23] = [0.355, -0.05, 0.08]                  # beside the -X face: |dx| = 7 cm (fingers span X)
        pinch[24] = [0.33, -0.05, 0.08]                   # 9.5 cm away: not near
        pinch[25] = [0.425, -0.12, 0.08]                  # 7 cm on Y: beyond the narrow Y extent
        grasped = np.zeros(n, dtype=bool)
        grasped[30:32] = True
        green = np.tile([0.425, 0, 0.05], (n, 1))
        green[30] = [0.425, 0, 0.12]
        green[31] = [0.425, 0, 0.05]
        f = episode_features(make_episode(angles, contact=contact, pinch=pinch, grasped=grasped, green=green), 0.05)
        self.assertEqual(np.flatnonzero(f["hn_contact_no_fall"]).tolist(), [5, 6, 22])
        self.assertEqual(np.flatnonzero(f["hn_tilt_20_30_recovered"]).tolist(), [10, 11, 12, 13, 14])
        self.assertEqual(np.flatnonzero(f["hn_near_no_contact"]).tolist(), [20, 23])
        self.assertEqual(np.flatnonzero(f["hn_held_green_near_guard"]).tolist(), [30])
        self.assertEqual(f["safe_tilt_band"][[0, 10]].tolist(), ["<5", "20-30"])
        persistent = np.zeros(n)
        persistent[10:] = 25
        self.assertFalse(episode_features(make_episode(persistent), 0.05)["hn_tilt_20_30_recovered"].any())
        # Same 20-30 excursion but a fall follows 25 frames later: still clean (>20 frames before onset).
        falls = np.zeros(n)
        falls[10:15] = 25
        falls[40:] = 90
        g = episode_features(make_episode(falls), 0.05)
        self.assertEqual(np.flatnonzero(g["hn_tilt_20_30_recovered"]).tolist(), [10, 11, 12, 13, 14])
        self.assertFalse(g["clean_safe"][20:].any())


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def features():
        n = 60
        contact = np.zeros((n, 2))
        contact[5:9, 0] = 0.5
        return {1: episode_features(make_episode(profile(n, 30, 45)), 0.05),
                2: episode_features(make_episode(profile(n, 30, 90), left_axis=(1, 0, 0)), 0.05),
                3: episode_features(make_episode(np.zeros(n), contact=contact), 0.05),
                4: episode_features(make_episode(profile(n, 30, -90)), 0.05)}

    @staticmethod
    def scores(eid, f):
        s = np.full(f["n"], 0.1)
        if eid == 1:
            s[f["unsafe"]] = 0.9
        elif eid == 3:
            s[[5, 6, 7, 20]] = 0.9
        elif eid == 4:
            s[33:] = 0.9
            s[28] = 0.9
        return s

    def test_frame_and_event_strata_macro_and_hard_negatives(self):
        features = self.features()
        summary, table = evaluate_set(features, dense(features, self.scores), 0.5, 0.05)
        by = {r["stratum"]: r for r in summary["frame_recall_by"]["direction"]}
        self.assertEqual((by["toward_camera"]["alarms"], by["toward_camera"]["unsafe_frames"]), (30, 30))
        self.assertEqual((by["lateral_outward"]["alarms"], by["lateral_outward"]["unsafe_frames"]), (0, 30))
        self.assertEqual((by["away_camera"]["alarms"], by["away_camera"]["unsafe_frames"]), (27, 30))
        self.assertIsNone(by["lateral_inward"]["recall"])
        macro = summary["frame_macro_recall"]["direction"]
        self.assertAlmostEqual(macro["macro_recall"], (1 + 0 + 0.9) / 3)
        self.assertEqual(macro["strata_empty"], ["lateral_inward"])
        ev = summary["events"]
        self.assertEqual((ev["total"], ev["scored"], ev["caught"]), (3, 3, 2))
        self.assertEqual(ev["window_coverage_mean"], 1.0)
        leads = {e["episode_id"]: e["first_alarm_minus_onset"] for e in ev["per_event"]}
        self.assertEqual(leads, {1: 0, 2: None, 4: -2})
        self.assertAlmostEqual(ev["macro_recall"]["direction"]["macro_recall"], 2 / 3)
        self.assertAlmostEqual(summary["primary_metrics"]["macro_event_recall_direction_x_severity"], 2 / 3)
        self.assertEqual(ev["macro_recall"]["leaning"]["strata_used"], ["leaning", "not_leaning"])
        hard = summary["hard_negatives"]["contact_no_fall"]
        self.assertEqual((hard["alarms"], hard["frames"]), (3, 4))
        quiet = summary["never_unsafe_episodes"]
        self.assertEqual((quiet["episodes"], quiet["episodes_with_alarm"], quiet["alarm_frames"], quiet["alarm_runs"]), (1, 1, 4, 2))
        self.assertAlmostEqual(quiet["alarm_frames_per_minute"], 4 / (60 * 0.05 / 60))
        ctx = {r["group"]: r for r in summary["safe_frames"]["by_context"]}
        self.assertEqual((ctx["1-5_frames_before_onset"]["alarms"], ctx["1-5_frames_before_onset"]["safe_frames"]), (1, 15))
        self.assertEqual(len(table["frame"]), 240)
        json.dumps(native(summary), allow_nan=False)                                   # release: was cli.native (same function)
        strict, _ = evaluate_set(features, dense(features, self.scores), 0.95, 0.05)
        self.assertEqual(strict["frames"]["unsafe_recall"], 0.0)
        self.assertEqual(strict["events"]["caught"], 0)

    def test_connected_alarm_segment_start_and_earlier_disconnected_blips(self):
        """Audit A3: the alarm segment CONNECTED to the event window, separate from the episode's first alarm."""
        features = self.features()          # events 1, 2, 4 have onset 30 (window [20, 35]); 3 is never unsafe

        def scores(eid, f):
            s = np.full(f["n"], 0.1)
            if eid == 1:
                s[[5, 6]] = 0.9            # early disconnected blip
                s[24:41] = 0.9             # segment connected to the window starts at 24
            elif eid == 2:
                s[27:] = 0.9               # single segment starting 3 frames before onset
            elif eid == 4:
                s[15:41] = 0.9             # one segment that starts before the window and runs into it
            return s
        summary, _ = evaluate_set(features, dense(features, scores), 0.5, 0.05)
        ev = summary["events"]
        per = {e["episode_id"]: e for e in ev["per_event"]}
        self.assertEqual({k: v["first_alarm_minus_onset"] for k, v in per.items()}, {1: -25, 2: -3, 4: -15})
        self.assertEqual({k: v["connected_alarm_start_minus_onset"] for k, v in per.items()}, {1: -6, 2: -3, 4: -15})
        self.assertEqual({k: v["first_alarm_in_earlier_segment"] for k, v in per.items()}, {1: True, 2: False, 4: False})
        self.assertEqual(ev["first_alarm_minus_onset_frames"], {"median": -15.0, "q25": -20.0, "q75": -9.0, "count": 3})
        self.assertEqual(ev["connected_alarm_start_minus_onset_frames"], {"median": -6.0, "q25": -10.5, "q75": -4.5, "count": 3})
        self.assertEqual(ev["first_alarm_in_earlier_disconnected_segment"], 1)
        # not caught (alarm only after the window): both connected fields are None and the counts exclude it
        late = {1: {"frame": np.arange(60), "score": np.where(np.arange(60) >= 40, 0.9, 0.1)}}
        s2, _ = evaluate_set(features, late, 0.5, 0.05)
        e = s2["events"]["per_event"][0]
        self.assertEqual((e["caught"], e["first_alarm_minus_onset"], e["connected_alarm_start_minus_onset"], e["first_alarm_in_earlier_segment"]),
                         (False, 10, None, None))
        self.assertIsNone(s2["events"]["connected_alarm_start_minus_onset_frames"])
        self.assertEqual(s2["events"]["first_alarm_in_earlier_disconnected_segment"], 0)
        # unit: a gap in a sparse table ends the walk backwards; the blip before the gap counts as an earlier segment
        frame, alarm = np.array([21, 23, 24]), np.array([True, True, True])
        self.assertEqual(connected_alarm_start(frame, alarm, np.array([False, True, True]), 30), (-7, True))
        self.assertEqual(connected_alarm_start(frame, alarm, np.array([True, True, True]), 30), (-9, False))
        self.assertEqual(connected_alarm_start(frame, np.array([True, False, True]), np.array([True, True, True]), 30), (-9, False))
        self.assertEqual(connected_alarm_start(frame, np.array([False, False, False]), np.array([True, True, True]), 30), (None, None))

    def test_sparse_tables_report_window_coverage(self):
        features = self.features()
        sparse = {1: {"frame": np.array([30]), "score": np.array([0.9])},
                  2: {"frame": np.array([30]), "score": np.array([0.1])},
                  4: {"frame": np.array([0]), "score": np.array([0.9])}}
        summary, _ = evaluate_set(features, sparse, 0.5, 0.05)
        ev = summary["events"]
        self.assertEqual((ev["total"], ev["scored"], ev["caught"], ev["unscored_episode_ids"]), (3, 2, 1, [4]))
        self.assertAlmostEqual(ev["window_coverage_mean"], (1 / 16 + 1 / 16 + 0) / 3)
        self.assertEqual(summary["frames"]["scored"], 3)
        per = {e["episode_id"]: e for e in ev["per_event"]}
        self.assertEqual((per[1]["connected_alarm_start_minus_onset"], per[1]["first_alarm_in_earlier_segment"]), (0, False))
        self.assertEqual(ev["connected_alarm_start_minus_onset_frames"]["count"], 1)

    def test_invalid_inputs_rejected(self):
        features = self.features()
        good = {1: {"frame": np.array([30, 31]), "score": np.array([0.9, 0.9])}}
        evaluate_set(features, good, 0.5, 0.05)
        for bad in ({1: {"frame": np.array([30, 30]), "score": np.array([0.9, 0.9])}},
                    {1: {"frame": np.array([60]), "score": np.array([0.9])}},
                    {1: {"frame": np.array([30]), "score": np.array([1.5])}},
                    {1: {"frame": np.array([30.0]), "score": np.array([0.9])}}, {}):
            with self.assertRaises(ValueError):
                evaluate_set(features, bad, 0.5, 0.05)
        for threshold in (np.nan, -0.1, 1.5):
            with self.assertRaises(ValueError):
                evaluate_set(features, good, threshold, 0.05)
        with self.assertRaises(ValueError):
            evaluate_set(features, good, 0.5, 0.05, (-1, 5))

    def test_onset_records_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [{"episode_id": 5, "lead": 1, "anchor_frame": 12, "scores": {"wm": 0.7, "real_future": 0.2}},
                    {"episode_id": 5, "lead": 3, "anchor_frame": 12, "scores": {"wm": 0.1, "real_future": 0.2}},
                    {"episode_id": 6, "lead": 1, "anchor_frame": 40, "scores": {"wm": 0.9, "real_future": 0.8}}]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            sets = onset_records_to_predictions(path)
            self.assertEqual(sorted(sets), ["onsets_real_future_lead1", "onsets_real_future_lead3", "onsets_wm_lead1", "onsets_wm_lead3"])
            self.assertEqual(sets["onsets_wm_lead1"][5]["frame"].tolist(), [12])
            self.assertEqual(sets["onsets_wm_lead1"][6]["score"].tolist(), [0.9])
            self.assertEqual(sorted(sets["onsets_wm_lead3"]), [5])


# release: CommandLineTests (wm_training.evaluate_danger_strata end to end, a world-model evaluation tool) is not shipped


if __name__ == "__main__":
    unittest.main()
