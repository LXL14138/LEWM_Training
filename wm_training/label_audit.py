# release: verbatim extract of wm_training/safety_classifier.py (audit_labels + LABEL_RULE); the classifier itself is not shipped
from __future__ import annotations

import numpy as np

LABEL_RULE = {"safe": 0, "unsafe": 1, "scope": "guard_tilt_only",
              "threshold_deg": 30.0, "numerical_tolerance_deg": 1e-7,
              "com_displacement_used": False, "episode_failure_latch_used": False}


def audit_labels(diagnostics, count):
    """Cross-check every frame against recorded object quaternions, not RGB.

    This checks stored geometry/labels, not a fresh simulator replay or visibility.
    World-body quaternion order is wxyz; objects 1 and 2 are the red guards.
    """
    quat = np.asarray(diagnostics["object_quat_wxyz"], dtype=np.float64)
    tilt = np.asarray(diagnostics["guard_tilt_deg"], dtype=np.float64)
    unsafe = np.asarray(diagnostics["unsafe"])
    classes = np.asarray(diagnostics["class_label"])
    if (quat.shape != (count, 3, 4) or tilt.shape != (count, 2) or
            unsafe.shape != (count,) or classes.shape != (count,)):
        raise ValueError("State/label alignment mismatch")
    if (not np.isfinite(quat).all() or not np.isfinite(tilt).all() or
            not np.isin(unsafe, [0, 1]).all() or not np.isin(classes, [0, 1, 2]).all()):
        raise ValueError("Nonfinite or invalid safety diagnostics")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError("Non-unit recorded object quaternion")
    q = quat[:, 1:] / np.linalg.norm(quat[:, 1:], axis=-1, keepdims=True)
    expected_tilt = np.rad2deg(np.arccos(np.clip(1 - 2*(q[..., 1]**2 + q[..., 2]**2), -1, 1)))
    if not np.allclose(tilt, expected_tilt, atol=1e-4, rtol=0):
        raise ValueError("Guard tilt disagrees with recorded quaternion; do not train on these labels")
    expected = (tilt > LABEL_RULE["threshold_deg"] + LABEL_RULE["numerical_tolerance_deg"]).any(1)
    if not np.array_equal(unsafe, expected) or not np.array_equal(classes == 1, expected):
        raise ValueError("Unsafe/class labels disagree with the current-state tilt rule")
    return expected.astype(np.uint8), tilt.max(1).astype(np.float32)
