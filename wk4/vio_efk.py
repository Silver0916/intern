from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wk2.vo_mono import est_vo_pair as wk2_est_vo_pair


def _iter_csv_rows(csv_path: str | Path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for raw_row in csv.DictReader(f):
            yield {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in raw_row.items()}


def _pick_first(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _project_to_psd(M: np.ndarray, min_eig: float = 1e-9) -> np.ndarray:
    M_sym = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.clip(eigvals, min_eig, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _rotation_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = np.asarray(v_from, dtype=np.float64)
    b = np.asarray(v_to, dtype=np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-12 or b_norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    a = a / a_norm
    b = b / b_norm
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - a * np.dot(axis, a)
        axis = axis / np.linalg.norm(axis)
        K = _skew(axis)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    K = _skew(cross)
    return np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - dot) / (cross_norm ** 2))


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_sensor_extrinsics(sensor_yaml_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = _read_yaml(Path(sensor_yaml_path))
    t_bs = data.get("T_BS", {})
    raw = t_bs.get("data")
    if raw is None:
        raise ValueError(f"Could not find T_BS.data in {sensor_yaml_path}")
    T_bs = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    R_bs = T_bs[:3, :3]
    t_bs_vec = T_bs[:3, 3]
    return R_bs, t_bs_vec


def load_camera_calibration(sensor_yaml_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = _read_yaml(Path(sensor_yaml_path))

    intrinsics = data.get("intrinsics")
    if intrinsics is None or len(intrinsics) < 4:
        raise ValueError(f"Could not parse camera intrinsics from {sensor_yaml_path}")

    K = np.array(
        [
            [float(intrinsics[0]), 0.0, float(intrinsics[2])],
            [0.0, float(intrinsics[1]), float(intrinsics[3])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = data.get("distortion_coefficients")
    dist_coeffs = None
    if dist is not None:
        dist_coeffs = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        if dist_coeffs.size == 0:
            dist_coeffs = None

    return K, dist_coeffs


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def estimate_initial_quaternion_from_imu(
    imu_list: list[dict],
    gravity: np.ndarray,
    num_samples: int = 200,
) -> np.ndarray:
    if not imu_list:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    use_count = min(num_samples, len(imu_list))
    accel_samples = np.asarray([imu_list[idx]["accel"] for idx in range(use_count)], dtype=np.float64)
    accel_mean = np.mean(accel_samples, axis=0)
    target_specific_force = -np.asarray(gravity, dtype=np.float64)
    R_wb = _rotation_from_two_vectors(accel_mean, target_specific_force)
    return _rotmat_to_quat(R_wb)


def estimate_mean_accel(imu_list: list[dict], num_samples: int = 200) -> np.ndarray:
    if not imu_list:
        return np.array([0.0, 0.0, 9.81], dtype=np.float64)
    use_count = min(num_samples, len(imu_list))
    accel_samples = np.asarray([imu_list[idx]["accel"] for idx in range(use_count)], dtype=np.float64)
    return np.mean(accel_samples, axis=0)


def initialize_state_in_vo_frame(
    imu_list: list[dict],
    cam0_sensor_yaml: str | Path | None,
    gt_list: list[dict] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    accel_mean = estimate_mean_accel(imu_list)

    if cam0_sensor_yaml is not None and Path(cam0_sensor_yaml).exists():
        R_bc, _ = load_sensor_extrinsics(cam0_sensor_yaml)
        R_cb = R_bc.T
        q_wb = _rotmat_to_quat(R_cb)
        gravity = -R_cb @ accel_mean
    else:
        gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        q_wb = estimate_initial_quaternion_from_imu(imu_list, gravity)

    state = {
        "p": np.zeros(3, dtype=np.float64),
        "v": np.zeros(3, dtype=np.float64),
        "q": q_wb,
        "ba": np.zeros(3, dtype=np.float64),
        "bg": np.zeros(3, dtype=np.float64),
    }

    if gt_list:
        first_gt = gt_list[0]
        if first_gt.get("ba") is not None:
            state["ba"] = first_gt["ba"].copy()
        if first_gt.get("bg") is not None:
            state["bg"] = first_gt["bg"].copy()

    return state, gravity


def initialize_state(
    imu_list: list[dict],
    gravity: np.ndarray,
    gt_list: list[dict] | None = None,
) -> dict[str, np.ndarray]:
    state = {
        "p": np.zeros(3, dtype=np.float64),
        "v": np.zeros(3, dtype=np.float64),
        "q": estimate_initial_quaternion_from_imu(imu_list, gravity),
        "ba": np.zeros(3, dtype=np.float64),
        "bg": np.zeros(3, dtype=np.float64),
    }

    if gt_list:
        first_gt = gt_list[0]
        if first_gt.get("ba") is not None:
            state["ba"] = first_gt["ba"].copy()
        if first_gt.get("bg") is not None:
            state["bg"] = first_gt["bg"].copy()

    return state


def load_imu_csv(imu_csv_path):
    """
    Read EuRoC IMU CSV.

    Input:
        imu_csv_path: str or Path

    Output:
        imu_list: list of dict
            each item contains:
                {
                    "timestamp_ns": int,
                    "gyro": np.ndarray,   # shape (3,)
                    "accel": np.ndarray,  # shape (3,)
                }
    """
    imu_list = []
    for row in _iter_csv_rows(imu_csv_path):
        ts_val = _pick_first(row, ["#timestamp [ns]", "#timestamp", "timestamp_ns", "timestamp"])
        gx_val = _pick_first(row, ["w_RS_S_x [rad s^-1]", "gyro_x", "gx"])
        gy_val = _pick_first(row, ["w_RS_S_y [rad s^-1]", "gyro_y", "gy"])
        gz_val = _pick_first(row, ["w_RS_S_z [rad s^-1]", "gyro_z", "gz"])
        ax_val = _pick_first(row, ["a_RS_S_x [m s^-2]", "accel_x", "ax"])
        ay_val = _pick_first(row, ["a_RS_S_y [m s^-2]", "accel_y", "ay"])
        az_val = _pick_first(row, ["a_RS_S_z [m s^-2]", "accel_z", "az"])
        if None in (ts_val, gx_val, gy_val, gz_val, ax_val, ay_val, az_val):
            continue

        imu_list.append(
            {
                "timestamp_ns": int(ts_val),
                "gyro": np.array([float(gx_val), float(gy_val), float(gz_val)], dtype=np.float64),
                "accel": np.array([float(ax_val), float(ay_val), float(az_val)], dtype=np.float64),
            }
        )

    imu_list.sort(key=lambda item: item["timestamp_ns"])
    if not imu_list:
        raise ValueError(f"No valid IMU rows found in {imu_csv_path}")
    return imu_list


def load_frame_index_csv(frame_index_csv_path):
    """
    Read frame index CSV generated by your VO pipeline.

    Output:
        frame_list: list of dict
            {
                "frame": int,
                "timestamp_ns": int
            }
    """
    frame_list = []
    for row in _iter_csv_rows(frame_index_csv_path):
        frame_val = _pick_first(row, ["frame"])
        ts_val = _pick_first(row, ["timestamp_ns", "#timestamp [ns]", "#timestamp", "timestamp"])
        if frame_val is None or ts_val is None:
            continue
        frame_list.append(
            {
                "frame": int(frame_val),
                "timestamp_ns": int(ts_val),
                "filename": row.get("filename", ""),
            }
        )

    frame_list.sort(key=lambda item: item["frame"])
    if not frame_list:
        raise ValueError(f"No valid frame rows found in {frame_index_csv_path}")
    return frame_list


def load_vo_estimated_poses(vo_csv_path):
    """
    Read VO estimated positions.

    Output:
        vo_dict: dict[int, np.ndarray]
            key: frame index
            value: position np.ndarray with shape (3,)
    """
    vo_dict = {}
    for row in _iter_csv_rows(vo_csv_path):
        frame_val = _pick_first(row, ["frame"])
        tx_val = _pick_first(row, ["tx", "x", "p_RS_R_x [m]", "p_RS_R_x"])
        ty_val = _pick_first(row, ["ty", "y", "p_RS_R_y [m]", "p_RS_R_y"])
        tz_val = _pick_first(row, ["tz", "z", "p_RS_R_z [m]", "p_RS_R_z"])
        status = row.get("status", "").strip().lower()
        if frame_val is None or tx_val is None or ty_val is None or tz_val is None:
            continue
        if status and status not in {"success", "init"}:
            continue

        vo_dict[int(frame_val)] = np.array([float(tx_val), float(ty_val), float(tz_val)], dtype=np.float64)

    if not vo_dict:
        raise ValueError(f"No valid VO pose rows found in {vo_csv_path}")
    return vo_dict


def load_gt_csv(gt_csv_path):
    """
    Optional GT loader for initialization or evaluation.

    Output:
        gt_list: list of dict
            {
                "timestamp_ns": int | None,
                "p": np.ndarray | None,   # shape (3,)
                "v": np.ndarray | None,   # shape (3,)
                "q": np.ndarray | None,   # shape (4,) as [qw, qx, qy, qz]
                "ba": np.ndarray | None,  # shape (3,)
                "bg": np.ndarray | None,  # shape (3,)
            }
    """
    gt_list = []
    for row in _iter_csv_rows(gt_csv_path):
        ts_val = _pick_first(row, ["#timestamp [ns]", "#timestamp", "timestamp_ns", "timestamp"])
        px_val = _pick_first(row, ["p_RS_R_x [m]", "p_RS_R_x", "tx", "x"])
        py_val = _pick_first(row, ["p_RS_R_y [m]", "p_RS_R_y", "ty", "y"])
        pz_val = _pick_first(row, ["p_RS_R_z [m]", "p_RS_R_z", "tz", "z"])
        qw_val = _pick_first(row, ["q_RS_w []", "qw"])
        qx_val = _pick_first(row, ["q_RS_x []", "qx"])
        qy_val = _pick_first(row, ["q_RS_y []", "qy"])
        qz_val = _pick_first(row, ["q_RS_z []", "qz"])
        vx_val = _pick_first(row, ["v_RS_R_x [m s^-1]", "v_RS_R_x", "vx"])
        vy_val = _pick_first(row, ["v_RS_R_y [m s^-1]", "v_RS_R_y", "vy"])
        vz_val = _pick_first(row, ["v_RS_R_z [m s^-1]", "v_RS_R_z", "vz"])
        bgx_val = _pick_first(row, ["b_w_RS_S_x [rad s^-1]", "bgx"])
        bgy_val = _pick_first(row, ["b_w_RS_S_y [rad s^-1]", "bgy"])
        bgz_val = _pick_first(row, ["b_w_RS_S_z [rad s^-1]", "bgz"])
        bax_val = _pick_first(row, ["b_a_RS_S_x [m s^-2]", "bax"])
        bay_val = _pick_first(row, ["b_a_RS_S_y [m s^-2]", "bay"])
        baz_val = _pick_first(row, ["b_a_RS_S_z [m s^-2]", "baz"])

        gt_list.append(
            {
                "timestamp_ns": int(ts_val) if ts_val is not None else None,
                "p": (
                    np.array([float(px_val), float(py_val), float(pz_val)], dtype=np.float64)
                    if None not in (px_val, py_val, pz_val)
                    else None
                ),
                "v": (
                    np.array([float(vx_val), float(vy_val), float(vz_val)], dtype=np.float64)
                    if None not in (vx_val, vy_val, vz_val)
                    else None
                ),
                "q": (
                    quat_normalize(np.array([float(qw_val), float(qx_val), float(qy_val), float(qz_val)], dtype=np.float64))
                    if None not in (qw_val, qx_val, qy_val, qz_val)
                    else None
                ),
                "bg": (
                    np.array([float(bgx_val), float(bgy_val), float(bgz_val)], dtype=np.float64)
                    if None not in (bgx_val, bgy_val, bgz_val)
                    else None
                ),
                "ba": (
                    np.array([float(bax_val), float(bay_val), float(baz_val)], dtype=np.float64)
                    if None not in (bax_val, bay_val, baz_val)
                    else None
                ),
            }
        )

    gt_list.sort(
        key=lambda item: item["timestamp_ns"] if item["timestamp_ns"] is not None else -1,
    )
    return gt_list


def quat_normalize(q):
    """
    Normalize quaternion.

    Input:
        q: np.ndarray, shape (4,)
    Output:
        normalized quaternion, shape (4,)
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply(q1, q2):
    """
    Quaternion multiplication.

    Input:
        q1, q2: np.ndarray, shape (4,)
    Output:
        q: np.ndarray, shape (4,)
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)

    q = np.empty(4, dtype=np.float64)
    q[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return q


def small_angle_quat(delta_theta):
    """
    Convert small angle vector to quaternion.

    Input:
        delta_theta: np.ndarray, shape (3,)
    Output:
        dq: np.ndarray, shape (4,)
    """
    delta_theta = np.asarray(delta_theta, dtype=np.float64)
    theta = np.linalg.norm(delta_theta)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if theta < 1e-6:
        dq = np.array([1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]], dtype=np.float64)
        return quat_normalize(dq)

    axis = delta_theta / theta
    half_theta = 0.5 * theta
    sin_half = np.sin(half_theta)
    dq = np.array(
        [
            np.cos(half_theta),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        ],
        dtype=np.float64,
    )
    return quat_normalize(dq)


def quat_to_rotmat(q):
    """
    Convert quaternion to rotation matrix.

    Input:
        q: np.ndarray, shape (4,)
    Output:
        R: np.ndarray, shape (3, 3)
    """
    qw, qx, qy, qz = quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (qy ** 2 + qz ** 2), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx ** 2 + qz ** 2), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx ** 2 + qy ** 2)],
        ],
        dtype=np.float64,
    )


def quat_conjugate(q):
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def rotmat_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-9:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ],
            dtype=np.float64,
        )

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sin_theta)
    return axis * theta


def propagate_nominal_state(state, accel_m, gyro_m, dt, gravity):
    """
    IMU propagation for nominal state.

    Input:
        state: dict
            {
                "p": np.ndarray,   # (3,)
                "v": np.ndarray,   # (3,)
                "q": np.ndarray,   # (4,)
                "ba": np.ndarray,  # (3,)
                "bg": np.ndarray,  # (3,)
            }
        accel_m: np.ndarray, shape (3,)
        gyro_m: np.ndarray, shape (3,)
        dt: float
        gravity: np.ndarray, shape (3,)

    Output:
        state_pred: dict with same structure
    """
    if dt <= 0.0:
        return {key: value.copy() for key, value in state.items()}

    accel_unbiased = np.asarray(accel_m, dtype=np.float64) - state["ba"]
    gyro_unbiased = np.asarray(gyro_m, dtype=np.float64) - state["bg"]
    R_wb = quat_to_rotmat(state["q"])
    accel_world = R_wb @ accel_unbiased + np.asarray(gravity, dtype=np.float64)

    state_pred = {
        "p": state["p"] + state["v"] * dt + 0.5 * accel_world * (dt ** 2),
        "v": state["v"] + accel_world * dt,
        "q": quat_normalize(quat_multiply(state["q"], small_angle_quat(gyro_unbiased * dt))),
        "ba": state["ba"].copy(),
        "bg": state["bg"].copy(),
    }
    return state_pred


def build_F_and_Q(state, accel_m, gyro_m, dt, sigma_a, sigma_g, sigma_ba, sigma_bg):
    """
    Build EKF error-state transition matrix F and process noise Q.

    Output:
        F: np.ndarray, shape (15, 15)
        Q: np.ndarray, shape (15, 15)
    """
    F = np.eye(15, dtype=np.float64)
    Q = np.zeros((15, 15), dtype=np.float64)
    if dt <= 0.0:
        return F, Q

    I3 = np.eye(3, dtype=np.float64)

    F[0:3, 3:6] = I3 * dt

    q_pos = 0.25 * (sigma_a ** 2) * (dt ** 4)
    q_pos_vel = 0.5 * (sigma_a ** 2) * (dt ** 3)
    q_vel = (sigma_a ** 2) * (dt ** 2)
    q_theta = (sigma_g ** 2) * (dt ** 2)
    q_ba = (sigma_ba ** 2) * dt
    q_bg = (sigma_bg ** 2) * dt

    Q[0:3, 0:3] = I3 * q_pos
    Q[0:3, 3:6] = I3 * q_pos_vel
    Q[3:6, 0:3] = I3 * q_pos_vel
    Q[3:6, 3:6] = I3 * q_vel
    Q[6:9, 6:9] = I3 * q_theta
    Q[9:12, 9:12] = I3 * q_ba
    Q[12:15, 12:15] = I3 * q_bg
    return F, Q


def propagate_covariance(P, F, Q):
    """
    EKF covariance propagation.

    Input:
        P: np.ndarray, shape (15, 15)
    Output:
        P_pred: np.ndarray, shape (15, 15)
    """
    P_pred = F @ P @ F.T + Q
    return 0.5 * (P_pred + P_pred.T)


def ekf_update_with_vo_position(state, P, z_vo, R_meas):
    """
    EKF update using VO position only.

    Input:
        state: dict
        P: np.ndarray, shape (15, 15)
        z_vo: np.ndarray, shape (3,)
        R_meas: np.ndarray, shape (3, 3)

    Output:
        state_upd: dict
        P_upd: np.ndarray, shape (15, 15)
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[:, 0:3] = np.eye(3, dtype=np.float64)

    P = _project_to_psd(P)
    residual = np.asarray(z_vo, dtype=np.float64) - state["p"]
    S = H @ P @ H.T + np.asarray(R_meas, dtype=np.float64)
    S = 0.5 * (S + S.T)
    min_s_eig = float(np.min(np.linalg.eigvalsh(S)))
    if min_s_eig < 1e-9:
        S = S + np.eye(3, dtype=np.float64) * (1e-9 - min_s_eig)
    K = P @ H.T @ np.linalg.pinv(S)
    delta_x = K @ residual

    state_upd = {
        "p": state["p"] + delta_x[0:3],
        "v": state["v"] + delta_x[3:6],
        "q": quat_normalize(quat_multiply(state["q"], small_angle_quat(delta_x[6:9]))),
        "ba": state["ba"] + delta_x[9:12],
        "bg": state["bg"] + delta_x[12:15],
    }

    I = np.eye(15, dtype=np.float64)
    I_KH = I - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
    P_upd = _project_to_psd(P_upd)
    return state_upd, P_upd


def ekf_update_with_vo_attitude(state, P, q_vo, R_meas_theta):
    """
    EKF update using VO attitude observation.

    The attitude error is represented with a right-multiplicative small angle:
    R_obs ~= R_pred * Exp(delta_theta^)
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[:, 6:9] = np.eye(3, dtype=np.float64)

    P = _project_to_psd(P)
    R_pred = quat_to_rotmat(state["q"])
    R_obs = quat_to_rotmat(q_vo)
    residual = rotmat_log(R_pred.T @ R_obs)

    S = H @ P @ H.T + np.asarray(R_meas_theta, dtype=np.float64)
    S = 0.5 * (S + S.T)
    min_s_eig = float(np.min(np.linalg.eigvalsh(S)))
    if min_s_eig < 1e-9:
        S = S + np.eye(3, dtype=np.float64) * (1e-9 - min_s_eig)

    K = P @ H.T @ np.linalg.pinv(S)
    delta_x = K @ residual

    state_upd = {
        "p": state["p"] + delta_x[0:3],
        "v": state["v"] + delta_x[3:6],
        "q": quat_normalize(quat_multiply(state["q"], small_angle_quat(delta_x[6:9]))),
        "ba": state["ba"] + delta_x[9:12],
        "bg": state["bg"] + delta_x[12:15],
    }

    I = np.eye(15, dtype=np.float64)
    I_KH = I - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R_meas_theta @ K.T
    P_upd = _project_to_psd(P_upd)
    return state_upd, P_upd


def estimate_vo_body_rotation_for_pair(
    prev_frame: dict,
    curr_frame: dict,
    frames_dir: Path,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None,
    R_bc: np.ndarray,
) -> np.ndarray | None:
    prev_path = frames_dir / prev_frame["filename"]
    curr_path = frames_dir / curr_frame["filename"]
    if not prev_path.exists() or not curr_path.exists():
        return None

    dt = (curr_frame["timestamp_ns"] - prev_frame["timestamp_ns"]) * 1e-9
    pair_result = wk2_est_vo_pair(
        gt_poses={},
        gt_velocities={},
        intrinsics=intrinsics,
        dist_coeffs=dist_coeffs,
        curr_image_dir=prev_path,
        next_image_dir=curr_path,
        curr_frame=int(prev_frame["frame"]),
        next_frame=int(curr_frame["frame"]),
        fallback_scale=1.0,
        delta_t=dt if dt > 0.0 else 1.0,
    )
    if pair_result.get("status") != "success":
        return None

    R_c2_c1 = np.asarray(pair_result["R"], dtype=np.float64).reshape(3, 3)
    R_cb = R_bc.T
    return R_bc @ R_c2_c1 @ R_cb


def find_imu_segment(imu_list, t_start_ns, t_end_ns):
    """
    Collect all IMU samples between two timestamps.

    Output:
        imu_segment: list of dict
    """
    if not imu_list:
        return []

    cache_key = "_cached_timestamps"
    if getattr(find_imu_segment, "_cached_id", None) != id(imu_list):
        setattr(
            find_imu_segment,
            cache_key,
            np.asarray([item["timestamp_ns"] for item in imu_list], dtype=np.int64),
        )
        setattr(find_imu_segment, "_cached_id", id(imu_list))

    timestamps = getattr(find_imu_segment, cache_key)
    left = int(np.searchsorted(timestamps, int(t_start_ns), side="left"))
    right = int(np.searchsorted(timestamps, int(t_end_ns), side="right"))
    return imu_list[left:right]


def save_vio_results(output_csv_path, results):
    """
    Save VIO results.

    Input:
        results: list of dict
            {
                "frame": int,
                "timestamp_ns": int,
                "p": np.ndarray,  # (3,)
                "v": np.ndarray,  # (3,)
                "q": np.ndarray,  # (4,)
            }
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["frame", "timestamp_ns", "tx", "ty", "tz", "vx", "vy", "vz", "qw", "qx", "qy", "qz"]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            p = np.asarray(item["p"], dtype=np.float64)
            v = np.asarray(item["v"], dtype=np.float64)
            q = quat_normalize(item["q"])
            writer.writerow(
                {
                    "frame": int(item["frame"]),
                    "timestamp_ns": int(item["timestamp_ns"]),
                    "tx": f"{p[0]:.8f}",
                    "ty": f"{p[1]:.8f}",
                    "tz": f"{p[2]:.8f}",
                    "vx": f"{v[0]:.8f}",
                    "vy": f"{v[1]:.8f}",
                    "vz": f"{v[2]:.8f}",
                    "qw": f"{q[0]:.8f}",
                    "qx": f"{q[1]:.8f}",
                    "qy": f"{q[2]:.8f}",
                    "qz": f"{q[3]:.8f}",
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu_csv", type=str, required=True)
    parser.add_argument("--frame_index_csv", type=str, required=True)
    parser.add_argument("--vo_est_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--gt_csv", type=str, default=None)
    parser.add_argument("--cam0_sensor_yaml", type=str, default=None)
    args = parser.parse_args()

    imu_list = load_imu_csv(args.imu_csv)
    frame_list = load_frame_index_csv(args.frame_index_csv)
    vo_dict = load_vo_estimated_poses(args.vo_est_csv)
    gt_list = load_gt_csv(args.gt_csv) if args.gt_csv is not None else None

    cam0_sensor_yaml = args.cam0_sensor_yaml
    if cam0_sensor_yaml is None:
        candidate = Path(args.frame_index_csv).resolve().parent / "sensor.yaml"
        cam0_sensor_yaml = str(candidate) if candidate.exists() else None

    frames_dir = Path(args.frame_index_csv).resolve().parent / "data"
    intrinsics = None
    dist_coeffs = None
    R_bc = None
    if cam0_sensor_yaml is not None and Path(cam0_sensor_yaml).exists():
        intrinsics, dist_coeffs = load_camera_calibration(cam0_sensor_yaml)
        R_bc, _ = load_sensor_extrinsics(cam0_sensor_yaml)

    state, gravity = initialize_state_in_vo_frame(imu_list, cam0_sensor_yaml, gt_list=gt_list)
    P = np.eye(15, dtype=np.float64) * 1e-2

    sigma_a = 0.05
    sigma_g = 0.01
    sigma_ba = 0.001
    sigma_bg = 0.001
    sigma_vo_pos = 0.10
    sigma_vo_att = 0.08
    R_meas = np.eye(3, dtype=np.float64) * (sigma_vo_pos ** 2)
    R_meas_theta = np.eye(3, dtype=np.float64) * (sigma_vo_att ** 2)

    results = []

    for i in range(len(frame_list)):
        frame_id = frame_list[i]["frame"]
        frame_ts = frame_list[i]["timestamp_ns"]

        if i == 0:
            results.append(
                {
                    "frame": frame_id,
                    "timestamp_ns": frame_ts,
                    "p": state["p"].copy(),
                    "v": state["v"].copy(),
                    "q": state["q"].copy(),
                }
            )
            continue

        prev_state = {key: value.copy() for key, value in state.items()}
        prev_ts = frame_list[i - 1]["timestamp_ns"]
        imu_segment = find_imu_segment(imu_list, prev_ts, frame_ts)

        for k in range(1, len(imu_segment)):
            t0 = imu_segment[k - 1]["timestamp_ns"]
            t1 = imu_segment[k]["timestamp_ns"]
            dt = (t1 - t0) * 1e-9
            if dt <= 0.0:
                continue

            accel_m = imu_segment[k - 1]["accel"]
            gyro_m = imu_segment[k - 1]["gyro"]

            F, Q = build_F_and_Q(state, accel_m, gyro_m, dt, sigma_a, sigma_g, sigma_ba, sigma_bg)
            state = propagate_nominal_state(state, accel_m, gyro_m, dt, gravity)
            P = propagate_covariance(P, F, Q)

        if intrinsics is not None and R_bc is not None and frames_dir.exists():
            R_b2_b1 = estimate_vo_body_rotation_for_pair(
                frame_list[i - 1],
                frame_list[i],
                frames_dir,
                intrinsics,
                dist_coeffs,
                R_bc,
            )
            if R_b2_b1 is not None:
                R_b1_b2 = R_b2_b1.T
                q_rel = _rotmat_to_quat(R_b1_b2)
                q_vo_att = quat_normalize(quat_multiply(prev_state["q"], q_rel))
                state, P = ekf_update_with_vo_attitude(state, P, q_vo_att, R_meas_theta)

        if frame_id in vo_dict:
            z_vo = vo_dict[frame_id]
            state, P = ekf_update_with_vo_position(state, P, z_vo, R_meas)

        results.append(
            {
                "frame": frame_id,
                "timestamp_ns": frame_ts,
                "p": state["p"].copy(),
                "v": state["v"].copy(),
                "q": state["q"].copy(),
            }
        )

    save_vio_results(args.out_csv, results)


if __name__ == "__main__":
    main()
