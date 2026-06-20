import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import math
import time
from datetime import datetime
from ultralytics import YOLO

# --- 기존 상수 및 설정값 ---
CONF_TH = 0.35
KP_CONF_TH = 0.25
SMOOTHING_ALPHA = 0.45
WARMUP_SECONDS = 5.0
INFER_EVERY_N_FRAMES = 2
RIGHT_PANEL_W = 250
SHOW_RIGHT_PANEL = True
MAX_DISPLAY_WORKERS = 4
MAX_SIMPLE_WORKERS = 4
TRACK_MATCH_DISTANCE = 160

# 색상 및 골격 정의
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
DARK = (25, 25, 25)
DARK2 = (15, 15, 15)
LINE = (90, 90, 90)

NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 0, 5, 6, 7, 8
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

# --- REBA/OWAS 계산 함수들 ---
def fmt_short(sec):
    sec = int(sec)
    return f"{sec//60:02d}:{sec%60:02d}"

def fmt_angle(a):
    return "0" if a is None else f"{a:.0f}"

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def center_from_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def angle_3pt(a, b, c):
    if a is None or b is None or c is None: return None
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by); v2 = (cx - bx, cy - by)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 == 0 or n2 == 0: return None
    cosv = max(-1, min(1, dot / (n1 * n2)))
    return abs(math.degrees(math.acos(cosv)))

def mean_point(points):
    valid = [p for p in points if p is not None]
    if not valid: return None
    return (sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid))

def smooth(old, new, alpha=SMOOTHING_ALPHA):
    if new is None: return old
    if old is None: return new
    return alpha * new + (1 - alpha) * old

def get_kp(kps, idx):
    if kps is None or idx >= len(kps): return None
    x, y, conf = kps[idx]
    if conf < KP_CONF_TH: return None
    return (float(x), float(y))

def calculate_angles(kps):
    nose = get_kp(kps, NOSE)
    l_sh = get_kp(kps, L_SHOULDER); r_sh = get_kp(kps, R_SHOULDER)
    l_el = get_kp(kps, L_ELBOW); r_el = get_kp(kps, R_ELBOW)
    l_hip = get_kp(kps, L_HIP); r_hip = get_kp(kps, R_HIP)
    l_knee = get_kp(kps, L_KNEE); r_knee = get_kp(kps, R_KNEE)
    l_ankle = get_kp(kps, L_ANKLE); r_ankle = get_kp(kps, R_ANKLE)

    shoulder_mid = mean_point([l_sh, r_sh])
    hip_mid = mean_point([l_hip, r_hip])
    knee_mid = mean_point([l_knee, r_knee])

    waist = None
    raw_waist = angle_3pt(shoulder_mid, hip_mid, knee_mid)
    if raw_waist is not None: waist = max(0, 180 - raw_waist)

    neck = None
    raw_neck = angle_3pt(nose, shoulder_mid, hip_mid)
    if raw_neck is not None: neck = max(0, 180 - raw_neck)

    knee_values = []
    lk = angle_3pt(l_hip, l_knee, l_ankle)
    rk = angle_3pt(r_hip, r_knee, r_ankle)
    if lk is not None: knee_values.append(max(0, 180 - lk))
    if rk is not None: knee_values.append(max(0, 180 - rk))
    knee = None if not knee_values else sum(knee_values) / len(knee_values)

    shoulder_values = []
    ls = angle_3pt(l_hip, l_sh, l_el)
    rs = angle_3pt(r_hip, r_sh, r_el)
    if ls is not None: shoulder_values.append(ls)
    if rs is not None: shoulder_values.append(rs)
    shoulder = None if not shoulder_values else max(shoulder_values)

    return waist, neck, knee, shoulder

def classify_part(part, angle):
    if angle is None: return "HOLD", GRAY
    if part == "waist":
        if angle >= 60: return "RISK", RED
