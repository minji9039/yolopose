import os
import sys
import cv2
import time
import math
import numpy as np
import streamlit as st
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ==========================================
# OpenPose 환경 설정 (로컬 PC 경로에 맞게 수정 필요)
# ==========================================
try:
    sys.path.append(os.path.abspath("bin/python/openpose/Release"))
    os.environ["PATH"] += ";" + os.path.abspath("bin")
    import pyopenpose as op
except ImportError as e:
    st.error(f"OpenPose 모듈을 불러올 수 없습니다. 경로를 확인해주세요: {e}")

# --- 상수 설정 ---
MAX_WORKERS = 3
WARMUP_SECONDS = 8.0
DRAW_CONF_TH = 0.15
ANGLE_CONF_TH = 0.25
TRACK_MATCH_DISTANCE = 250
SMOOTHING_ALPHA = 0.35

DASHBOARD_W, DASHBOARD_H = 980, 580
VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H = 10, 20, 710, 480
RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H = 728, 20, 242, 480
BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H = 10, 505, 960, 65

BG = (15, 20, 25)
PANEL = (32, 38, 44)
WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
GREEN = (0, 220, 60)
YELLOW = (0, 230, 230)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

NOSE, NECK = 0, 1
R_SHOULDER, R_ELBOW, L_SHOULDER, L_ELBOW = 2, 3, 5, 6
MID_HIP, R_HIP, R_KNEE, R_ANKLE = 8, 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14

# 클라우드/리눅스 환경 테스트를 대비한 폰트 팩백 처리
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" # 리눅스 나눔고딕 경로

# --- UI 그리기 함수들 ---
def draw_korean_text(img, text, pos, font_size=16, color=WHITE):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    rgb = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_center_text(img, text, box_x, box_y, box_w, y_offset, font_size, color):
    temp = Image.new("RGB", (box_w, 40))
    draw = ImageDraw.Draw(temp)
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = box_x + (box_w - text_w) // 2
    y = box_y + y_offset
    return draw_korean_text(img, text, (x, y), font_size, color)

def draw_panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 65, 75), 1)
    if title: img = draw_korean_text(img, title, (x + 15, y + 12), 18, WHITE)
    return img

def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"

def get_most_risky_joint(w, n, k, s):
    values = {"허리": w, "목": n, "무릎": k, "어깨": s}
    joint = max(values, key=values.get)
    if values[joint] <= 0: return "없음", 0
    return joint, values[joint]

# --- 계산 및 분류 로직 ---
def calculate_angle(a, b, c):
    if a is None or b is None or c is None: return None
    ax, ay = a; bx, by = b; cx, cy = c
    angle1 = math.atan2(ay - by, ax - bx)
    angle2 = math.atan2(cy - by, cx - bx)
    angle = abs(math.degrees(angle2 - angle1))
    if angle > 180: angle = 360 - angle
    return angle

def get_point(person, idx, conf_th=ANGLE_CONF_TH):
    x, y, c = person[idx]
    if c < conf_th: return None
    return (float(x), float(y))

def get_draw_point(person, idx): return get_point(person, idx, DRAW_CONF_TH)
def distance(p1, p2): return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mean_point(points):
    valid = [p for p in points if p is not None]
    if not valid: return None
    return (sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid))

def smooth_value(old, new, alpha=SMOOTHING_ALPHA):
    if new is None: return old
    if old is None: return new
    return alpha * new + (1 - alpha) * old

def classify_waist(angle, valid=True):
    if not valid or angle is None: return "보류", GRAY
    if angle >= 60: return "위험", RED
    elif angle >= 20: return "주의", YELLOW
    return "안전", GREEN

def classify_neck(angle, valid=True):
    if not valid or angle is None: return "보류", GRAY
    if angle >= 45: return "위험", RED
    elif angle >= 20: return "주의", YELLOW
    return "안전", GREEN

def classify_shoulder(angle, valid=True):
    if not valid or angle is None: return "보류", GRAY
    if angle >= 90: return "위험", RED
    elif angle >= 45: return "주의", YELLOW
    return "안전", GREEN

def classify_knee(angle, valid=True):
    if not valid or angle is None: return "보류", GRAY
    if angle >= 60: return "위험", RED
    elif angle >= 30: return "주의", YELLOW
    return "안전", GREEN

def get_overall_risk(risks):
    valid = [r for r in risks if r in ["안전", "주의", "위험", "관찰"]]
    if not valid: return "보류", GRAY
    if valid.count("위험") >= 1: return "위험", RED
    if valid.count("주의") >= 2: return "주의", YELLOW
    if valid.count("주의") == 1: return "관찰", ORANGE
    return "안전", GREEN

def reba_part_score(part, angle, valid=True):
    if not valid or angle is None: return 0
    if part == "waist": return 1 if angle < 20 else (2 if angle < 60 else 3)
    if part == "neck": return 1 if angle < 20 else 2
    if part == "knee": return 1 if angle < 30 else (2 if angle < 60 else 3)
    if part == "shoulder": return 1 if angle < 45 else (2 if angle < 90 else 3)
    return 0

def calculate_simple_reba(waist, neck, knee, shoulder, waist_valid, neck_valid, knee_valid, shoulder_valid):
    w_sc = reba_part_score("waist", waist, waist_valid)
    n_sc = reba_part_score("neck", neck, neck_valid)
    k_sc = reba_part_score("knee", knee, knee_valid)
    s_sc = reba_part_score("shoulder", shoulder, shoulder_valid)
    total = w_sc + n_sc + k_sc + s_sc
    
    if total >= 9: level, color = "높음", RED
    elif total >= 6: level, color = "중간", YELLOW
    elif total >= 1: level, color = "낮음", GREEN
    else: level, color = "보류", GRAY
    return total, level, color, {"허리": w_sc, "목": n_sc, "무릎": k_sc, "어깨": s_sc}

def classify_owas_type(waist, neck, knee, shoulder, waist_valid, neck_valid, knee_valid, shoulder_valid):
    if shoulder_valid and neck_valid and shoulder >= 90 and neck >= 20: return "위보기/상부 작업"
    if waist_valid and knee_valid and waist >= 20 and knee >= 30: return "쪼그림/바닥 작업"
    if waist_valid and shoulder_valid and waist >= 20 and shoulder >= 45: return "운반/취급 작업 가능성"
    if waist_valid and knee_valid and waist >= 20 and knee < 30: return "허리 굽힘 작업"
    return "일반 작업"

def get_risk_parts(tr):
    parts = []
    if tr.get("waist_display") in ["주의", "위험"]: parts.append("허리")
    if tr.get("neck_display") in ["주의", "위험"]: parts.append("목")
    if tr.get("knee_display") in ["주의", "위험"]: parts.append("무릎")
    if tr.get("shoulder_display") in ["주의", "위험"]: parts.append("어깨")
    return ", ".join(parts) if parts else "없음"

def calculate_openpose_angles(person):
    nose = get_point(person, NOSE); neck = get_point(person, NECK); mid_hip = get_point(person, MID_HIP)
    l_sh = get_point(person, L_SHOULDER); r_sh = get_point(person, R_SHOULDER)
    l_hip = get_point(person, L_HIP); r_hip = get_point(person, R_HIP)
    l_knee = get_point(person, L_KNEE); r_knee = get_point(person, R_KNEE)
    l_ankle = get_point(person, L_ANKLE); r_ankle = get_point(person, R_ANKLE)
    l_elbow = get_point(person, L_ELBOW); r_elbow = get_point(person, R_ELBOW)

    if l_hip is None: l_hip = mid_hip
    if r_hip is None: r_hip = mid_hip

    shoulder_mid = mean_point([l_sh, r_sh, neck])
    hip_mid = mean_point([l_hip, r_hip, mid_hip])
    knee_mid = mean_point([l_knee, r_knee])

    waist_angle, neck_angle, knee_angle, shoulder_angle = None, None, None, None
    if shoulder_mid and hip_mid and knee_mid:
        raw = calculate_angle(shoulder_mid, hip_mid, knee_mid)
        if raw is not None: waist_angle = max(0, 180 - raw)
    if nose and neck and hip_mid:
        raw = calculate_angle(nose, neck, hip_mid)
        if raw is not None: neck_angle = abs(180 - raw)
    
    knee_values = []
    lk_raw = calculate_angle(l_hip, l_knee, l_ankle)
    rk_raw = calculate_angle(r_hip, r_knee, r_ankle)
    if lk_raw is not None: knee_values.append(max(0, 180 - lk_raw))
    if rk_raw is not None: knee_values.append(max(0, 180 - rk_raw))
    if knee_values: knee_angle = sum(knee_values) / len(knee_values)

    shoulder_values = []
    ls_angle = calculate_angle(l_hip, l_sh, l_elbow)
    rs_angle = calculate_angle(r_hip, r_sh, r_elbow)
    if ls_angle is not None: shoulder_values.append(ls_angle)
    if rs_angle is not None: shoulder_values.append(rs_angle)
    if shoulder_values: shoulder_angle = max(shoulder_values)

    return waist_angle, neck_angle, knee_angle, shoulder_angle

def person_center(person):
    pts = [get_draw_point(person, i) for i in range(25)]
    valid_pts = [p for p in pts if p is not None]
    if len(valid_pts) < 4: return None
    return mean_point(valid_pts)

def initialize_tracks():
    tracks = {}
    for i in range(1, MAX_WORKERS + 1):
        tracks[f"W{i:02d}"] = {
            "active": False, "center": None, "last_seen": 0,
            "waist_angle": None, "neck_angle": None, "knee_angle": None, "shoulder_angle": None,
            "waist_valid": False, "neck_valid": False, "knee_valid": False, "shoulder_valid": False,
            "waist_display": "보류", "neck_display": "보류", "knee_display": "보류", "shoulder_display": "보류",
            "waist_record": "보류", "neck_record": "보류", "knee_record": "보류", "shoulder_record": "보류",
            "waist_hold": 0.0, "neck_hold": 0.0, "knee_hold": 0.0, "shoulder_hold": 0.0,
            "waist_time": 0.0, "neck_time": 0.0, "knee_time": 0.0, "shoulder_time": 0.0,
            "overall": "미인식", "color": GRAY,
            "reba_score": 0, "reba_level": "보류", "reba_color": GRAY, "reba_parts": {"허리": 0, "목": 0, "무릎": 0, "어깨": 0},
            "reba_sum": 0.0, "work_type": "일반 작업", "risk_parts": "없음"
        }
    return tracks

def assign_worker_ids(people, tracks, current_time):
    detections = []
    for person in people:
        center = person_center(person)
        if center: detections.append({"person": person, "center": center, "assigned": False})

    assignments, used = [], set()
    for det in detections:
        best_wid, best_dist = None, float("inf")
        for wid, tr in tracks.items():
            if wid in used or tr["center"] is None: continue
            d = distance(det["center"], tr["center"])
            if d < best_dist and d <= TRACK_MATCH_DISTANCE:
                best_dist = d; best_wid = wid
        if best_wid:
            assignments.append((best_wid, det["person"], det["center"]))
            used.add(best_wid); det["assigned"] = True

    for det in detections:
        if det["assigned"]: continue
        for wid, tr in tracks.items():
            if wid not in used and (not tr["active"] or current_time - tr["last_seen"] > 2.0):
                assignments.append((wid, det["person"], det["center"]))
                used.add(wid); break
    return assignments

def update_record(track, joint, display_risk, dt, is_warmup):
    hold_key, record_key, time_key = f"{joint}_hold", f"{joint}_record", f"{joint}_time"
    hold_required = {"waist": 1.0, "neck": 2.0, "knee": 1.0, "shoulder": 2.0}[joint]

    if display_risk in ["주의", "위험"]: track[hold_key] += dt
    else:
        track[hold_key] = 0.0; track[record_key] = display_risk
        return

    if track[hold_key] >= hold_required:
        track[record_key] = display_risk
        if not is_warmup: track[time_key] += dt
    else:
        track[record_key] = "관찰"

# --- 화면 오버레이 및 그리기 함수 ---
def draw_angle_label(img, text, point, color):
    if point is None: return img
    x, y = int(point[0]), int(point[1])
    x = max(0, min(x + 8, img.shape[1] - 120))
    y = max(0, min(y - 28, img.shape[0] - 25))
    cv2.rectangle(img, (x, y), (x + 115, y + 24), (15, 15, 15), -1)
    cv2.rectangle(img, (x, y), (x + 115, y + 24), color, 1)
    return draw_korean_text(img, text, (x + 5, y + 2), 13, color)

def draw_openpose_risk_overlay(frame, person, tr):
    hip = get_draw_point(person, MID_HIP); neck = get_draw_point(person, NECK)
    l_sh = get_draw_point(person, L_SHOULDER); r_sh = get_draw_point(person, R_SHOULDER)
    l_el = get_draw_point(person, L_ELBOW); r_el = get_draw_point(person, R_ELBOW)
    l_hip = get_draw_point(person, L_HIP); r_hip = get_draw_point(person, R_HIP)
    l_knee = get_draw_point(person, L_KNEE); r_knee = get_draw_point(person, R_KNEE)
    l_ankle = get_draw_point(person, L_ANKLE); r_ankle = get_draw_point(person, R_ANKLE)

    if l_hip is None: l_hip = hip
    if r_hip is None: r_hip = hip

    def line(p1, p2, color, thick=3):
        if p1 and p2: cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick)

    for p1, p2 in [(neck, hip), (l_sh, l_el), (r_sh, r_el), (l_hip, l_knee), (r_hip, r_knee), (l_knee, l_ankle), (r_knee, r_ankle)]:
        line(p1, p2, (120, 120, 120), 2)

    w_col = classify_waist(tr["waist_angle"], tr["waist_valid"])[1]
    n_col = classify_neck(tr["neck_angle"], tr["neck_valid"])[1]
    k_col = classify_knee(tr["knee_angle"], tr["knee_valid"])[1]
    s_col = classify_shoulder(tr["shoulder_angle"], tr["shoulder_valid"])[1]

    if tr["waist_display"] in ["주의", "위험"]:
        line(neck, hip, w_col, 6); line(hip, l_knee, w_col, 5); line(hip, r_knee, w_col, 5)
        frame = draw_angle_label(frame, f"허리 {tr['waist_angle']:.0f}°", hip, w_col)
    if tr["neck_display"] in ["주의", "위험"]:
        frame = draw_angle_label(frame, f"목 {tr['neck_angle']:.0f}°", neck, n_col)
    if tr["knee_display"] in ["주의", "위험"]:
        line(l_hip, l_knee, k_col, 6); line(r_hip, r_knee, k_col, 6); line(l_knee, l_ankle, k_col, 6); line(r_knee, r_ankle, k_col, 6)
        frame = draw_angle_label(frame, f"무릎 {tr['knee_angle']:.0f}°", mean_point([l_knee, r_knee]), k_col)
    if tr["shoulder_display"] in ["주의", "위험"]:
        line(l_hip, l_sh, s_col, 5); line(r_hip, r_sh, s_col, 5); line(l_sh, l_el, s_col, 6); line(r_sh, r_el, s_col, 6)
        frame = draw_angle_label(frame, f"어깨 {tr['shoulder_angle']:.0f}°", mean_point([l_sh, r_sh]), s_col)
    return frame

def draw_worker_label(frame, center, wid, tr):
    x, y = int(center[0]), int(center[1])
    label_w, label_h = 205, 78
    lx, ly = max(5, min(x - label_w // 2, frame.shape[1] - label_w - 5)), y - 120
    if ly < 5: ly = y + 25
    ly = max(5, min(ly, frame.shape[0] - label_h - 5))

    overlay = frame.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    cv2.rectangle(frame, (lx, ly), (lx + label_w, ly + label_h), (70, 70, 70), 1)

    hold_txt = "HOLD" if tr.get("active") else "OFF"
    frame = draw_korean_text(frame, f"{wid}  {hold_txt}", (lx + 8, ly + 6), 12, WHITE)
    frame = draw_korean_text(frame, f"W:{tr['waist_time']:.0f} N:{tr['neck_time']:.0f} K:{tr['knee_time']:.0f} S:{tr['shoulder_time']:.0f}", (lx + 8, ly + 24), 10, WHITE)
    frame = draw_korean_text(frame, f"Type: {tr.get('work_type', 'Normal')}", (lx + 8, ly + 42), 10, GREEN)
    frame = draw_korean_text(frame, f"Risk: {tr.get('risk_parts', 'None')}", (lx + 8, ly + 58), 10, ORANGE)
    return frame

def draw_worker_status_panel(dashboard, tracks, elapsed):
    x0, y0, w0, h0 = RIGHT_X, RIGHT_Y, RIGHT_W, RIGHT_H
    overlay = dashboard.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w0, y0 + h0), (18, 18, 18), -1)
    dashboard = cv2.addWeighted(overlay, 0.90, dashboard, 0.10, 0)
    cv2.rectangle(dashboard, (x0, y0), (x0 + w0, y0 + h0), (60, 60, 60), 1)

    dashboard = draw_korean_text(dashboard, "Worker Summary", (x0 + 12, y0 + 12), 20, WHITE)
    dashboard = draw_korean_text(dashboard, "Cumulative risk time", (x0 + 12, y0 + 42), 11, GRAY)

    card_x, card_w, card_h, y = x0 + 10, w0 - 20, 92, y0 + 70
    for wid, tr in tracks.items():
        if y + card_h > y0 + h0 - 10: break
        cv2.rectangle(dashboard, (card_x, y), (card_x + card_w, y + card_h), (24, 24, 24), -1)
        cv2.rectangle(dashboard, (card_x, y), (card_x + card_w, y + card_h), (90, 90, 90), 1)

        state = "ON" if tr.get("active") else "OFF"
        dashboard = draw_korean_text(dashboard, f"{wid}  {state}  {tr.get('overall', 'HOLD')}", (card_x + 8, y + 8), 12, tr.get("color", GRAY))
        dashboard = draw_korean_text(dashboard, f"W:{format_time(tr['waist_time'])}   N:{format_time(tr['neck_time'])}", (card_x + 8, y + 31), 10, WHITE)
        dashboard = draw_korean_text(dashboard, f"K:{format_time(tr['knee_time'])}   S:{format_time(tr['shoulder_time'])}", (card_x + 8, y + 50), 10, WHITE)
        most_joint, most_time = get_most_risky_joint(tr['waist_time'], tr['neck_time'], tr['knee_time'], tr['shoulder_time'])
        dashboard = draw_korean_text(dashboard, f"Most: {most_joint} {format_time(most_time)}", (card_x + 8, y + 70), 10, GRAY)
        y += card_h + 12
    return dashboard

def draw_risk_time_panel(dashboard, tracks, elapsed, total_reba_score):
    x0, y0, w0, h0 = BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H
    overlay = dashboard.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w0, y0 + h0), (14, 14, 14), -1)
    dashboard = cv2.addWeighted(overlay, 0.86, dashboard, 0.14, 0)
    cv2.rectangle(dashboard, (x0, y0), (x0 + w0, y0 + h0), (60, 60, 60), 1)

    t_w = sum(tr["waist_time"] for tr in tracks.values())
    t_n = sum(tr["neck_time"] for tr in tracks.values())
    t_k = sum(tr["knee_time"] for tr in tracks.values())
    t_s = sum(tr["shoulder_time"] for tr in tracks.values())
    most_joint, most_time = get_most_risky_joint(t_w, t_n, t_k, t_s)

    items = [
        ("Waist", format_time(t_w), YELLOW), ("Neck", format_time(t_n), WHITE),
        ("Knee", format_time(t_k), ORANGE), ("Shoulder", format_time(t_s), WHITE),
        ("Total REBA", f"{total_reba_score:.1f}", YELLOW), ("AVG Most", f"{most_joint} {format_time(most_time)}", GREEN)
    ]

    box_w, gap, x = 150, 7, x0 + 8
    for name, value, color in items:
        cv2.rectangle(dashboard, (x, y0 + 10), (x + box_w, y0 + h0 - 8), (20, 20, 20), -1)
        cv2.rectangle(dashboard, (x, y0 + 10), (x + box_w, y0 + h0 - 8), (75, 75, 75), 1)
        dashboard = draw_korean_text(dashboard, name, (x + 8, y0 + 16), 10, color)
        dashboard = draw_korean_text(dashboard, value, (x + 8, y0 + 38), 11, color)
        x += box_w + gap
    return dashboard

# ==========================================
# WebRTC 프로세서 클래스 (메인 루프 대체)
# ==========================================
class OpenPoseProcessor(VideoProcessorBase):
    def __init__(self):
        # OpenPose Wrapper 초기화
        params = {
            "model_folder": "models/",
            "model_pose": "BODY_25",
            "number_people_max": MAX_WORKERS,
            "render_pose": 1,
            "net_resolution": "-1x368",
            "scale_number": 1,
            "scale_gap": 0.25
        }
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        self.tracks = initialize_tracks()
        self.start_time = time.time()
        self.prev_time = self.start_time

    def recv(self, frame):
        raw_frame = frame.to_ndarray(format="bgr24")
        
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        elapsed = current_time - self.start_time
        is_warmup = elapsed < WARMUP_SECONDS

        # 비디오 리사이즈
        video_frame = cv2.resize(raw_frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_CUBIC)

        # OpenPose 데이터 세팅 및 추론
        datum = op.Datum()
        datum.cvInputData = video_frame
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.cvOutputData is not None and float(np.mean(datum.cvOutputData)) > 3:
            output = datum.cvOutputData.copy()
        else:
            output = video_frame.copy()

        people = datum.poseKeypoints
        detected_ids = set()

        if people is not None:
            assignments = assign_worker_ids(people, self.tracks, current_time)

            for wid, person, center in assignments:
                detected_ids.add(wid)
                tr = self.tracks[wid]
                tr["active"] = True; tr["center"] = center; tr["last_seen"] = current_time

                waist, neck, knee, shoulder = calculate_openpose_angles(person)
                tr["waist_valid"], tr["neck_valid"], tr["knee_valid"], tr["shoulder_valid"] = waist is not None, neck is not None, knee is not None, shoulder is not None
                tr["waist_angle"] = smooth_value(tr["waist_angle"], waist)
                tr["neck_angle"] = smooth_value(tr["neck_angle"], neck)
                tr["knee_angle"] = smooth_value(tr["knee_angle"], knee)
                tr["shoulder_angle"] = smooth_value(tr["shoulder_angle"], shoulder)

                tr["waist_display"], _ = classify_waist(tr["waist_angle"], tr["waist_valid"])
                tr["neck_display"], _ = classify_neck(tr["neck_angle"], tr["neck_valid"])
                tr["knee_display"], _ = classify_knee(tr["knee_angle"], tr["knee_valid"])
                tr["shoulder_display"], _ = classify_shoulder(tr["shoulder_angle"], tr["shoulder_valid"])

                update_record(tr, "waist", tr["waist_display"], dt, is_warmup)
                update_record(tr, "neck", tr["neck_display"], dt, is_warmup)
                update_record(tr, "knee", tr["knee_display"], dt, is_warmup)
                update_record(tr, "shoulder", tr["shoulder_display"], dt, is_warmup)

                tr["overall"], tr["color"] = get_overall_risk([tr["waist_record"], tr["neck_record"], tr["knee_record"], tr["shoulder_record"]])
                tr["reba_score"], tr["reba_level"], tr["reba_color"], tr["reba_parts"] = calculate_simple_reba(tr["waist_angle"], tr["neck_angle"], tr["knee_angle"], tr["shoulder_angle"], tr["waist_valid"], tr["neck_valid"], tr["knee_valid"], tr["shoulder_valid"])
                tr["work_type"] = classify_owas_type(tr["waist_angle"], tr["neck_angle"], tr["knee_angle"], tr["shoulder_angle"], tr["waist_valid"], tr["neck_valid"], tr["knee_valid"], tr["shoulder_valid"])
                tr["risk_parts"] = get_risk_parts(tr)

                if not is_warmup and tr["reba_score"] > 0:
                    tr["reba_sum"] += tr["reba_score"] * dt

                output = draw_openpose_risk_overlay(output, person, tr)
                output = draw_worker_label(output, center, wid, tr)

        for wid, tr in self.tracks.items():
            if wid not in detected_ids and current_time - tr["last_seen"] > 2:
                tr["active"] = False; tr["overall"] = "미인식"; tr["color"] = GRAY; tr["reba_score"] = 0
                tr["reba_level"] = "보류"; tr["reba_color"] = GRAY; tr["work_type"] = "일반 작업"; tr["risk_parts"] = "없음"

        # 대시보드 조립 (980 x 580)
        dashboard = np.zeros((DASHBOARD_H, DASHBOARD_W, 3), dtype=np.uint8)
        dashboard[:] = BG
        dashboard[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = output

        status = f"Detected: {0 if people is None else len(people)}  | OpenPose | {'Warmup ' + str(round(WARMUP_SECONDS - elapsed, 1)) + 's' if is_warmup else 'Worker ID tracking'}"
        dashboard = draw_korean_text(dashboard, status, (VIDEO_X + 12, VIDEO_Y + 10), 14, YELLOW if is_warmup else WHITE)
        dashboard = draw_worker_status_panel(dashboard, self.tracks, elapsed)
        
        total_reba_score = sum(tr.get("reba_sum", 0.0) for tr in self.tracks.values())
        dashboard = draw_risk_time_panel(dashboard, self.tracks, elapsed, total_reba_score)

        return frame.from_ndarray(dashboard, format="bgr24")

# ==========================================
# Streamlit 웹 앱 설정
# ==========================================
st.set_page_config(layout="wide")
st.title("OpenPose 기반 작업자 자세 위험도 정밀 분석")
st.markdown("""
* 이 앱은 `pyopenpose`가 로컬 PC 또는 서버에 올바르게 설치되고 빌드되어 있어야 정상 작동합니다.
* 클라우드(Streamlit Cloud)에서 배포하기 위해서는 Docker 기반의 커스텀 환경이 필요합니다.
""")

webrtc_streamer(
    key="openpose-reba-owas",
    video_processor_factory=OpenPoseProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
