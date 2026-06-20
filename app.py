import os
import cv2
import math
import time
import csv
import numpy as np
import streamlit as st
import mediapipe as mp

from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from av import VideoFrame


# ==========================================
# 1. 전역 설정 및 상수
# ==========================================
MAX_WORKERS = 3
WARMUP_SECONDS = 8.0

WAIST_HOLD_SECONDS = 1.0
NECK_HOLD_SECONDS = 2.0
KNEE_HOLD_SECONDS = 1.0
SHOULDER_HOLD_SECONDS = 2.0

DRAW_VISIBILITY_THRESHOLD = 0.35
ANGLE_VISIBILITY_THRESHOLD = 0.60
TRACK_MATCH_DISTANCE = 220
SMOOTHING_ALPHA = 0.35

DASHBOARD_W, DASHBOARD_H = 1280, 720
VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H = 20, 85, 820, 470
RIGHT_X, RIGHT_Y, RIGHT_W = 860, 85, 400
BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H = 20, 575, 1240, 125

# BGR 색상
BG = (15, 20, 25)
PANEL = (32, 38, 44)
WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
GREEN = (0, 220, 60)
YELLOW = (0, 230, 230)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
BLUE = (255, 130, 0)
SKELETON_GRAY = (150, 150, 150)

# 폰트 설정
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


# ==========================================
# 2. 공통 유틸 함수
# ==========================================
def draw_korean_text(img, text, position, font_size=22, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()

    # BGR → RGB
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 65, 75), 1)

    if title:
        img = draw_korean_text(img, title, (x + 18, y + 15), 21, WHITE)

    return img


def draw_progress_bar(img, x, y, w, h, percent, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), -1)
    fill_w = int(w * min(max(percent, 0), 100) / 100)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    return img


def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def smooth_value(old_value, new_value, alpha=SMOOTHING_ALPHA):
    if new_value is None:
        return old_value
    if old_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * old_value


def angle_between_vectors(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return None

    cos_value = np.dot(v1, v2) / (norm1 * norm2)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return abs(math.degrees(math.acos(cos_value)))


def calculate_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return abs(math.degrees(math.acos(cos_value)))


# ==========================================
# 3. 위험도 분류 함수
# ==========================================
def classify_waist_risk(angle, valid=True):
    if not valid or angle is None:
        return "보류", GRAY
    if angle >= 60:
        return "위험", RED
    elif angle >= 20:
        return "주의", YELLOW
    return "안전", GREEN


def classify_neck_risk(angle, valid=True):
    if not valid or angle is None:
        return "보류", GRAY
    if angle >= 45:
        return "위험", RED
    elif angle >= 20:
        return "주의", YELLOW
    return "안전", GREEN


def classify_knee_risk(angle, valid=True):
    if not valid or angle is None:
        return "보류", GRAY
    if angle >= 60:
        return "위험", RED
    elif angle >= 30:
        return "주의", YELLOW
    return "안전", GREEN


def classify_shoulder_risk(angle, valid=True):
    if not valid or angle is None:
        return "보류", GRAY
    if angle >= 90:
        return "위험", RED
    elif angle >= 45:
        return "주의", YELLOW
    return "안전", GREEN


def get_overall_risk(risks):
    valid = [r for r in risks if r in ["안전", "주의", "위험"]]

    if not valid:
        return "보류", GRAY

    danger = valid.count("위험")
    caution = valid.count("주의")

    if danger >= 1:
        return "위험", RED
    elif caution >= 2:
        return "주의", YELLOW
    elif caution == 1:
        return "관찰", ORANGE

    return "안전", GREEN


def get_simple_reba_score(waist, neck, knee, shoulder, wv=True, nv=True, kv=True, sv=True):
    score = 0

    if wv and waist is not None:
        score += 1 if waist < 20 else (2 if waist < 60 else 3)

    if nv and neck is not None:
        score += 1 if neck < 20 else (2 if neck < 45 else 3)

    if kv and knee is not None:
        score += 1 if knee < 30 else (2 if knee < 60 else 3)

    if sv and shoulder is not None:
        score += 1 if shoulder < 45 else (2 if shoulder < 90 else 3)

    return score


def classify_reba_level(score):
    if score >= 9:
        return "높음", RED
    elif score >= 6:
        return "중간", YELLOW
    elif score >= 1:
        return "낮음", GREEN
    return "보류", GRAY


def classify_work_type(waist, knee, shoulder, wv=True, kv=True, sv=True):
    waist_v = waist if wv and waist is not None else 0
    knee_v = knee if kv and knee is not None else 0
    shoulder_v = shoulder if sv and shoulder is not None else 0

    if waist_v < 20 and knee_v < 30 and shoulder_v < 45:
        return "P0 일반 자세"

    if waist_v >= 20 and knee_v < 30 and shoulder_v < 45:
        return "P1 몸통 굽힘"

    if waist_v >= 20 and knee_v >= 30:
        return "P2 몸통·무릎 굽힘"

    if waist_v < 20 and shoulder_v >= 90:
        return "P3 상지 거상"

    if waist_v >= 20 and shoulder_v >= 45:
        return "P4 몸통 굽힘·상지 사용"

    return "기타 자세"


# ==========================================
# 4. MediaPipe 랜드마크 계산 함수
# ==========================================
def get_landmark_point(landmarks, idx, crop_w, crop_h, offset_x, offset_y):
    lm = landmarks[idx]
    x = int(offset_x + lm.x * crop_w)
    y = int(offset_y + lm.y * crop_h)
    visibility = lm.visibility
    return (x, y, visibility)


def valid_point(p, threshold=ANGLE_VISIBILITY_THRESHOLD):
    return p is not None and p[2] >= threshold


def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None

    return (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2),
        min(p1[2], p2[2])
    )


def calculate_body_angles(landmarks, crop_w, crop_h, offset_x, offset_y):
    lm = mp.solutions.pose.PoseLandmark

    nose = get_landmark_point(landmarks, lm.NOSE.value, crop_w, crop_h, offset_x, offset_y)

    l_sh = get_landmark_point(landmarks, lm.LEFT_SHOULDER.value, crop_w, crop_h, offset_x, offset_y)
    r_sh = get_landmark_point(landmarks, lm.RIGHT_SHOULDER.value, crop_w, crop_h, offset_x, offset_y)

    l_el = get_landmark_point(landmarks, lm.LEFT_ELBOW.value, crop_w, crop_h, offset_x, offset_y)
    r_el = get_landmark_point(landmarks, lm.RIGHT_ELBOW.value, crop_w, crop_h, offset_x, offset_y)

    l_hip = get_landmark_point(landmarks, lm.LEFT_HIP.value, crop_w, crop_h, offset_x, offset_y)
    r_hip = get_landmark_point(landmarks, lm.RIGHT_HIP.value, crop_w, crop_h, offset_x, offset_y)

    l_knee = get_landmark_point(landmarks, lm.LEFT_KNEE.value, crop_w, crop_h, offset_x, offset_y)
    r_knee = get_landmark_point(landmarks, lm.RIGHT_KNEE.value, crop_w, crop_h, offset_x, offset_y)

    l_ankle = get_landmark_point(landmarks, lm.LEFT_ANKLE.value, crop_w, crop_h, offset_x, offset_y)
    r_ankle = get_landmark_point(landmarks, lm.RIGHT_ANKLE.value, crop_w, crop_h, offset_x, offset_y)

    sh_mid = midpoint(l_sh, r_sh)
    hip_mid = midpoint(l_hip, r_hip)

    # 허리 각도: 골반-어깨 벡터가 수직축에서 얼마나 벗어나는지
    waist_angle = None
    waist_valid = False

    if valid_point(sh_mid) and valid_point(hip_mid):
        trunk_vec = (sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        waist_angle = angle_between_vectors(trunk_vec, (0, -1))
        waist_valid = waist_angle is not None

    # 목 각도: 어깨-코 벡터가 수직축에서 얼마나 벗어나는지
    neck_angle = None
    neck_valid = False

    if valid_point(sh_mid) and valid_point(nose):
        neck_vec = (nose[0] - sh_mid[0], nose[1] - sh_mid[1])
        neck_angle = angle_between_vectors(neck_vec, (0, -1))
        neck_valid = neck_angle is not None

    # 무릎 굽힘 각도: 180도에서 실제 무릎각을 빼서 굽힘 정도로 변환
    knee_values = []

    if valid_point(l_hip) and valid_point(l_knee) and valid_point(l_ankle):
        raw = calculate_angle(l_hip, l_knee, l_ankle)
        if raw is not None:
            knee_values.append(max(0, 180 - raw))

    if valid_point(r_hip) and valid_point(r_knee) and valid_point(r_ankle):
        raw = calculate_angle(r_hip, r_knee, r_ankle)
        if raw is not None:
            knee_values.append(max(0, 180 - raw))

    knee_angle = max(knee_values) if knee_values else None
    knee_valid = knee_angle is not None

    # 어깨 각도: 팔이 아래로 내려간 상태를 0도, 들릴수록 증가
    shoulder_values = []

    if valid_point(l_sh) and valid_point(l_el):
        upper_arm_vec = (l_el[0] - l_sh[0], l_el[1] - l_sh[1])
        angle = angle_between_vectors(upper_arm_vec, (0, 1))
        if angle is not None:
            shoulder_values.append(angle)

    if valid_point(r_sh) and valid_point(r_el):
        upper_arm_vec = (r_el[0] - r_sh[0], r_el[1] - r_sh[1])
        angle = angle_between_vectors(upper_arm_vec, (0, 1))
        if angle is not None:
            shoulder_values.append(angle)

    shoulder_angle = max(shoulder_values) if shoulder_values else None
    shoulder_valid = shoulder_angle is not None

    points = {
        "nose": nose,
        "l_sh": l_sh,
        "r_sh": r_sh,
        "l_el": l_el,
        "r_el": r_el,
        "l_hip": l_hip,
        "r_hip": r_hip,
        "l_knee": l_knee,
        "r_knee": r_knee,
        "l_ankle": l_ankle,
        "r_ankle": r_ankle
    }

    return {
        "waist": waist_angle,
        "neck": neck_angle,
        "knee": knee_angle,
        "shoulder": shoulder_angle,
        "waist_valid": waist_valid,
        "neck_valid": neck_valid,
        "knee_valid": knee_valid,
        "shoulder_valid": shoulder_valid,
        "points": points
    }


def draw_skeleton(img, points):
    pairs = [
        ("l_sh", "r_sh"),
        ("l_sh", "l_el"),
        ("r_sh", "r_el"),
        ("l_sh", "l_hip"),
        ("r_sh", "r_hip"),
        ("l_hip", "r_hip"),
        ("l_hip", "l_knee"),
        ("r_hip", "r_knee"),
        ("l_knee", "l_ankle"),
        ("r_knee", "r_ankle"),
        ("nose", "l_sh"),
        ("nose", "r_sh")
    ]

    for a, b in pairs:
        p1 = points.get(a)
        p2 = points.get(b)

        if p1 is not None and p2 is not None and p1[2] >= DRAW_VISIBILITY_THRESHOLD and p2[2] >= DRAW_VISIBILITY_THRESHOLD:
            cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), SKELETON_GRAY, 2)

    for p in points.values():
        if p is not None and p[2] >= DRAW_VISIBILITY_THRESHOLD:
            cv2.circle(img, (p[0], p[1]), 4, WHITE, -1)

    return img


# ==========================================
# 5. 모델 로딩
# ==========================================
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


yolo_model = load_yolo_model()


# ==========================================
# 6. WebRTC Processor
# ==========================================
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.tracks = {}
        for i in range(1, MAX_WORKERS + 1):
            self.tracks[f"W{i:02d}"] = self.create_empty_track()

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.frame_number = 0

        self.run_mode = "평가모드"
        self.experiment_condition = "조건 미설정"

        self.last_csv_time = 0.0
        self.csv_path = f"/tmp/pose_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()

    def create_empty_track(self):
        return {
            "active": False,
            "center": None,
            "last_seen": 0,

            "waist_angle": None,
            "neck_angle": None,
            "knee_angle": None,
            "shoulder_angle": None,

            "waist_valid": False,
            "neck_valid": False,
            "knee_valid": False,
            "shoulder_valid": False,

            "waist_risk": "보류",
            "neck_risk": "보류",
            "knee_risk": "보류",
            "shoulder_risk": "보류",

            "waist_hold": 0.0,
            "neck_hold": 0.0,
            "knee_hold": 0.0,
            "shoulder_hold": 0.0,

            "waist_time": 0.0,
            "neck_time": 0.0,
            "knee_time": 0.0,
            "shoulder_time": 0.0,

            "overall_risk": "보류",
            "overall_color": GRAY,

            "reba_score": 0,
            "reba_level": "보류",
            "reba_color": GRAY,
            "work_type": "미인식"
        }

    def init_csv(self):
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "time_sec",
                    "condition",
                    "worker_id",
                    "active",
                    "waist",
                    "neck",
                    "knee",
                    "shoulder",
                    "waist_risk",
                    "neck_risk",
                    "knee_risk",
                    "shoulder_risk",
                    "reba",
                    "reba_level",
                    "work_type",
                    "waist_time",
                    "neck_time",
                    "knee_time",
                    "shoulder_time",
                    "overall"
                ])
        except:
            pass

    def write_csv(self, elapsed_time):
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)

                for wid, tr in self.tracks.items():
                    writer.writerow([
                        round(elapsed_time, 2),
                        self.experiment_condition,
                        wid,
                        tr["active"],
                        None if tr["waist_angle"] is None else round(tr["waist_angle"], 2),
                        None if tr["neck_angle"] is None else round(tr["neck_angle"], 2),
                        None if tr["knee_angle"] is None else round(tr["knee_angle"], 2),
                        None if tr["shoulder_angle"] is None else round(tr["shoulder_angle"], 2),
                        tr["waist_risk"],
                        tr["neck_risk"],
                        tr["knee_risk"],
                        tr["shoulder_risk"],
                        tr["reba_score"],
                        tr["reba_level"],
                        tr["work_type"],
                        round(tr["waist_time"], 2),
                        round(tr["neck_time"], 2),
                        round(tr["knee_time"], 2),
                        round(tr["shoulder_time"], 2),
                        tr["overall_risk"]
                    ])
        except:
            pass

    def assign_worker_ids(self, person_boxes, current_time):
        assignments = []
        used_ids = set()

        detections = []
        for box in person_boxes:
            x1, y1, x2, y2, conf, area = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({"box": box, "center": (cx, cy), "assigned": False})

        for det in detections:
            best_wid = None
            best_dist = float("inf")

            for wid, tr in self.tracks.items():
                if wid in used_ids:
                    continue
                if tr["center"] is None:
                    continue

                d = distance(det["center"], tr["center"])
                if d < best_dist and d <= TRACK_MATCH_DISTANCE:
                    best_dist = d
                    best_wid = wid

            if best_wid:
                assignments.append((best_wid, det["box"], det["center"]))
                used_ids.add(best_wid)
                det["assigned"] = True

        for det in detections:
            if det["assigned"]:
                continue

            free_wid = None
            for wid, tr in self.tracks.items():
                if wid not in used_ids and (not tr["active"] or current_time - tr["last_seen"] > 2.0):
                    free_wid = wid
                    break

            if free_wid:
                assignments.append((free_wid, det["box"], det["center"]))
                used_ids.add(free_wid)

        return assignments

    def update_hold_and_time(self, tr, dt, is_warmup):
        hold_map = {
            "waist": WAIST_HOLD_SECONDS,
            "neck": NECK_HOLD_SECONDS,
            "knee": KNEE_HOLD_SECONDS,
            "shoulder": SHOULDER_HOLD_SECONDS
        }

        for part in ["waist", "neck", "knee", "shoulder"]:
            risk_key = f"{part}_risk"
            hold_key = f"{part}_hold"
            time_key = f"{part}_time"

            if tr[risk_key] in ["주의", "위험"]:
                tr[hold_key] += dt
            else:
                tr[hold_key] = 0.0

            if not is_warmup and tr[hold_key] >= hold_map[part]:
                tr[time_key] += dt

    def recv(self, frame):
        try:
            raw_frame = frame.to_ndarray(format="bgr24")
            analysis_frame = cv2.flip(raw_frame, 1)
            analysis_frame = cv2.resize(analysis_frame, (VIDEO_W, VIDEO_H))
            video_frame = analysis_frame.copy()

            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
            elapsed_time = current_time - self.start_time
            is_warmup = elapsed_time < WARMUP_SECONDS

            self.frame_number += 1

            # YOLO 사람 검출
            yolo_results = yolo_model.predict(
                analysis_frame,
                classes=[0],
                conf=0.35,
                imgsz=416,
                verbose=False
            )

            person_boxes = []
            if len(yolo_results) > 0:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)

                    if area >= 5000:
                        person_boxes.append((x1, y1, x2, y2, conf, area))

            person_boxes = sorted(person_boxes, key=lambda b: b[5], reverse=True)[:MAX_WORKERS]
            assignments = self.assign_worker_ids(person_boxes, current_time)
            detected_ids = set()

            for wid, person_box, center in assignments:
                detected_ids.add(wid)

                x1, y1, x2, y2, conf, area = person_box

                pad = 35
                x1p = max(x1 - pad, 0)
                y1p = max(y1 - pad, 0)
                x2p = min(x2 + pad, VIDEO_W - 1)
                y2p = min(y2 + pad, VIDEO_H - 1)

                crop = analysis_frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                crop_h, crop_w, _ = crop.shape

                tr = self.tracks[wid]
                tr["active"] = True
                tr["center"] = center
                tr["last_seen"] = current_time

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results = self.pose_model.process(crop_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    angle_data = calculate_body_angles(landmarks, crop_w, crop_h, x1p, y1p)

                    waist = angle_data["waist"]
                    neck = angle_data["neck"]
                    knee = angle_data["knee"]
                    shoulder = angle_data["shoulder"]

                    tr["waist_angle"] = smooth_value(tr["waist_angle"], waist)
                    tr["neck_angle"] = smooth_value(tr["neck_angle"], neck)
                    tr["knee_angle"] = smooth_value(tr["knee_angle"], knee)
                    tr["shoulder_angle"] = smooth_value(tr["shoulder_angle"], shoulder)

                    tr["waist_valid"] = angle_data["waist_valid"]
                    tr["neck_valid"] = angle_data["neck_valid"]
                    tr["knee_valid"] = angle_data["knee_valid"]
                    tr["shoulder_valid"] = angle_data["shoulder_valid"]

                    waist_risk, waist_color = classify_waist_risk(tr["waist_angle"], tr["waist_valid"])
                    neck_risk, neck_color = classify_neck_risk(tr["neck_angle"], tr["neck_valid"])
                    knee_risk, knee_color = classify_knee_risk(tr["knee_angle"], tr["knee_valid"])
                    shoulder_risk, shoulder_color = classify_shoulder_risk(tr["shoulder_angle"], tr["shoulder_valid"])

                    tr["waist_risk"] = waist_risk
                    tr["neck_risk"] = neck_risk
                    tr["knee_risk"] = knee_risk
                    tr["shoulder_risk"] = shoulder_risk

                    overall, overall_color = get_overall_risk([
                        waist_risk,
                        neck_risk,
                        knee_risk,
                        shoulder_risk
                    ])

                    tr["overall_risk"] = overall
                    tr["overall_color"] = overall_color

                    tr["reba_score"] = get_simple_reba_score(
                        tr["waist_angle"],
                        tr["neck_angle"],
                        tr["knee_angle"],
                        tr["shoulder_angle"],
                        tr["waist_valid"],
                        tr["neck_valid"],
                        tr["knee_valid"],
                        tr["shoulder_valid"]
                    )

                    tr["reba_level"], tr["reba_color"] = classify_reba_level(tr["reba_score"])

                    tr["work_type"] = classify_work_type(
                        tr["waist_angle"],
                        tr["knee_angle"],
                        tr["shoulder_angle"],
                        tr["waist_valid"],
                        tr["knee_valid"],
                        tr["shoulder_valid"]
                    )

                    self.update_hold_and_time(tr, dt, is_warmup)

                    video_frame = draw_skeleton(video_frame, angle_data["points"])

                color = self.tracks[wid]["overall_color"]
                cv2.rectangle(video_frame, (x1p, y1p), (x2p, y2p), color, 2)
                video_frame = draw_korean_text(video_frame, wid, (x1p, max(0, y1p - 28)), 20, color)

                info_text = f"{self.tracks[wid]['overall_risk']} | {self.tracks[wid]['work_type']}"
                video_frame = draw_korean_text(video_frame, info_text, (x1p, y2p + 5), 16, color)

            # 미검출 작업자 비활성화
            for wid, tr in self.tracks.items():
                if wid not in detected_ids and current_time - tr["last_seen"] > 1.5:
                    tr["active"] = False

            # CSV 저장
            if current_time - self.last_csv_time >= 1.0:
                self.write_csv(elapsed_time)
                self.last_csv_time = current_time

            # 발표모드 대시보드
            if self.run_mode == "발표모드":
                dashboard = np.zeros((DASHBOARD_H, DASHBOARD_W, 3), dtype=np.uint8)
                dashboard[:] = BG

                cv2.rectangle(dashboard, (0, 0), (DASHBOARD_W, 75), (10, 14, 18), -1)
                dashboard = draw_korean_text(
                    dashboard,
                    "작업발판 높이 조정 전·후 위험 자세 비교 시스템",
                    (170, 20),
                    26,
                    WHITE
                )

                dashboard = draw_panel(
                    dashboard,
                    VIDEO_X - 5,
                    VIDEO_Y - 5,
                    VIDEO_W + 10,
                    VIDEO_H + 10
                )

                dashboard[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = video_frame

                status_text = (
                    f"조건: {self.experiment_condition} | 초기 안정화 중 {max(0, WARMUP_SECONDS - elapsed_time):.1f}초"
                    if is_warmup
                    else f"조건: {self.experiment_condition} | 실시간 분석 중"
                )

                dashboard = draw_korean_text(
                    dashboard,
                    status_text,
                    (VIDEO_X + 14, VIDEO_Y + 6),
                    16,
                    YELLOW if is_warmup else WHITE
                )

                # 오른쪽 패널
                dashboard = draw_panel(dashboard, RIGHT_X, RIGHT_Y, RIGHT_W, 470, "작업자별 분석 결과")

                yy = RIGHT_Y + 60
                for wid, tr in self.tracks.items():
                    color = tr["overall_color"] if tr["active"] else GRAY
                    active_text = "인식 중" if tr["active"] else "미인식"

                    dashboard = draw_korean_text(
                        dashboard,
                        f"{wid} | {active_text} | {tr['overall_risk']}",
                        (RIGHT_X + 20, yy),
                        18,
                        color
                    )

                    dashboard = draw_korean_text(
                        dashboard,
                        f"자세: {tr['work_type']}",
                        (RIGHT_X + 20, yy + 28),
                        15,
                        WHITE
                    )

                    dashboard = draw_korean_text(
                        dashboard,
                        f"허리 {tr['waist_risk']} / 무릎 {tr['knee_risk']} / 어깨 {tr['shoulder_risk']}",
                        (RIGHT_X + 20, yy + 52),
                        14,
                        GRAY
                    )

                    dashboard = draw_korean_text(
                        dashboard,
                        f"REBA 간이점수: {tr['reba_score']} ({tr['reba_level']})",
                        (RIGHT_X + 20, yy + 76),
                        14,
                        tr["reba_color"]
                    )

                    yy += 125

                # 하단 누적시간 패널
                dashboard = draw_panel(dashboard, BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H, "신체부위별 누적 위험시간")

                total_waist = sum(tr["waist_time"] for tr in self.tracks.values())
                total_neck = sum(tr["neck_time"] for tr in self.tracks.values())
                total_knee = sum(tr["knee_time"] for tr in self.tracks.values())
                total_shoulder = sum(tr["shoulder_time"] for tr in self.tracks.values())

                totals = [
                    ("허리", total_waist, RED),
                    ("목", total_neck, ORANGE),
                    ("무릎", total_knee, YELLOW),
                    ("어깨", total_shoulder, BLUE)
                ]

                bx = BOTTOM_X + 30
                by = BOTTOM_Y + 58

                max_time = max([t[1] for t in totals] + [1.0])

                for name, value, color in totals:
                    dashboard = draw_korean_text(dashboard, f"{name}: {value:.1f}s", (bx, by - 25), 15, WHITE)
                    dashboard = draw_progress_bar(dashboard, bx, by, 250, 18, value / max_time * 100, color)
                    bx += 300

                return VideoFrame.from_ndarray(dashboard, format="bgr24")

            # 평가모드
            return VideoFrame.from_ndarray(video_frame, format="bgr24")

        except Exception as e:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            error_frame[:] = (30, 30, 30)
            error_frame = draw_korean_text(error_frame, "영상 처리 중 오류 발생", (60, 180), 28, RED)
            error_frame = draw_korean_text(error_frame, str(e)[:60], (60, 230), 16, WHITE)
            return VideoFrame.from_ndarray(error_frame, format="bgr24")


# ==========================================
# 7. Streamlit UI
# ==========================================
st.set_page_config(layout="wide", page_title="현장 자세 위험도 분석")
st.title("건설 현장 작업자 다중 자세 분석")

with st.sidebar:
    st.header("실험 설정")

    selected_mode = st.selectbox(
        "실행 모드",
        ["발표모드", "평가모드"],
        index=0
    )

    selected_condition = st.selectbox(
        "실험 조건",
        ["작업발판 조정 전", "작업발판 조정 후", "모델 정확도 검증"],
        index=0
    )

    st.markdown("---")
    st.write("📌 **안내사항**")
    st.write("- **발표모드:** 전체 대시보드 UI가 표시됩니다.")
    st.write("- **평가모드:** 대시보드 없이 영상 화면만 표시됩니다.")
    st.write("- 처음 8초는 초기 안정화 시간입니다.")
    st.write("- 결과 데이터는 서버의 임시 CSV로 기록됩니다.")


ctx = webrtc_streamer(
    key="yolo-mediapipe-pose",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PoseProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


if ctx.video_processor:
    ctx.video_processor.run_mode = selected_mode
    ctx.video_processor.experiment_condition = selected_condition
