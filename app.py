import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO

# 웹 페이지 레이아웃 설정
st.set_page_config(layout="wide")
st.title("작업자 자세 위험도 실시간 분석 (REBA/OWAS)")

# 1. 모델은 클래스 밖에서 한 번만 로드합니다.
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# 2. 들어오는 영상 프레임을 처리하는 클래스
class PoseProcessor(VideoTransformerBase):
    def recv(self, frame):
        # 웹캠에서 들어온 프레임을 OpenCV 이미지 배열로 변환
        img = frame.to_ndarray(format="bgr24")

        # --- [여기에 기존 로직을 넣습니다] ---
        # 1. results = model(img, conf=0.35)
        # 2. detections = parse_results(results)
        # 3. assign_simple_ids 또는 bytetrack 로직 적용
        # 4. update_worker_posture()로 각도/점수 계산
        # 5. out = draw_label(...)로 화면에 그리기
        # ------------------------------------
        
        # (테스트용: 기본 YOLO 결과만 그리기)
        results = model(img, verbose=False, conf=0.35)
        if results and len(results) > 0:
             img = results[0].plot()

        # 처리된 이미지를 다시 웹 브라우저로 반환
        return frame.from_ndarray(img, format="bgr24")

# 3. 화면에 웹캠 플레이어 띄우기
webrtc_streamer(
    key="reba-owas-pose",
    video_processor_factory=PoseProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)