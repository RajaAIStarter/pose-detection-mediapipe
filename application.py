import streamlit as st
import cv2
import mediapipe as mp
import time
import joblib
import numpy as np

# -------------------------
# Global Styling and Config
# -------------------------
st.set_page_config(page_title="Pose Suggestion App", layout="wide")
st.markdown("""
    <style>
      .stApp {
          background: url("https://images.unsplash.com/photo-1592580715317-19adca36288e?q=80&w=2070&auto=format&fit=crop") no-repeat center center fixed;
          background-size: cover;
      }
      body {
          color: white;
      }
      h1, h2, h3, h4, p, label {
          color: white !important;
      }
      .cards-container {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          padding-top: 100px;
      }
      .card {
          background-color: rgba(255, 255, 255, 0.85);
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.2);
          padding: 10px;
          margin: 10px;
          text-align: center;
          width: 350px;
          color: black;
          transition: transform 0.3s ease, box-shadow 0.3s ease;
          cursor: default;
      }
      .card:hover {
          transform: scale(1.05);
          box-shadow: 0 8px 16px rgba(0,0,0,0.4);
      }
      .card img {
          width: 100%;
          border-radius: 10px 10px 0 0;
      }
      .card-title {
          margin: 10px 0;
          font-size: 1.2rem;
          font-weight: bold;
          color: black;
      }
      .try-now-container {
          width: 100%;
          max-width: 640px;
          height: 480px;
          background-color: rgba(255,255,255,0.95);
          border-radius: 10px;
          box-shadow: 0 4px 8px rgba(0,0,0,0.2);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto;
      }
      .try-now-button {
          padding: 15px 30px;
          font-size: 24px;
          border: none;
          border-radius: 5px;
          background-color: #4CAF50;
          color: white;
          cursor: pointer;
          transition: background-color 0.3s ease;
      }
      .try-now-button:hover {
          background-color: #45a049;
      }
      /* Custom CSS for small buttons (e.g., the "Select" buttons and others) */
      div.stButton > button {
          background-color: #ff0000 !important;
          color: white !important;
          border: none;
          border-radius: 5px;
          padding: 8px 16px;
      }
      div.stButton > button:hover {
          background-color: #cc0000 !important;
      }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data(show_spinner=False)
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

@st.cache_resource(show_spinner=False)
def load_classifier():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, le

# -------------------------
# Classifier Constants & Landmarks
# -------------------------
CONF_THRESHOLD = 0.56
EMOJI_CONF_THRESHOLD = 0.78

landmark_indices = {
    'Left Shoulder': 11,
    'Right Shoulder': 12,
    'Left Elbow': 13,
    'Right Elbow': 14,
    'Left Wrist': 15,
    'Right Wrist': 16,
    'Left Hip': 23,
    'Right Hip': 24,
    'Left Knee': 25,
    'Right Knee': 26,
    'Left Ankle': 27,
    'Right Ankle': 28
}

ordered_landmarks = [
    'Left Shoulder',
    'Right Shoulder',
    'Left Elbow',
    'Right Elbow',
    'Left Wrist',
    'Right Wrist',
    'Left Hip',
    'Right Hip',
    'Left Knee',
    'Right Knee',
    'Left Ankle',
    'Right Ankle'
]

# -------------------------
# Navigation: Home vs Focus
# -------------------------
# Each card is defined as (image_url, title, focus)
cards_data = [
    ("https://via.placeholder.com/300x200.png?text=Weight+Loss", "Weight Loss", "Weight Loss"),
    ("https://via.placeholder.com/300x200.png?text=Stability", "Stability", "Stability"),
    ("https://via.placeholder.com/300x200.png?text=Strength", "Strength", "Strength"),
    ("https://via.placeholder.com/300x200.png?text=Flexibility+%26+Endurance", "Flexibility & Endurance",
     "Flexibility & Endurance")
]

if 'focus_area' not in st.session_state:
    st.session_state.focus_area = None

if st.session_state.focus_area is None:
    # Home Page: Display Cards in a grid layout
    st.title("Pose Suggestion App")
    st.markdown("<h3 style='text-align: center;'>“Your body hears everything your mind says.”</h3>",
                unsafe_allow_html=True)
    with st.container():
        num_cols = 2  # two cards per row
        rows = [cards_data[i:i + num_cols] for i in range(0, len(cards_data), num_cols)]
        for row in rows:
            cols = st.columns(num_cols)
            for idx, card in enumerate(row):
                with cols[idx]:
                    st.markdown(f"""
                        <div class="card">
                          <img src="{card[0]}" alt="{card[1]}">
                          <div class="card-title">{card[1]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("Select", key=card[2]):
                        st.session_state.focus_area = card[2]
                        st.experimental_rerun()
else:
    # Focus Page: Display Instructions and Real-Time Pose Detection Feed
    st.header(f"Selected Focus: {st.session_state.focus_area}")
    if st.button("Go Back"):
        st.session_state.focus_area = None
        st.experimental_rerun()

    focus_lower = st.session_state.focus_area.lower()

    if focus_lower == "weight loss":
        instructions = [
            "https://dobayouyoga.com/wp-content/uploads/2019/05/f99fce67-96e0-4f44-92df-82e8ce7438e2.jpeg",
            "https://yogajala.com/wp-content/uploads/goddess-pose.jpg",
            "https://i.ytimg.com/vi/KuZOYyH7nw4/maxresdefault.jpg",
            "https://cdn.create.vista.com/api/media/small/638084862/stock-photo-full-length-shirtless-man-green-pants-doing-goddess-pose-yoga",
            "https://www.theyogacollective.com/wp-content/uploads/2019/11/Goddess-Pose-for-Pose-Page-1200x800.jpeg"
        ]
    elif focus_lower == "stability":
        instructions = [
            "https://cdn.yogajournal.com/wp-content/uploads/2022/01/Tree-Pose_Alt-1_2400x1350_Andrew-Clark.jpeg",
            "https://www.gaia.com/wp-content/uploads/TreePose_StephSchwartz.jpg",
            "https://static2.bigstockphoto.com/4/4/7/large1500/74420284.jpg",
            "https://cdn.prod.website-files.com/67691f03eb5bfa3289b3dae7/67691f03eb5bfa3289b3eb4a_tree-pose-in-yoga.jpg",
            "https://c8.alamy.com/comp/RC94HD/concentrated-black-man-doing-tree-pose-RC94HD.jpg"
        ]
    elif focus_lower == "strength":
        instructions = [
            "https://authenticyogascottsdale.com/wp-content/uploads/warrior_yoga_pose.jpg",
            "https://www.verywellfit.com/thmb/56AayW1tVPCe7jSaIq8GB5xvJg4=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/Verywell-03-3567198-Warrior2-aa285698e49a48e5b9e7cb890ae26bb3.jpg",
            "https://srisrischoolofyoga.org/na/wp-content/uploads/2023/01/warrior-pose-three-variations-1-2-3.jpg",
            "https://www.ekhartyoga.com/media/images/articles/content/Warrior-2-yoga-pose-Virabhadrasana-II-Ekhart-Yoga.jpg",
            "https://media.yogauonline.com/app/uploads/2021/04/06034644/website_image_template-1.webp"
        ]
    elif "flexibility" in focus_lower and "endurance" in focus_lower:
        instructions = [
            "https://cdn.yogajournal.com/wp-content/uploads/2021/05/Plank-Pose_Andrew-Clark_2400x1350.jpeg",
            "https://www.ekhartyoga.com/media/images/articles/content/Plank-pose-Esther-Ekhart-Yoga.jpg",
            "https://cdn.yogajournal.com/wp-content/uploads/2021/10/Reverse-Plank-Upward-Facing-Plank_Andrew-Clark_1.jpg",
            "https://bod-blog-assets.prod.cd.beachbodyondemand.com/bod-blog/wp-content/uploads/2023/08/18152241/dolphin-plank-960.png",
            "https://manflowyoga.com/wp-content/uploads/2023/08/dolphin_modification-1024x576.jpg"
        ]
    else:
        instructions = []

    if instructions:
        cols = st.columns([1.5, 1])
        with cols[0]:
            st.markdown("""
            <div style="background-color: rgba(255,255,255,0.95); border-radius: 10px; padding: 10px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2); text-align: center; max-width: 280px;">
            """, unsafe_allow_html=True)
            st.subheader(f"Instructional Poses for {st.session_state.focus_area}")
            st.markdown("<span style='font-weight:900;'>Select a pose to view the reference image:</span>",
                        unsafe_allow_html=True)
            pose_choice = st.radio("Select Pose", [f"Pose {i}" for i in range(1, len(instructions) + 1)], index=0,
                                   horizontal=True)
            pose_index = int(pose_choice.split(" ")[-1]) - 1
            st.image(instructions[pose_index], width=280)
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[1]:
            st.subheader("REAL TIME POSE DETECTION 🧘🏻")
            st.markdown("""
                <style>
                  .stButton button {
                      background-color: #ff0000 !important;
                      color: white !important;
                      border: none;
                      border-radius: 5px;
                      padding: 8px 16px;
                  }
                  .stButton button:hover {
                      background-color: #cc0000 !important;
                  }
                </style>
                """, unsafe_allow_html=True)
            start_button = st.button("Start Webcam")
            stop_button = st.button("Stop Webcam")

            if "run" not in st.session_state:
                st.session_state.run = False
            if start_button:
                st.session_state.run = True
            if stop_button:
                st.session_state.run = False

            result_box = st.empty()
            result_box.markdown("""
                <div style="border: 1px solid white; padding: 8px; border-radius: 5px;
                            background-color: rgba(0,0,0,0.7); color: white; font-size: 25px; max-width: 850px;">
                    [Results will appear here]
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
                <style>
                  #detection-section {
                      transform: scaleY(0.95);
                      transform-origin: top;
                      margin-top: -120px;
                  }
                </style>
                <div id="detection-section"></div>
                """, unsafe_allow_html=True)
            video_placeholder = st.empty()

            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            model, scaler, le = load_classifier()

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                st.error("Unable to access webcam")
            else:
                with mp_pose.Pose(
                        static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as pose:
                    prev_time = time.time()
                    while st.session_state.run and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame_resized = cv2.resize(frame, (640, 480))
                        h, w, _ = frame_resized.shape
                        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_frame)

                        prediction_text = "NO POSE DETECTED 🙂. Do something!"
                        if results.pose_landmarks:
                            features = []
                            for name in ordered_landmarks:
                                idx = landmark_indices[name]
                                lm = results.pose_landmarks.landmark[idx]
                                x_px = int(lm.x * w)
                                y_px = int(lm.y * h)
                                features.extend([x_px, y_px])
                                cv2.circle(frame_resized, (x_px, y_px), 5, (0, 255, 0), -1)
                            features = np.array(features).reshape(1, -1)
                            features_scaled = scaler.transform(features)
                            probs = model.predict_proba(features_scaled)
                            max_prob = np.max(probs)
                            predicted_class = np.argmax(probs)
                            predicted_label = le.inverse_transform([predicted_class])[0]

                            if max_prob < CONF_THRESHOLD:
                                prediction_text = "NO CONFIDENT POSE 🙂. Do some pose!"
                            else:
                                if predicted_label.lower() in ["endurance", "flexibility"]:
                                    prediction_text = ("ENDUARNCE & FLEXIBILITY POSE ACTIVATED - YOU ROCK! 🤩"
                                                       if max_prob >= EMOJI_CONF_THRESHOLD
                                                       else "ENDUARNCE & FLEXIBILITY POSE ACTIVATED ☺️")
                                elif predicted_label.lower() == "weight_loss":
                                    prediction_text = ("WEIGHT LOSS POSE ACTIVATED - YOU ROCK! 🤩"
                                                       if max_prob >= EMOJI_CONF_THRESHOLD
                                                       else "WEIGHT LOSS POSE ACTIVATED ☺️")
                                elif predicted_label.lower() == "stability":
                                    prediction_text = ("STABILITY POSE ACTIVATED - YOU ROCK! 🤩"
                                                       if max_prob >= EMOJI_CONF_THRESHOLD
                                                       else "STABILITY POSE ACTIVATED ☺️")
                                elif predicted_label.lower() == "strength":
                                    prediction_text = ("STRENGTH POSE ACTIVATED - YOU ROCK! 🤩"
                                                       if max_prob >= EMOJI_CONF_THRESHOLD
                                                       else "STRENGTH POSE ACTIVATED ☺️")
                                else:
                                    prediction_text = f"{predicted_label} POSE ACTIVATED ☺️"
                            mp_drawing.draw_landmarks(frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        frame_display = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        display_frame = cv2.resize(frame_display, None, fx=1.5, fy=1.5)
                        video_placeholder.image(display_frame, channels="RGB")
                        result_box.markdown(f"""
                            <div style="border: 1px solid white; padding: 8px; border-radius: 5px;
                                        background-color: rgba(0,0,0,0.7); color: white; font-size: 20px; max-width: 800px;">
                                {prediction_text}
                            </div>
                            """, unsafe_allow_html=True)
                        time.sleep(0.03)
                    cap.release()
    else:
        st.write("No instructions available for this focus area.")
