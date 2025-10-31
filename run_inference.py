import cv2
import mediapipe as mp
import joblib
import numpy as np
import argparse
import time
from collections import deque, Counter
import tempfile
import pygame
import json
import os
import threading  # 👈 penting
from gtts import gTTS

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# === PREPROCESSING SAMA SEPERTI TRAINING ===
def preprocess_landmarks(coords):
    arr = np.array(coords).reshape((21, 3))
    origin = arr[0].copy()
    arr[:, :2] -= origin[:2]
    dists = np.linalg.norm(arr[:, :2], axis=1)
    maxd = dists.max() if dists.max() != 0 else 1.0
    arr[:, :2] /= maxd
    return arr.flatten().reshape(1, -1)


# === TTS dengan Google Translate (gTTS, bahasa Jepang) ===
def speak_japanese(text):
    """Jalankan TTS di thread supaya tidak memblokir kamera"""
    def _speak():
        print(f"🎙️ Suara Jepang: {text}")
        try:
            tts = gTTS(text=text, lang='ja')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_path = fp.name

            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

            # Tidak blocking, hanya menunggu sampai selesai
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.quit()
            os.remove(audio_path)
        except Exception as e:
            print("❌ Error TTS:", e)

    # Jalankan thread TTS
    threading.Thread(target=_speak, daemon=True).start()


# === MEMUAT MAPPING GESTUR KE TEKS (Bahasa Jepang Anime Style) ===
def load_gestures(path='gestures_jp.json'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            gestures = json.load(f)
            print(f"✅ Loaded {len(gestures)} gestures from {path}")
            return gestures
    except Exception as e:
        print("⚠️ Tidak bisa memuat gestures_jp.json:", e)
        return {
            'A': 'やっほー！こんにちは！',
            'B': '元気？今日も頑張ろうね！',
            'C': '今、勉強中なんだ！',
            'D': 'コンピュータービジョン！すごいでしょ？',
            'F': 'うわぁ〜、むずかしいよぉ！',
            'G': 'また失敗しちゃった…ちょっとショック〜。',
            'H': 'でも、これはめっちゃチャレンジだね！'
        }


# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='model.pkl', help='Model joblib file')
    parser.add_argument('--smoothing', type=int, default=15)
    parser.add_argument('--threshold', type=int, default=10)
    args = parser.parse_args()

    model = joblib.load(args.model)
    gestures_map = load_gestures('gestures.json')

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

    pred_window = deque(maxlen=args.smoothing)
    stable_label = None
    last_speak_time = 0
    muted = False

    print('🎥 Tekan M untuk mute/unmute, Q untuk keluar')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        display_label = 'NoHand'

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            coords = [v for lm in handLms.landmark for v in (lm.x, lm.y, lm.z)]

            if len(coords) == 63:
                X = preprocess_landmarks(coords)
                try:
                    pred = model.predict(X)[0]
                except Exception as e:
                    print('Prediction error:', e)
                    pred = None

                pred_window.append(pred)

                if len(pred_window) == pred_window.maxlen:
                    most_common = Counter(pred_window).most_common(1)[0]
                    label, votes = most_common

                    if label and votes >= args.threshold:
                        display_label = label

                        # bicara jika gestur berubah dan sudah lewat 3 detik
                        if label != stable_label and (time.time() - last_speak_time) > 3:
                            stable_label = label
                            last_speak_time = time.time()

                            if not muted:
                                phrase = gestures_map.get(label, "何これ？新しいジェスチャーかな？")
                                speak_japanese(phrase)
                    else:
                        display_label = 'Unstable'
            else:
                display_label = 'Incomplete'
        else:
            pred_window.clear()
            stable_label = None
            display_label = 'NoHand'

        cv2.putText(frame, f'Label: {display_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Muted: {muted}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if not muted else (0, 0, 255), 2)

        cv2.imshow('Inference Anime Voice (Google TTS)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            muted = not muted
            print('Muted:', muted)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
