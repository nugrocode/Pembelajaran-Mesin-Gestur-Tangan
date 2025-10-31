"""
data_collection.py

Merekam hand landmarks dari webcam menggunakan MediaPipe dan menyimpan setiap frame
sebagai satu baris CSV dengan format:
 x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label,timestamp

Kontrol:
 - Tekan angka 0-9 atau huruf (A-Z) untuk memilih label saat merekam
 - Tekan 's' untuk mulai/stop menyimpan untuk label terpilih
 - Tekan 'q' atau ESC untuk keluar
 - Contoh:
    python data_collection.py --output dataset.csv --label_name A --samples 500
"""

import os
# Supaya MediaPipe tidak otomatis memuat TensorFlow (menghindari error paging/TensorFlow)
os.environ["MEDIAPIPE_DISABLE_TENSORFLOW"] = "1"

import cv2
import mediapipe as mp
import argparse
import csv
from datetime import datetime
from collections import defaultdict

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    """
    Kembalikan list 63 berupa x0,y0,z0,...,x20,y20,z20 relative ke image
    Jika landmark tidak lengkap, kembalikan None.
    """
    if not hand_landmarks or not hand_landmarks.landmark:
        return None
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    if len(coords) != 63:
        return None
    return coords


def ensure_header(path):
    """Buat file CSV dengan header jika belum ada."""
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = []
            for i in range(21):
                header += [f'x{i}', f'y{i}', f'z{i}']
            header += ['label', 'timestamp']
            writer.writerow(header)


def append_row(path, row):
    """Tambahkan satu baris ke CSV (row berupa list)."""
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def valid_label_char(keycode):
    """Return karakter label dari keycode jika valid (0-9 atau A-Z atau a-z), else None."""
    # digits
    if ord('0') <= keycode <= ord('9'):
        return chr(keycode)
    # uppercase letters
    if ord('A') <= keycode <= ord('Z'):
        return chr(keycode)
    # lowercase letters -> convert to uppercase for consistency
    if ord('a') <= keycode <= ord('z'):
        return chr(keycode).upper()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='dataset.csv', help='CSV output file')
    parser.add_argument('--label_name', '-l', default='A', help='Nama label default')
    parser.add_argument('--samples', '-n', type=int, default=500, help='Jumlah sampel target per label (opsional)')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Device camera index (default 0)')
    args = parser.parse_args()

    ensure_header(args.output)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Tidak dapat membuka webcam. Periksa index kamera dan aplikasi lain yang memakai kamera.')
        return

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

    current_label = args.label_name.upper()
    recording = False
    saved_count_total = 0
    saved_count_per_label = defaultdict(int)

    print("Instruksi: Tekan angka 0-9 atau huruf (A-Z) untuk ganti label, 's' untuk mulai/stop, 'q' untuk keluar")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Tidak bisa membaca frame dari webcam.')
                break

            # 🔹 Balikkan kamera seperti cermin
            frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                # Gambarkan landmark pada frame
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Info overlay
            h, w, _ = frame.shape
            cv2.putText(frame, f'Label: {current_label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Recording: {recording} Total:{saved_count_total}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if recording else (0, 0, 255), 2)

            # Per label counts (tampilkan 3-4 label pertama jika ada)
            y_off = 100
            for i, (lbl, cnt) in enumerate(sorted(saved_count_per_label.items())):
                if i >= 6:
                    break
                cv2.putText(frame, f'{lbl}: {cnt}', (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_off += 25

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print('Keluar oleh user.')
                break
            elif key == ord('s'):
                recording = not recording
                print('Recording:', recording, 'Label sekarang:', current_label)
            else:
                lblch = valid_label_char(key)
                if lblch is not None:
                    current_label = lblch
                    print('Ganti label ke', current_label)

            # Jika recording, simpan frame bila ada hand landmarks lengkap
            if recording and results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]
                coords = extract_landmarks(handLms)
                if coords is not None:
                    timestamp = datetime.utcnow().isoformat()
                    append_row(args.output, coords + [current_label, timestamp])
                    saved_count_total += 1
                    saved_count_per_label[current_label] += 1

                    # Print tiap 50 sampel untuk feedback
                    if saved_count_total % 50 == 0:
                        print(f'Saved total {saved_count_total} samples (per label {dict(saved_count_per_label)})')

                    # Jika tercapai target untuk label ini, hentikan recording otomatis
                    if saved_count_per_label[current_label] >= args.samples:
                        print(f'Target sampel tercapai untuk label {current_label}: {saved_count_per_label[current_label]}')
                        recording = False

    except KeyboardInterrupt:
        print('Dihentikan oleh KeyboardInterrupt.')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print('Selesai. Total sampel tersimpan:', saved_count_total)
        print('Per label:', dict(saved_count_per_label))


if __name__ == '__main__':
    main()
