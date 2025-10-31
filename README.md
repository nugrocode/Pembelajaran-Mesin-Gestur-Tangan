# Real-time Hand-Gesture → Voice (Nembak Gebetan)

## Deskripsi singkat
Aplikasi ini menangkap gesture tangan via webcam, mengklasifikasikannya, lalu mengucapkan frase romantis ketika gesture stabil berubah. Cocok untuk keperluan fun seperti *nembak gebetan* — gunakan dengan persetujuan target.

## Struktur file
- `data_collection.py` — merekam dataset landmark ke CSV.
- `train.py` — melatih RandomForest dan menyimpan `model.pkl`.
- `run_inference.py` — inference real-time + TTS non-blocking.
- `gestures.json` — mapping gesture -> frase (editable).
- `requirements.txt` — daftar dependensi.

## Instalasi
1. Pastikan Python 3.10+ terpasang.
2. Buat virtualenv (disarankan) dan aktifkan.
3. Install dependensi:\n```\npip install -r requirements.txt\n```\n4. Jika pyttsx3 memerlukan driver tertentu (sapi5 di Windows, nsss di macOS), ikuti petunjuk paket.

## Pengumpulan dataset
Contoh perintah:\n```\npython data_collection.py --output dataset.csv --label_name thumbs_up --samples 500\n```\nInstruksi:\n- Tekan angka untuk ganti label, `s` untuk mulai/stop, `q` untuk keluar.\n- Minimal 200–500 frame per label direkomendasikan.\n- Pastikan pencahayaan dan latar konsisten.\n\n## Training\nContoh:\n```\npython train.py --input dataset.csv --model model.pkl\n```\n- Script akan membagi data 80/20, melatih RandomForest (default 200 trees), menyimpan `model.pkl`, dan menyimpan `confusion.png`.\n\n## Run Inference\nContoh:\n```\npython run_inference.py --model model.pkl\n```\nKontrol:\n- `m` mute/unmute TTS\n- `q` atau ESC keluar\n\n## Tuning\n- `n_estimators` di `train.py`: default 200. Lebih banyak akan stabil tapi lebih lambat.\n- `--smoothing` & `--threshold` di `run_inference.py`: default 15 window, threshold 10 votes.\n\n## Troubleshooting\n- Jika model.pkl tidak bisa dimuat — pastikan path benar dan model dibuat dengan joblib.\n- Jika MediaPipe tidak mendeteksi tangan — cek kamera, pencahayaan, dan min_detection_confidence.\n- Jika TTS tidak bunyi — periksa pyttsx3 backend sesuai OS.\n\n## Etika\nGunakan hanya dengan persetujuan pihak yang dituju. Jangan merekam atau menyebarkan data tanpa izin.\n
