# TREPS Fitness Detection

TREPS adalah sistem evaluasi gerakan `push-up` dan `squat` berbasis `MediaPipe Pose` yang memakai landmark tubuh, rule engine, dan finite state machine (FSM) untuk:

- mendeteksi fase gerakan,
- menghitung repetisi,
- memberi feedback teknik dasar,
- menjalankan evaluasi realtime di browser,
- dan mengukur performa arsitektur lewat evaluator offline.

Project ini **bukan custom trained model**. Backbone pose berasal dari `MediaPipe Pose`, sedangkan keputusan push-up/squat dibuat oleh pipeline heuristik dan FSM yang ditulis di project.

## 1. Tujuan Sistem

- Menjalankan evaluasi gerakan push-up dan squat dari kamera atau video.
- Menghitung repetisi pada siklus gerakan penuh, bukan hanya saat menyentuh posisi bawah.
- Memberi detail kesalahan per repetisi.
- Menyediakan evaluator offline yang bisa menghasilkan metrik seperti accuracy, precision, recall, F1, confusion matrix, rekap per video, dan ringkasan error.

## 2. Arsitektur Sistem

### 2.1 Komponen utama

```text
run.py
  -> app/__init__.py
  -> app/routes.py
  -> app/templates/*
  -> app/static/js/exercise-runtime.js
  -> core/engine.py
  -> core/pose_utils.py
  -> config.py
```

### 2.2 Arsitektur aktif

- **Live camera**:
  browser membuka kamera -> `PoseLandmarker` berjalan di browser -> landmark diproses oleh runtime JS yang meniru `core/engine.py` -> hasil, feedback, overlay, history, dan detail repetisi ditampilkan di UI.

- **Upload video browser**:
  browser memuat video -> video dianalisis frame demi frame di browser dengan seek bertahap -> overlay dan progres ditampilkan -> hasil rep dan detail kesalahan disimpan ke history.

- **Upload fallback backend batch-only**:
  dipakai hanya jika browser tidak kuat / format tidak stabil -> video dikirim ke Flask -> backend memanggil `core.video.run_video_session()` -> mengembalikan ringkasan sesi dan detail per rep.

- **Offline evaluation**:
  script evaluasi membaca manifest dataset -> `core.video.evaluate_labeled_videos()` menjalankan pipeline pada seluruh video berlabel -> artifact evaluasi ditulis ke `artifacts/evaluations/`.

## 3. Model / Metode yang Dipakai

### 3.1 Pose backbone

- `MediaPipe Pose Landmarker`
- Browser runtime menggunakan model `.task` dari MediaPipe Tasks Vision
- Offline/backend runtime menggunakan `mediapipe.solutions.pose`

### 3.2 Decision engine

Pipeline inti ada di `core/engine.py`:

1. pose extraction
2. side selection kiri/kanan berbasis visibility
3. quality gate:
   - no pose
   - low visibility
   - invalid side view
   - body terlalu jauh dari kamera
4. smoothing temporal
5. ready-pose gate
6. FSM posisi `up / mid / down`
7. rep window aggregation
8. evaluasi form per rep

### 3.3 Bukan custom trained model

Project ini **tidak memiliki checkpoint model push-up/squat terlatih sendiri** seperti `.pt`, `.onnx`, `.pkl`, atau model sequence custom. Yang ada adalah:

- pose backbone dari MediaPipe
- rule engine/FSM di `core/engine.py`
- evaluator dan artifact di `core/video.py` dan `core/evaluator.py`

## 4. Struktur Folder Singkat

```text
fitness-detection2/
  app/
    routes.py                  # Flask route, upload fallback batch-only
    templates/
      home.html                # home dengan 3 menu utama
      tutorial.html            # tutorial / placeholder video
      exercise.html            # UI evaluasi push-up / squat
    static/js/
      exercise-runtime.js      # runtime browser, overlay, upload, history
  core/
    engine.py                  # source of truth rule engine/FSM Python
    pose_utils.py              # side selection, side-view gate, torso ratio
    video.py                   # evaluator video + artifact
    evaluator.py               # metrics, confusion matrix, JSON/plot
  scripts/
    check.py                   # preflight runtime
    pushup_counter.py          # run push-up tanpa website
    squat_counter.py           # run squat tanpa website
    evaluate_pushup.py         # evaluasi push-up
    evaluate_squat.py          # evaluasi squat
    dataset_audit.py           # manifest + issue summary dataset
  dataset/                     # video mentah dan clip berlabel
  artifacts/                   # manifest, issue summary, output evaluasi
  config.py                    # threshold + payload runtime
  run.py                       # entrypoint web app
  requirements.txt            # dependency Python
  DxDiag.txt                  # profil device target untuk preset runtime
```

## 5. Threshold Penting dan Fungsinya

Source of truth ada di `config.py`.

### 5.1 Threshold global

| Threshold | Fungsi |
|---|---|
| `POSE_MIN_DETECTION_CONFIDENCE` | minimum confidence deteksi pose |
| `POSE_MIN_TRACKING_CONFIDENCE` | minimum confidence tracking pose |
| `LANDMARK_VISIBILITY_THRESHOLD` | visibility rata-rata minimum untuk memilih sisi tubuh |
| `LANDMARK_MIN_POINT_VISIBILITY_THRESHOLD` | visibility minimum tiap keypoint penting agar sisi tidak dipilih terlalu agresif |
| `ANGLE_SMOOTHING_WINDOW` | smoothing sudut utama dan fitur form |
| `LATERAL_RATIO_SMOOTHING_WINDOW` | smoothing rasio side-view dan torso ratio |
| `STATE_STABLE_FRAMES` | jumlah frame stabil sebelum transisi state diakui |
| `MIN_REP_WINDOW_FRAMES` | panjang minimum jendela rep agar noise tidak mudah dihitung sebagai rep |
| `FORM_VIOLATION_RATIO_THRESHOLD` | proporsi pelanggaran minimal dalam satu rep untuk menandai error teknik |
| `SIDE_VIEW_MAX_LATERAL_RATIO` | batas view samping; makin besar berarti makin toleran terhadap front angle |
| `MIN_TORSO_SIZE_RATIO` | deteksi tubuh terlalu jauh dari kamera |

Nilai `FORM_VIOLATION_RATIO_THRESHOLD` saat ini dikalibrasi ke `0.25` dari benchmark dataset aktif.

### 5.2 Push-up

| Threshold | Fungsi |
|---|---|
| `PUSHUP_TOP_ANGLE` | area posisi atas |
| `PUSHUP_BOTTOM_ANGLE` | area posisi bawah |
| `PUSHUP_TRACK_START_ANGLE` | sudut mulai mengaktifkan rep window |
| `PUSHUP_ROM_THRESHOLD` | batas minimum depth untuk full ROM |
| `PUSHUP_BODY_ALIGNMENT_THRESHOLD` | batas kelurusan tubuh |
| `PUSHUP_TRAPS_RAISE_RATIO_THRESHOLD` | heuristik bahu/trapezius terlalu naik |
| `PUSHUP_TRAPS_MIN_ELBOW_ANGLE` | guard agar `traps_naik` tidak terlalu mudah aktif pada rep yang sebenarnya penuh |

Error yang didukung:

- `tidak_full_rom`
- `badan_bungkuk`
- `traps_naik`

### 5.3 Squat

| Threshold | Fungsi |
|---|---|
| `SQUAT_TOP_ANGLE` | area posisi berdiri |
| `SQUAT_BOTTOM_ANGLE` | area posisi bawah |
| `SQUAT_TRACK_START_ANGLE` | sudut mulai mengaktifkan rep window |
| `SQUAT_ROM_THRESHOLD` | batas minimum depth |
| `SQUAT_TORSO_LEAN_THRESHOLD` | batas tubuh terlalu condong ke depan |
| `SQUAT_KNEE_FORWARD_RATIO` | batas lutut terlalu maju relatif terhadap kaki |
| `SQUAT_HEEL_LIFT_RATIO_THRESHOLD` | heuristik tumit terangkat / kaki jinjit |

Catatan: `SQUAT_KNEE_FORWARD_RATIO` saat ini dinonaktifkan (`None`) karena pada benchmark dataset aktif rule ini justru menaikkan false alarm pada rep yang sebenarnya benar.

Error yang didukung:

- `tidak_full_rom`
- `badan_bungkuk`
- `lutut_maju`
- `kaki_jinjit`

### 5.4 Asal nilai threshold

Threshold saat ini adalah **heuristik engineering**, bukan hasil training model. Nilai dipilih dari:

- struktur anatomi landmark MediaPipe,
- kebutuhan hysteresis agar counter stabil,
- tuning praktis terhadap dataset video yang ada,
- kompromi antara toleransi noise dan sensitivitas error teknik.

Kalibrasi terbaru yang paling berdampak:

- `PUSHUP_ROM_THRESHOLD` diturunkan ke `90`
- `PUSHUP_TRAPS_RAISE_RATIO_THRESHOLD` dinaikkan ke `2.5`
- `PUSHUP_TRAPS_MIN_ELBOW_ANGLE` ditetapkan ke `70`
- `FORM_VIOLATION_RATIO_THRESHOLD` diturunkan ke `0.25`
- `SQUAT_HEEL_LIFT_RATIO_THRESHOLD` dinaikkan ke `10.0`
- `SQUAT_KNEE_FORWARD_RATIO` dinonaktifkan sementara

## 6. Alur Sistem dari Input Sampai Hasil

### 6.1 Live camera

1. User membuka halaman `Squat` atau `Push-up`
2. Browser memuat runtime config dari `config.py`
3. User mengaktifkan kamera
4. `PoseLandmarker.detectForVideo()` berjalan di browser
5. Landmark diproses oleh `ClientExerciseSession` di `app/static/js/exercise-runtime.js`
6. UI menampilkan:
   - rep count
   - stage
   - feedback
   - overlay pose
   - detail repetisi aktif
   - history sesi

### 6.2 Upload video browser

1. User memilih video
2. Browser validasi format, ukuran, dan metadata dasar
3. Video diproses frame demi frame dengan seek bertahap
4. Overlay dan progress analisis ditampilkan
5. Hasil rep dan detail error disimpan ke history lokal

### 6.3 Upload fallback backend batch-only

Dipakai jika:

- format tidak stabil untuk browser,
- file terlalu berat untuk browser tetapi masih aman untuk server,
- decode browser gagal,
- runtime browser gagal memproses video tertentu.

Alurnya:

1. browser mengirim file ke `/api/process_uploaded_video`
2. Flask menyimpan file sementara
3. `core.video.run_video_session()` memproses video penuh
4. ringkasan sesi + detail rep dikirim kembali ke UI

Fallback ini **tidak dipakai untuk live camera**.

## 7. Detail Error Per Repetisi

Setiap rep yang selesai dievaluasi menyimpan:

- nomor rep
- status `benar` / `salah`
- `detail` teks
- `error_codes`
- `error_labels`
- `primary_error_code`
- metrik rep yang relevan

Contoh:

- Push-up:
  - Rep 1: benar
  - Rep 2: salah — Badan tidak lurus
  - Rep 3: salah — Tidak full ROM
- Squat:
  - Rep 1: benar
  - Rep 2: salah — Lutut terlalu maju
  - Rep 3: salah — Badan terlalu condong ke depan | Depth kurang / tidak full ROM

## 8. Evaluasi Offline dan Metrik

### 8.1 Script evaluasi utama

- `scripts/evaluate_pushup.py`
- `scripts/evaluate_squat.py`

Keduanya membaca `artifacts/dataset_manifest.csv` dan memanggil `core.video.evaluate_labeled_videos()`.

### 8.2 Metrik yang tersedia

`core/evaluator.py` dan `core/video.py` sekarang menulis:

- accuracy
- precision
- recall
- F1
- confusion matrix
- per-label metrics (`benar`, `salah`)
- runtime metrics:
  - mean FPS
  - no pose rate
  - low visibility rate
  - invalid view rate
  - too far rate
- count metrics jika `expected_reps` tersedia:
  - rep_count_error
  - absolute_rep_count_error
  - false_positive_reps
  - false_negative_reps
  - count_accuracy
- summary per exercise
- summary per label
- summary per error type
- per-video CSV
- per-rep CSV

### 8.3 Keterbatasan evaluator

Beberapa metrik masih `UNKNOWN` jika ground truth belum lengkap:

- phase metrics per frame
- robustness per `camera_view`
- robustness per `occlusion_level`
- generalisasi per `subject_id`
- rep counting benchmark yang kuat jika `expected_reps` belum terisi

Untuk error tag, evaluator sekarang memberi ringkasan **indikatif** berbasis label video yang diwariskan ke seluruh rep. Ini berguna untuk diagnosa awal, tetapi belum setara anotasi per-rep.

## 9. Cara Menjalankan Project

### 9.1 Install dependency

Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 9.2 Preflight

```powershell
.\.venv\Scripts\python.exe scripts/check.py
```

### 9.3 Jalankan website

```powershell
.\.venv\Scripts\python.exe run.py
```

Lalu buka:

```text
http://127.0.0.1:5000
```

## 10. Jalur Run Tanpa Website

### Push-up tanpa website

```powershell
.\.venv\Scripts\python.exe scripts/pushup_counter.py --camera 0 --target-reps 15
```

### Squat tanpa website

```powershell
.\.venv\Scripts\python.exe scripts/squat_counter.py --camera 0 --target-reps 14
```

Kedua script ini membuka webcam langsung dengan overlay OpenCV dan memakai `core/engine.py` yang sama dengan arsitektur utama.

## 11. Command Evaluasi

### Evaluasi push-up

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_pushup.py
```

Smoke run cepat:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_pushup.py --limit 2 --max-reps 1 --run-name smoke_pushup
```

### Evaluasi squat

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_squat.py
```

Smoke run cepat:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_squat.py --limit 2 --max-reps 1 --run-name smoke_squat
```

### Dataset audit / refresh manifest

```powershell
.\.venv\Scripts\python.exe scripts/dataset_audit.py
```

## 12. Artifact Evaluasi

Output utama ditulis ke:

```text
artifacts/evaluations/<exercise>/<timestamp>_<run_name>/
```

Isi penting:

- `summary.json`
- `per_video_results.csv`
- `per_rep_results.csv`
- `evaluated_manifest_snapshot.csv`
- `*_metrics.json`
- `*_confusion_matrix.png` jika `matplotlib` tersedia

## 13. Keterbatasan Sistem

- Sistem masih bergantung pada kualitas landmark MediaPipe.
- Side view tetap jauh lebih stabil dibanding front / oblique view.
- Upload browser diproses frame demi frame, jadi progress video bukan playback realtime penuh.
- Fallback backend upload hanya batch/offline, bukan live stream.
- Belum ada custom trained model khusus push-up/squat.
- Evaluasi error per rep masih dibatasi oleh ground truth dataset yang dominan level video, bukan level rep atau phase.

## 14. Arah Pengembangan Selanjutnya

- Menambahkan ground truth `expected_reps`, `subject_id`, `camera_view`, dan anotasi phase per frame.
- Menambah benchmark error tag yang benar-benar per rep.
- Jika dataset sudah siap, mengembangkan classifier kecil di atas window landmark untuk:
  - phase classification
  - rep event detection
  - rep quality classification

## 15. Ringkasannya

- Runtime live camera: client-side
- Upload video: client-side dengan fallback backend batch-only
- Logic push-up dan squat: merged di core
- Model custom: belum ada
- Source of truth: `config.py`, `core/engine.py`, `core/video.py`
