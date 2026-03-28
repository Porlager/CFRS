<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=240&color=0:1D3557,45:2A9D8F,100:E9C46A&text=CFRS&fontSize=56&fontColor=ffffff&desc=Classroom%20Facial%20Recognition%20System&descAlignY=66&animation=fadeIn" alt="CFRS Banner" />
</p>

<p align="center">
  ระบบเช็คชื่อและติดตามพฤติกรรมในห้องเรียนแบบเรียลไทม์<br>
  ครบทั้งกล้อง, Dashboard สด, ลงทะเบียนผ่านเว็บ และรายงานเชิงพฤติกรรม
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/Flask-API%20%26%20Dashboard-111111?style=for-the-badge&logo=flask&logoColor=white" alt="flask">
  <img src="https://img.shields.io/badge/OpenCV-Real--Time-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv">
  <img src="https://img.shields.io/badge/Face%20Recognition-Attendance-0B8F8A?style=for-the-badge" alt="face recognition">
</p>

<p align="center">
  <a href="#highlight">Highlight</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#run-modes">Run Modes</a> •
  <a href="#api-reference">API</a> •
  <a href="#tuning-cookbook">Tuning</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## Highlight

<table>
  <tr>
    <td width="33%">
      <strong>Real-Time Monitoring</strong><br>
      Dashboard อัปเดตสดจากกล้อง พร้อมจำนวนคนและสถานะรายบุคคล
    </td>
    <td width="33%">
      <strong>Smart Attendance</strong><br>
      เช็คชื่ออัตโนมัติด้วย face recognition + tracker ลดการนับซ้ำ
    </td>
    <td width="33%">
      <strong>Behavior Insights</strong><br>
      รองรับสถานะตั้งใจเรียน/ไม่ตั้งใจ/หลับเหม่อ/unknown ครบในระบบเดียว
    </td>
  </tr>
</table>

สิ่งที่โดดเด่นของระบบนี้

- Face recognition แบบ real-time พร้อม dynamic threshold และ margin check
- มี Body Fallback กรณีหน้าไม่ชัดชั่วคราว เพื่อคง continuity ของการติดตาม
- หน้า Register พร้อม webcam capture สำหรับเพิ่มข้อมูลบุคคลใหม่
- Dashboard รองรับทั้ง Desktop และมือถือ ใช้ได้จริงในการพรีเซนต์

---

## System Overview

```mermaid
flowchart LR
    A[Camera Stream] --> B[main.py
    detect + classify + track]
    B --> C[POST /api/result]
    C --> D[dashboard_api.py
    runtime + db write]
    D --> E[Dashboard UI]
    D --> F[Reports API]
    G[Register Page] --> H[/api/register-face]
    H --> I[known_faces]
```

---

## Supported States

| สถานะ | ความหมายเชิงปฏิบัติ |
| --- | --- |
| ตั้งใจเรียน | พฤติกรรมโดยรวมปกติและมีแนวโน้มจดจ่อ |
| ไม่ตั้งใจเรียน | ตรวจพบลักษณะ pose/การหันที่เข้าเกณฑ์ inattentive |
| หลับ/เหม่อ | ตรวจจากดวงตา (EAR) หรือพฤติกรรมง่วง |
| ฟุบหลับ/หันหลัง | มาจาก body fallback เมื่อใบหน้าไม่พร้อมใช้งาน |
| Unknown / ไม่ทราบสถานะ | ยังยืนยันตัวตนหรือสถานะไม่ได้ |

---

## Project Structure

```text
CFRS/
|- main.py                 # กล้องหลัก: detect, classify, track, send payload
|- dashboard_api.py        # Flask API + Dashboard + Register page
|- backend_db.py           # บันทึก attendance และ behavior logs
|- templates/
|  |- dashboard.html
|  |- register.html
|  `- error.html
|- static/
|  |- dashboard.js
|  |- dashboard.css
|  |- register.js
|  `- register.css
|- known_faces/            # รูปที่ใช้เป็นฐานข้อมูลใบหน้า
|- data/                   # ฐานข้อมูล SQLite
|- storage/                # latest_frame.jpg สำหรับหน้า dashboard
|- tests/
|  `- test_dashboard_e2e.py
|- run_dashboard.py        # สคริปต์รัน cross-platform (Linux/macOS/Windows)
|- run_dashboard.ps1       # สคริปต์รัน full stack / dashboard-only (PowerShell)
`- run_e2e_tests.ps1       # สคริปต์รันทดสอบ end-to-end
```

---

## Quick Start

### 1) ติดตั้ง dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) รันระบบแบบครบชุด

```bash
python run_dashboard.py
```

สิ่งที่จะได้ทันที

- Dashboard: http://127.0.0.1:5000
- Register page: http://127.0.0.1:5000/register
- Camera backend ส่ง attendance + behavior เข้า dashboard อัตโนมัติ

---

## Run Modes

| โหมด | คำสั่ง | ใช้เมื่อ |
| --- | --- | --- |
| Full Stack (Cross-platform) | `python run_dashboard.py` | ใช้งานจริงหรือเดโมแบบครบระบบ (Linux/macOS/Windows) |
| Dashboard Only (Cross-platform) | `python run_dashboard.py --dashboard-only` | ทดสอบหน้าเว็บ/รายงานโดยไม่เปิดกล้อง |
| Full Stack (PowerShell) | `./run_dashboard.ps1` | ทางเลือกเดิมสำหรับ Windows PowerShell |
| Dashboard Only (PowerShell) | `./run_dashboard.ps1 -DashboardOnly` | ทางเลือกเดิมสำหรับ Windows PowerShell |
| Direct Flask | `python dashboard_api.py` | รัน API ตรงสำหรับ debug |

> หมายเหตุ (macOS): ถ้า port `5000` ถูกใช้งาน (เช่น AirPlay Receiver) `dashboard_api.py` จะเลื่อนไปใช้พอร์ตถัดไปอัตโนมัติ เช่น `5001`, `5002` และจะแจ้งใน console

### ใช้มือถือเป็น IP Camera

รองรับแล้วผ่านตัวแปร `CFRS_CAMERA_SOURCE` โดยใส่ URL จากแอปมือถือ เช่น IP Webcam

```bash
CFRS_CAMERA_SOURCE="http://192.168.1.105:8080/video" python run_dashboard.py
```

ถ้าใส่แค่ base URL เช่น `http://192.168.1.105:8080` ระบบจะลองเติม `/video` ให้อัตโนมัติ

### เลือกกล้องในคอมแบบ Choice Option

`main.py` รองรับการเลือกกล้องจากรายการที่ตรวจพบได้แล้ว (interactive choice)

```bash
source .venv/bin/activate
CFRS_CAMERA_SOURCE=ask python main.py
```

พฤติกรรมที่รองรับ

- จะแสดงรายการ camera index ที่หาเจอ แล้วให้เลือกเลข
- กด Enter ได้เพื่อใช้ค่า default (ตัวแรก)
- หากไม่ตั้งค่า `CFRS_CAMERA_SOURCE` ระบบจะพยายามเลือกตัวแรกให้อัตโนมัติ

ตัวแปรที่เกี่ยวข้อง

- `CFRS_CAMERA_SOURCE=ask` บังคับให้ถามเลือกกล้อง
- `CFRS_CAMERA_SOURCE=<index>` เลือกกล้องตรง เช่น `0`, `1`
- `CFRS_CAMERA_CHOICE=0` ปิด interactive prompt
- `CFRS_CAMERA_SCAN_MAX=6` จำนวน index สูงสุดที่ใช้สแกนกล้อง

### โหมดติดตามนักศึกษาแม่นขึ้น + ลื่นขึ้น

ระบบตอนนี้รองรับการสืบทอด `student_code` และชื่อจาก track เดิม เมื่อหน้าโดนบัง/หายชั่วคราว แล้วกลับมาได้ต่อเนื่อง

- ถ้าไฟล์ `known_faces` ตั้งชื่อแบบ `รหัสนักศึกษา_ชื่อ.jpg` เช่น `66051281_Apirak.jpg` ระบบจะดึงรหัสนศ.มาใช้ทันที
- payload จะส่ง `student_code` ไป backend เพื่อช่วย mapping ตัวตนและลดการสลับชื่อ
- มี state smoothing เพื่อลดการแกว่งของสถานะพฤติกรรมเฟรมต่อเฟรม
- มี body posture fallback แยก `ฟุบหลับ/นอน` กับ `หันหลัง/ไม่ตั้งใจ` ตามสัดส่วนกรอบตัว
- การส่งผลไป backend เป็น async queue ลดโอกาสแลคจาก network delay

ตัวแปรปรับจูนใหม่

- `CFRS_STATE_SMOOTHING_WINDOW=6` จำนวนเฟรมสำหรับโหวตสถานะ
- `CFRS_BODY_LYING_RATIO=1.08` ยิ่งสูง ยิ่งจับนอนง่ายขึ้น
- `CFRS_BODY_SLOUCH_RATIO=1.45` ยิ่งสูง ยิ่งจัดเป็นหันหลัง/ไม่ตั้งใจง่ายขึ้น
- `CFRS_POST_TIMEOUT_SEC=1.2` timeout ส่ง payload ต่อครั้ง
- `CFRS_POST_QUEUE_SIZE=4` ขนาดคิว payload แบบ async
- `CFRS_STUDENT_DIR_REFRESH_SEC=20` ความถี่รีเฟรช mapping นักศึกษาจากฐานข้อมูล

---

## Register Flow

1. เปิดหน้า http://127.0.0.1:5000/register
2. กรอกเลขนักศึกษาและชื่อ
3. อัปโหลดรูป หรือเปิดกล้องแล้ว Capture
4. กดลงทะเบียน

หมายเหตุ: ถ้า main.py กำลังรันอยู่ ให้รีสตาร์ตกล้อง backend เพื่อ reload known_faces

---

## API Reference

| Method | Endpoint | วัตถุประสงค์ |
| --- | --- | --- |
| GET | / | หน้า Dashboard |
| GET | /register | หน้าลงทะเบียนใบหน้า |
| GET | /health | ตรวจสุขภาพ backend |
| GET | /api/camera/frame | ดึงภาพกล้องล่าสุด |
| POST | /api/result | รับ payload จาก backend กล้อง |
| GET | /api/runtime/status | runtime ล่าสุด (counts + current students) |
| GET | /api/reports/dashboard?days=7 | สรุปภาพรวม dashboard |
| GET | /api/reports/today | รายงานวันนี้ |
| GET | /api/reports/behavior?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD | รายงานพฤติกรรม |
| POST | /api/register-face | ลงทะเบียนใบหน้าใหม่ |

---

## Tuning Cookbook

| เป้าหมาย | ตัวแปรหลัก | แนวทางปรับ |
| --- | --- | --- |
| เพิ่ม FPS | `CFRS_PROCESS_EVERY_N_FRAMES` | เพิ่มค่าเป็น 4-5 บนเครื่องสเปกต่ำ |
| ลดภาระ body detect | `CFRS_BODY_DETECT_EVERY_N_FRAMES` | เพิ่มค่าเป็น 5-6 |
| เร่ง inference | `CFRS_FRAME_RESIZE_SCALE` | ลดสเกล เช่น 0.42 หรือ 0.40 |
| ปรับความไว inattentive | `CFRS_POSE_INATTENTIVE_PENALTY` | ลดค่า = ไวขึ้น, เพิ่มค่า = เข้มขึ้น |
| ปรับความไวหลับ/เหม่อ | `CFRS_EAR_THRESH` | เพิ่มค่า = จัดเป็นง่วงง่ายขึ้น |
| ลดการแกว่งสถานะ | `CFRS_STATE_SMOOTHING_WINDOW` | เพิ่มค่าเพื่อให้สถานะนิ่งขึ้น |
| ปรับเกณฑ์ท่านอน | `CFRS_BODY_LYING_RATIO` | เพิ่มค่าเพื่อจับท่านอนได้ไวขึ้น |
| ปรับเกณฑ์หันหลัง | `CFRS_BODY_SLOUCH_RATIO` | เพิ่มค่าเพื่อตีความ inattentive ง่ายขึ้น |

ค่าตั้งต้นที่ใช้อยู่ในสคริปต์

- `CFRS_PROCESS_EVERY_N_FRAMES=3`
- `CFRS_BODY_DETECT_EVERY_N_FRAMES=4`
- `CFRS_BODY_RESIZE_SCALE=0.4`
- `CFRS_CAMERA_WIDTH=640`, `CFRS_CAMERA_HEIGHT=480`
- `CFRS_FRAME_WRITE_INTERVAL_SEC=0.45`
- `CFRS_STATE_SMOOTHING_WINDOW=6`

---

## Test Commands

```powershell
./run_e2e_tests.ps1
```

หรือ

```bash
python -m pytest
```

---

## Troubleshooting

### รันแล้ว Exit Code 1

ตรวจ dependency หลักก่อน

```powershell
py -3 -c "import flask, requests, cv2, face_recognition"
```

ถ้ายังไม่ครบให้ติดตั้ง

```bash
python -m pip install -r requirements.txt
```

### หน้า dashboard ไม่ขึ้นภาพกล้อง

- ใช้โหมด full stack: `python run_dashboard.py`
- ตรวจว่าไฟล์ `storage/latest_frame.jpg` ถูกอัปเดตต่อเนื่อง

### Port 5000 ถูกใช้งาน (macOS)

- ระบบรองรับ fallback พอร์ตอัตโนมัติแล้ว (จาก `5000` ไปพอร์ตถัดไปที่ว่าง)
- ถ้าต้องการกำหนดเอง: `CFRS_PORT=5001 python dashboard_api.py`
- ถ้าไม่ต้องการ fallback อัตโนมัติ: `CFRS_PORT_AUTO_FALLBACK=0`
- ปรับช่วงสแกน fallback ได้ด้วย `CFRS_PORT_FALLBACK_RANGE` (default: `20`)

### ลงทะเบียนแล้วไม่รู้จักหน้าทันที

- รีสตาร์ต backend กล้องเพื่อโหลดไฟล์ใน `known_faces/` ใหม่

---

## Accuracy Notes

- เหมาะกับงาน real-time monitoring บนเครื่องทั่วไป
- ควรมีรูปต่อคนหลายมุมและหลายแสงเพื่อเพิ่มความเสถียร
- สถานะพฤติกรรมเป็น heuristic-based interpretation ควรใช้ร่วมกับบริบทจริง

---

## License

ใช้งานเพื่อการเรียนรู้ วิจัย และสาธิต ตามนโยบายของผู้ดูแลโปรเจกต์
