PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_code TEXT UNIQUE,
    first_name TEXT NOT NULL,
    last_name TEXT DEFAULT '',
    full_name TEXT NOT NULL UNIQUE,
    face_image_dir TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    attendance_date TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'ai-confirmed',
    confidence REAL,
    track_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id),
    UNIQUE(student_id, attendance_date)
);

CREATE TABLE IF NOT EXISTS behavior_status_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    track_id INTEGER,
    state TEXT NOT NULL,
    is_drowsy INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    event_time TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'ai',
    created_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS face_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    original_path TEXT NOT NULL UNIQUE,
    cache_path TEXT,
    file_hash TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(attendance_date);
CREATE INDEX IF NOT EXISTS idx_behavior_event_time ON behavior_status_logs(event_time);
CREATE INDEX IF NOT EXISTS idx_behavior_state ON behavior_status_logs(state);
CREATE INDEX IF NOT EXISTS idx_face_images_student_id ON face_images(student_id);
