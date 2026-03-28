import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "smartvoice.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS gesture_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture   TEXT NOT NULL,
            sentence  TEXT NOT NULL,
            language  TEXT DEFAULT 'en',
            spoken    INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS saved_phrases (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            label    TEXT NOT NULL,
            sentence TEXT NOT NULL,
            category TEXT DEFAULT 'custom',
            language TEXT DEFAULT 'en'
        );

        INSERT OR IGNORE INTO saved_phrases (id, label, sentence, category) VALUES
          (1,  'Hello',       'Hello! How are you today?',               'social'),
          (2,  'Help',        'Please help me. I need assistance.',       'urgent'),
          (3,  'Water',       'I am thirsty. Can I have some water?',     'needs'),
          (4,  'Food',        'I am hungry. I need food please.',         'needs'),
          (5,  'Pain',        'I am in pain. Please help me.',            'urgent'),
          (6,  'Bathroom',    'I need to use the bathroom.',              'needs'),
          (7,  'Yes',         'Yes, I agree. That is correct.',           'response'),
          (8,  'No',          'No, I do not agree with that.',            'response'),
          (9,  'Thank you',   'Thank you very much. I appreciate it.',    'social'),
          (10, 'Stop',        'Please stop. I need a moment.',            'urgent'),
          (11, 'Doctor',      'Please call a doctor for me.',             'urgent'),
          (12, 'Tired',       'I am feeling very tired. I need to rest.', 'feelings');
    """)
    conn.commit()
    conn.close()
    print("[DB] Initialised smartvoice.db")
