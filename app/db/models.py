# SQLite migrations (kept simple for v1). Use Alembic later if needed.

MIGRATIONS = """
-- =============================================================================
-- EPISODES TABLE (Episodic Memory) - Day-to-day conversations, DOES decay
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT CHECK(memory_type IN ('episodic','semantic','summary')) NOT NULL,
    importance REAL DEFAULT 0.5,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    temporal_tags TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_user_time ON memories(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(user_id, importance DESC);

-- =============================================================================
-- FACTS TABLE (Semantic Memory) - Personal truths, ONE value per key, NO decay
-- Key-value pairs that get UPDATED, not accumulated
-- =============================================================================
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    last_confirmed_at TEXT DEFAULT (datetime('now')),
    first_learned_at TEXT DEFAULT (datetime('now')),
    provenance_memory_id INTEGER,
    embedding BLOB,
    history TEXT,  -- JSON array of previous values for contradiction tracking
    UNIQUE(user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(user_id, key);

-- =============================================================================
-- MILESTONES TABLE (Life Events) - Timestamped events, NO decay, permanent
-- Deaths, marriages, graduations, major life changes
-- =============================================================================
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_date TEXT,
    description TEXT NOT NULL,
    confidence REAL DEFAULT 0.9,
    created_at TEXT DEFAULT (datetime('now')),
    provenance_memory_id INTEGER,
    embedding BLOB,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_milestones_user ON milestones(user_id);
CREATE INDEX IF NOT EXISTS idx_milestones_type ON milestones(user_id, event_type);
CREATE INDEX IF NOT EXISTS idx_milestones_date ON milestones(user_id, event_date);

CREATE TABLE IF NOT EXISTS adaptation_profiles (
    user_id INTEGER PRIMARY KEY,
    warmth REAL DEFAULT 0.5,
    formality REAL DEFAULT 0.5,
    initiative REAL DEFAULT 0.5,
    check_in_frequency REAL DEFAULT 0.5,
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relationship_metrics (
    user_id INTEGER NOT NULL,
    week INTEGER NOT NULL,
    relationship_depth REAL,
    disclosure_avg REAL,
    emotional_events INTEGER,
    return_rate REAL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, week)
);

CREATE TABLE IF NOT EXISTS temporal_patterns (
    user_id INTEGER NOT NULL,
    pattern_type TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    detected_at TEXT DEFAULT (datetime('now')),
    example_memory_ids TEXT
);

CREATE INDEX IF NOT EXISTS idx_temporal_patterns_lookup ON temporal_patterns(user_id, pattern_type);

-- =============================================================================
-- SCHEMA UPGRADES (for existing databases)
-- These are safe to run multiple times
-- =============================================================================

-- Add missing columns to facts table (for older databases)
-- SQLite doesn't support IF NOT EXISTS for columns, so we use a workaround
"""

# Additional migration to add missing columns
SCHEMA_UPGRADES = """
-- Upgrade facts table if columns are missing
-- These will fail silently if columns already exist
"""

def run_schema_upgrades(conn) -> None:
    """Add missing columns to existing databases."""
    # Check facts table columns
    cursor = conn.execute("PRAGMA table_info(facts)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    upgrades = []
    if "history" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN history TEXT")
    if "first_learned_at" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN first_learned_at TEXT DEFAULT (datetime('now'))")
    if "embedding" not in existing_columns:
        upgrades.append("ALTER TABLE facts ADD COLUMN embedding BLOB")

    for sql in upgrades:
        try:
            conn.execute(sql)
            print(f"[DB] Applied: {sql[:50]}...")
        except Exception as e:
            # Column might already exist
            pass

    if upgrades:
        conn.commit()
