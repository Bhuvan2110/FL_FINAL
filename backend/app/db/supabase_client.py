"""
Supabase client — initialized from env settings.
Uses the service-role key for server-side operations.
If Supabase URL is offline or invalid, falls back to a local SQLite database and local folder storage.
"""
import os
import socket
import json
import uuid
import sqlite3
import threading
from urllib.parse import urlparse
from functools import lru_cache
from supabase import create_client, Client
from app.core.config import get_settings

# --- SQLite Mock Database Setup ---

def init_db(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            uploaded_by TEXT,
            filename TEXT,
            storage_path TEXT,
            cols TEXT, -- JSON
            row_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            dataset_id TEXT,
            algorithm TEXT,
            status TEXT,
            hyperparams TEXT, -- JSON
            celery_task_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            round_num INTEGER,
            loss REAL,
            accuracy REAL,
            val_loss REAL,
            val_accuracy REAL,
            num_clients INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS privacy_budget (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            round_num INTEGER,
            epsilon REAL,
            delta REAL,
            noise_multiplier REAL,
            clip_norm REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            weights_path TEXT,
            version INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            accuracy REAL,
            f1 REAL,
            auc REAL,
            precision_score REAL,
            recall REAL,
            confusion_matrix TEXT, -- JSON
            roc_curve TEXT, -- JSON
            feature_importance TEXT, -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            model_id TEXT,
            input_hash TEXT,
            output INTEGER,
            confidence REAL,
            batch_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT,
            resource TEXT,
            ip TEXT,
            detail TEXT, -- JSON
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

class ExecuteResult:
    def __init__(self, data):
        self.data = data

class LocalMockQueryBuilder:
    def __init__(self, table_name, db_path):
        self.table_name = table_name
        self.db_path = db_path
        self.conditions = []
        self.order_by = None
        self.limit_val = None
        self.is_single = False
        self.operation = "select"
        self.data_to_save = None

    def select(self, fields="*"):
        self.operation = "select"
        return self

    def insert(self, data):
        self.operation = "insert"
        self.data_to_save = data
        return self

    def update(self, data):
        self.operation = "update"
        self.data_to_save = data
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, column, value):
        self.conditions.append((column, value))
        return self

    def order(self, column, desc=False):
        self.order_by = (column, "DESC" if desc else "ASC")
        return self

    def limit(self, n):
        self.limit_val = n
        return self

    def single(self):
        self.is_single = True
        return self

    def execute(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        json_cols = {"cols", "hyperparams", "confusion_matrix", "roc_curve", "feature_importance", "detail"}
        
        try:
            if self.operation == "insert":
                if isinstance(self.data_to_save, dict):
                    records = [self.data_to_save]
                else:
                    records = self.data_to_save
                
                inserted_rows = []
                for rec in records:
                    rec = dict(rec)
                    if "id" not in rec or not rec["id"]:
                        rec["id"] = str(uuid.uuid4())
                    
                    for k, v in rec.items():
                        if k in json_cols and not isinstance(v, (str, type(None))):
                            rec[k] = json.dumps(v)
                    
                    cols = list(rec.keys())
                    vals = list(rec.values())
                    placeholders = ", ".join(["?"] * len(cols))
                    sql = f"INSERT INTO {self.table_name} ({', '.join(cols)}) VALUES ({placeholders})"
                    cursor.execute(sql, vals)
                    
                    cursor.execute(f"SELECT * FROM {self.table_name} WHERE id = ?", (rec["id"],))
                    row = dict(cursor.fetchone())
                    for k in json_cols:
                        if k in row and row[k]:
                            try:
                                row[k] = json.loads(row[k])
                            except Exception:
                                pass
                    inserted_rows.append(row)
                
                conn.commit()
                return ExecuteResult(inserted_rows)
                
            elif self.operation == "update":
                rec = dict(self.data_to_save)
                for k, v in rec.items():
                    if k in json_cols and not isinstance(v, (str, type(None))):
                        rec[k] = json.dumps(v)
                
                set_clause = ", ".join([f"{k} = ?" for k in rec.keys()])
                vals = list(rec.values())
                
                where_clause = ""
                where_vals = []
                if self.conditions:
                    where_clause = " WHERE " + " AND ".join([f"{c[0]} = ?" for c in self.conditions])
                    where_vals = [c[1] for c in self.conditions]
                
                sql = f"UPDATE {self.table_name} SET {set_clause}{where_clause}"
                cursor.execute(sql, vals + where_vals)
                conn.commit()
                return ExecuteResult([])
                
            elif self.operation == "delete":
                where_clause = ""
                where_vals = []
                if self.conditions:
                    where_clause = " WHERE " + " AND ".join([f"{c[0]} = ?" for c in self.conditions])
                    where_vals = [c[1] for c in self.conditions]
                
                sql = f"DELETE FROM {self.table_name}{where_clause}"
                cursor.execute(sql, where_vals)
                conn.commit()
                return ExecuteResult([])
                
            elif self.operation == "select":
                where_clause = ""
                where_vals = []
                if self.conditions:
                    where_clause = " WHERE " + " AND ".join([f"{c[0]} = ?" for c in self.conditions])
                    where_vals = [c[1] for c in self.conditions]
                
                sql = f"SELECT * FROM {self.table_name}{where_clause}"
                if self.order_by:
                    sql += f" ORDER BY {self.order_by[0]} {self.order_by[1]}"
                if self.limit_val:
                    sql += f" LIMIT {self.limit_val}"
                    
                cursor.execute(sql, where_vals)
                rows = [dict(r) for r in cursor.fetchall()]
                
                for row in rows:
                    for k in json_cols:
                        if k in row and row[k]:
                            try:
                                row[k] = json.loads(row[k])
                            except Exception:
                                pass
                    
                    if self.table_name == "experiments":
                        ds_id = row.get("dataset_id")
                        if ds_id:
                            cursor.execute("SELECT filename FROM datasets WHERE id = ?", (ds_id,))
                            ds_row = cursor.fetchone()
                            row["datasets"] = {"filename": ds_row[0]} if ds_row else {"filename": "Unknown Dataset"}
                
                if self.is_single:
                    return ExecuteResult(rows[0] if rows else None)
                return ExecuteResult(rows)
        finally:
            conn.close()

class LocalMockStorageBucket:
    def __init__(self, bucket_name, storage_dir):
        self.bucket_name = bucket_name
        self.storage_dir = os.path.join(storage_dir, bucket_name)
        os.makedirs(self.storage_dir, exist_ok=True)

    def upload(self, path, content, options=None):
        full_path = os.path.join(self.storage_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if isinstance(content, str):
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        elif isinstance(content, bytes):
            with open(full_path, "wb") as f:
                f.write(content)
        else:
            # Handle other stream types
            try:
                data = content.read()
                with open(full_path, "wb") as f:
                    f.write(data)
            except Exception:
                with open(full_path, "wb") as f:
                    f.write(content)
        return {"path": path}

    def download(self, path):
        full_path = os.path.join(self.storage_dir, path)
        if not os.path.exists(full_path):
            raise Exception("File not found")
        with open(full_path, "rb") as f:
            return f.read()

    def remove(self, paths):
        for path in paths:
            full_path = os.path.join(self.storage_dir, path)
            if os.path.exists(full_path):
                os.remove(full_path)
        return {"message": "Success"}

class LocalMockStorage:
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir

    def from_(self, bucket_name):
        return LocalMockStorageBucket(bucket_name, self.storage_dir)

class LocalMockSupabaseClient:
    def __init__(self, db_path, storage_dir):
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.storage = LocalMockStorage(storage_dir)
        
        conn = sqlite3.connect(self.db_path)
        init_db(conn)
        conn.close()

    def table(self, table_name):
        return LocalMockQueryBuilder(table_name, self.db_path)

# --- DNS Reachability check ---

def is_supabase_reachable(url: str) -> bool:
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        # Immediate bypass for known invalid/deleted host to prevent DNS hang
        if "gxrnjyurzoegbhljkyda" in host:
            return False
            
        # Fast thread-based DNS check with a 1.2-second timeout
        result = [False]
        def worker():
            try:
                socket.gethostbyname(host)
                result[0] = True
            except Exception:
                pass
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        t.join(1.2)
        return result[0]
    except Exception:
        return False

# --- Export Client Getters ---

@lru_cache()
def get_supabase() -> Client:
    s = get_settings()
    if not is_supabase_reachable(s.supabase_url):
        # Fall back to local SQLite and local storage
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "local_db.sqlite")
        storage_dir = os.path.join(base_dir, "local_storage")
        print(f"⚠️ Supabase URL {s.supabase_url} is unreachable. Falling back to local database: {db_path}")
        return LocalMockSupabaseClient(db_path, storage_dir)
        
    return create_client(s.supabase_url, s.supabase_service_role_key)


@lru_cache()
def get_supabase_anon() -> Client:
    """Anon client for auth operations."""
    s = get_settings()
    if not is_supabase_reachable(s.supabase_url):
        # Fall back to local SQLite and local storage
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "local_db.sqlite")
        storage_dir = os.path.join(base_dir, "local_storage")
        return LocalMockSupabaseClient(db_path, storage_dir)
        
    return create_client(s.supabase_url, s.supabase_anon_key)
