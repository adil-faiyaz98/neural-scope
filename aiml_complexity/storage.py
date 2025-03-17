# aiml_complexity/storage.py

import json
import os
from typing import Dict, Any, Optional

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

class AnalysisReportStorage:
    """
    Stores analysis results to JSON or DB.
    """

    def __init__(self, db_type: str = "json", db_path: Optional[str] = None):
        """
        :param db_type: 'json' or 'sqlite' or 'none'
        :param db_path: path to DB or JSON file, etc.
        """
        self.db_type = db_type
        self.db_path = db_path if db_path else "analysis_report.json"
        # if sqlite, handle connection, etc.

        if db_type == "sqlite":
            if not SQLITE_AVAILABLE:
                raise RuntimeError("sqlite3 not available in this environment.")
            self.conn = sqlite3.connect(self.db_path)
            self._init_sqlite()

    def _init_sqlite(self):
        sql = """CREATE TABLE IF NOT EXISTS analysis_reports(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                 );
              """
        self.conn.execute(sql)
        self.conn.commit()

    def save_report(self, report: Dict[str, Any]):
        """
        Save a single analysis report. If JSON, append to file. If sqlite, insert a new row.
        """
        if self.db_type == "json":
            data = []
            if os.path.exists(self.db_path):
                with open(self.db_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except:
                        data = []
            data.append(report)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif self.db_type == "sqlite":
            json_str = json.dumps(report)
            self.conn.execute("INSERT INTO analysis_reports (report) VALUES (?)", (json_str,))
            self.conn.commit()

    def get_all_reports(self) -> Any:
        """
        Fetch all saved reports.
        """
        if self.db_type == "json":
            if os.path.exists(self.db_path):
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return []
        elif self.db_type == "sqlite":
            rows = self.conn.execute("SELECT report FROM analysis_reports").fetchall()
            return [json.loads(r[0]) for r in rows]
        return []

    def close(self):
        """Close DB connection if any."""
        if self.db_type == "sqlite":
            self.conn.close()
