# storage.py
import datetime
import json

import psycopg2


class PostgresStorage:
    def __init__(self, host="localhost", port=5432, dbname="analysisdb", user="postgres", password="postgres"):
        self.conn = psycopg2.connect(host=host, port=port, database=dbname, user=user, password=password)
        self.cur = self.conn.cursor()
        self._setup_tables()

    def _setup_tables(self):
        create_table = """
        CREATE TABLE IF NOT EXISTS ml_analysis_results (
            id SERIAL PRIMARY KEY,
            algorithm_name VARCHAR(255),
            theoretical_complexity VARCHAR(255),
            empirical_results JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        self.cur.execute(create_table)
        self.conn.commit()

    def save_analysis(self, algo_name, theoretical_complex, empirical, timestamp=None):
        if not timestamp:
            timestamp = datetime.datetime.now()
        emp_json = json.dumps(empirical)
        insert_q = """
        INSERT INTO ml_analysis_results (algorithm_name, theoretical_complexity, empirical_results, timestamp)
        VALUES (%s, %s, %s, %s);
        """
        self.cur.execute(insert_q, (algo_name, theoretical_complex, emp_json, timestamp))
        self.conn.commit()

    def fetch_all(self, algo_name=None):
        if algo_name:
            self.cur.execute(
                "SELECT id, algorithm_name, theoretical_complexity, empirical_results, timestamp FROM ml_analysis_results WHERE algorithm_name=%s ORDER BY timestamp;",
                (algo_name,))
        else:
            self.cur.execute(
                "SELECT id, algorithm_name, theoretical_complexity, empirical_results, timestamp FROM ml_analysis_results ORDER BY timestamp;")

        rows = self.cur.fetchall()
        results = []
        for r in rows:
            r_id, a_name, t_comp, emp_json, ts = r
            emp = json.loads(emp_json)
            results.append({
                "id": r_id,
                "algorithm_name": a_name,
                "theoretical_complexity": t_comp,
                "empirical_results": emp,
                "timestamp": str(ts)
            })
        return results
