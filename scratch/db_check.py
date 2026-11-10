import sqlite3, pathlib, json

db_path = pathlib.Path('storage/incidents.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Sessions
c.execute('SELECT session_id, start_ts, end_ts FROM analysis_sessions')
sessions = c.fetchall()
print('Sessions count:', len(sessions))
print('Sessions sample:', sessions[:5])

# Incidents
c.execute('SELECT id, session_id, track_id, final_risk_score, risk_level, evidence_image_path FROM incidents')
incidents = c.fetchall()
print('Incidents count:', len(incidents))
print('Incidents sample:', incidents[:5])

# Evidence rows
c.execute('SELECT id, incident_id, path FROM evidence')
print('Evidence rows sample:', c.fetchall()[:5])

conn.close()
