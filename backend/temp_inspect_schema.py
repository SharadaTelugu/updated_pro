import sqlite3, sys, os

db_path = r'C:\\Users\\naroj\\OneDrive\\Desktop\\prjct2\\storage\\incidents.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
rows = cur.fetchall()
for row in rows:
    print(row[0])
conn.close()
