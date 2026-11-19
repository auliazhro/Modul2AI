
import sqlite3
from pathlib import Path

# Ganti sesuai lokasi database lokal project kamu
DB_FILE = r"modul2ai.db"

db_path = Path(DB_FILE)

if not db_path.exists():
    print("Database tidak ditemukan:", db_path)
    raise SystemExit

# Membuka database dalam mode read-only
uri = db_path.resolve().as_uri() + "?mode=ro"

conn = sqlite3.connect(uri, uri=True)
cursor = conn.cursor()

print("=" * 55)
print("PEMERIKSAAN DATABASE MODUL2AI")
print("=" * 55)

# Menampilkan semua tabel
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""")

tables = [row[0] for row in cursor.fetchall()]

print("\nDAFTAR TABEL:")

for table in tables:
    print("-", table)

# Memeriksa struktur setiap tabel
for table in tables:
    print(f"\nSTRUKTUR TABEL: {table}")

    cursor.execute(f'PRAGMA table_info("{table}")')

    for column in cursor.fetchall():
        print(
            f"Kolom: {column[1]} | "
            f"Tipe: {column[2]} | "
            f"Primary Key: {column[5]}"
        )

    cursor.execute(f'PRAGMA foreign_key_list("{table}")')

    for fk in cursor.fetchall():
        print(
            f"Relasi: {fk[3]} -> "
            f"{fk[2]}.{fk[4]}"
        )

# Menampilkan daftar role jika tersedia
if "roles" in tables:
    print("\nDATA ROLE:")
    cursor.execute('SELECT * FROM "roles" LIMIT 20')

    for row in cursor.fetchall():
        print(row)
else:
    print("\nTabel roles belum ditemukan.")

conn.close()

print("\nPEMERIKSAAN SELESAI")
