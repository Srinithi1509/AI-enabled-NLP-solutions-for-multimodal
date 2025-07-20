import sqlite3

def create_database():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Create the users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
