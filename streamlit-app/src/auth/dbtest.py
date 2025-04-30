import mysql.connector
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv

# Clear any existing environment variables
if 'DB_HOST' in os.environ: del os.environ['DB_HOST']
if 'DB_USER' in os.environ: del os.environ['DB_USER'] 
if 'DB_PASSWORD' in os.environ: del os.environ['DB_PASSWORD']
if 'DB_NAME' in os.environ: del os.environ['DB_NAME']

# Get absolute path to project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
dotenv_path = os.path.join(project_root, ".env")

# Load environment variables from .env file with explicit path
print(f"Looking for .env file at: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("Loaded .env file successfully")
    
    # Print environment variables for debugging
    print(f"DB_HOST: {os.getenv('DB_HOST')}")
    print(f"DB_USER: {os.getenv('DB_USER')}")
    print(f"DB_NAME: {os.getenv('DB_NAME')}")
    print(f"DB_PASSWORD set: {'Yes' if os.getenv('DB_PASSWORD') else 'No'}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'candlesage_db')
}

print("Attempting connection with following configuration:")
for key, value in DB_CONFIG.items():
    if key != 'password':
        print(f"{key}: {value}")
    else:
        print(f"password: {'[SET]' if value else '[EMPTY]'}")

def get_db_connection():
    """Create and return a database connection"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None
        
def test_connection():
    conn = get_db_connection()
    if conn and conn.is_connected():
        print("Connection successful!")
        cursor = conn.cursor()
        cursor.execute("SELECT USER(), CURRENT_USER(), DATABASE()")
        result = cursor.fetchone()
        print(f"Connected as: {result[0]}")
        print(f"Authenticated as: {result[1]}")
        print(f"Using database: {result[2]}")
        cursor.close()
        conn.close()
        return True
    else:
        print("Connection failed!")
        return False

# Add this line at the end of db.py for testing
if __name__ == "__main__":
    test_connection()