import mysql.connector
import hashlib
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Get absolute path to project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
dotenv_path = os.path.join(project_root, ".env")

# Load environment variables from .env file with explicit path
print(f"Looking for .env file at: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print("Loaded .env file successfully")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

# Add debug information about loaded environment variables
print(f"DB_HOST: {os.getenv('DB_HOST', 'localhost')}")
print(f"DB_USER: {os.getenv('DB_USER', 'root')}")
print(f"DB_NAME: {os.getenv('DB_NAME', 'candlesage_db')}")
print(f"DB_PASSWORD set: {'Yes' if os.getenv('DB_PASSWORD') else 'No'}")


DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'candlesage_db')
}

def get_db_connection():
    """Create and return a database connection"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

def init_database():
    """Initialize the database and create necessary tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    cursor = conn.cursor()
    
    try:
        # Create users table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100),
            password_hash VARCHAR(256) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Check if default users exist
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_exists = cursor.fetchone()[0]
        
        # Create default admin user if it doesn't exist
        if not admin_exists:
            admin_hash = hashlib.sha256("admin".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                ("admin", "admin@candlesage.com", admin_hash)
            )
        
        # Check if demo user exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'demo'")
        demo_exists = cursor.fetchone()[0]
        
        # Create default demo user if it doesn't exist
        if not demo_exists:
            demo_hash = hashlib.sha256("demo123".encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                ("demo", "demo@candlesage.com", demo_hash)
            )
        
        conn.commit()
        return True
        
    except mysql.connector.Error as err:
        print(f"Database initialization error: {err}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_user(username, password):
    """Verify user credentials against database"""
    conn = get_db_connection()
    if not conn:
        return False, None
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = "SELECT * FROM users WHERE username = %s AND password_hash = %s"
        cursor.execute(query, (username, password_hash))
        
        user = cursor.fetchone()
        return user is not None, user if user else None
        
    except mysql.connector.Error as err:
        print(f"Verification error: {err}")
        return False, None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def register_user(username, email, password):
    """Register a new user in the database"""
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"
    
    cursor = conn.cursor()
    
    try:
        # Check if username already exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
        if cursor.fetchone()[0] > 0:
            return False, "Username already exists"
        
        # Check if email already exists
        if email:
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            if cursor.fetchone()[0] > 0:
                return False, "Email already exists"
        
        # Hash the password and insert the new user
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        query = "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, password_hash))
        
        conn.commit()
        return True, "User registered successfully"
        
    except mysql.connector.Error as err:
        print(f"Registration error: {err}")
        return False, f"Registration error: {str(err)}"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()