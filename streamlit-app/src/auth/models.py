# User model and authentication functions
import hashlib
from .db import get_connection

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email=None, is_admin=False):
    """Create a new user in the database."""
    # Implementation here