# CandleSage - Market Analysis and AI Assistant

CandleSage is a comprehensive trading platform that combines market analysis tools with an AI-powered assistant. This document provides an overview of the login system and chatbot functionality.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Login System](#login-system)
3. [AI Assistant](#ai-assistant)
4. [Running the Application](#running-the-application)
5. [Configuration](#configuration)

## Getting Started

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/CandleSage.git
cd CandleSage/streamlit-app
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install Groq package for AI assistant:
```bash
pip install groq
```
 
## Database Setup and Initialization

The bot uses MySQL for user authentication and data storage. The database is automatically initialized when the application starts.

 

### Environment Variables
Create a `.env` file in the project root with the following variables:
```
DB_HOST=localhost
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=candlesage_db
```

### Database Initialization
The database and tables are automatically created when the application starts. The initialization process:
1. Creates the database if it doesn't exist
2. Creates the users table with necessary fields
3. Sets up default admin and demo users:
   - Admin user: username="admin", password="admin"
   - Demo user: username="demo", password="demo123"

### Security Note
- The default admin and demo users are created for testing purposes
- In a production environment, you should change these default credentials
- Passwords are stored as SHA-256 hashes in the database

## Database Schema

### Users Table
```sql
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    password_hash VARCHAR(256) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify MySQL server is running
   - Check environment variables in `.env` file
   - Ensure correct MySQL credentials

2. **Table Creation Failed**
   - Check MySQL user permissions
   - Verify database exists
   - Check for conflicting table names

3. **User Authentication Failed**
   - Verify password hashing
   - Check user exists in database
   - Ensure correct username/password combination

### Database Maintenance

1. **Backup**
   ```bash
   mysqldump -u [username] -p candlesage_db > backup.sql
   ```

2. **Restore**
   ```bash
   mysql -u [username] -p candlesage_db < backup.sql
   ```

## Development Notes

- The database initialization is handled in `streamlit-app/src/auth/db.py`
- All database operations use parameterized queries for security
- Connection pooling is implemented for better performance
- Error handling and logging are implemented for debugging 

## Login System

### Features
- Secure user authentication
- User registration
- Session management
- Demo credentials for testing

### Login Page
The login page provides two tabs:
1. **Login Tab**
   - Username and password authentication
   - Error handling for invalid credentials
   - Success feedback on login

2. **Register Tab**
   - New user registration
   - Email (optional) and password fields
   - Password confirmation
   - Validation for password strength

### Demo Credentials
For testing purposes, you can use:
- Username: `demo`
- Password: `demo123`

## AI Assistant

### Features
- Powered by Groq's Llama3-8b model
- Market analysis and trading insights
- Technical indicator explanations
- Trading strategy suggestions
- Chat history management

### Capabilities
The AI assistant can help with:
- Understanding chart patterns
- Explaining technical indicators
- Market trend analysis
- Trading strategy discussions
- Risk management advice

### Chat Interface
- Real-time chat with the AI
- Message history display
- Clear chat history option
- Error handling and fallback responses

## Running the Application

1. **Set up environment variables**
   Create a `.env` file in the `streamlit-app` directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

   Or use Streamlit secrets by creating `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

2. **Start the application**
   ```bash
   streamlit run src/app.py
   ```

3. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Log in using your credentials or demo account
   - Navigate between Market Analysis and AI Assistant tabs

## Configuration


### AI Assistant Configuration
- Model: Llama3-8b-8192
- Temperature: 0.7
- Max tokens: 1024
- Context window: Last 4 messages

### Customization Options
- System prompt can be modified in `chatbot/responses.py`
- Chat history length can be adjusted
- Fallback responses can be customized

## Troubleshooting

### Common Issues
1. **API Key Not Found**
   - Ensure GROQ_API_KEY is set in environment variables or Streamlit secrets
   - Check for typos in the API key

2. **Database Connection Issues**
   - Verify MySQL server is running
   - Check database credentials
   - Ensure proper database initialization

3. **Chat Not Responding**
   - Check internet connection
   - Verify Groq API key is valid
   - Clear chat history and try again

 