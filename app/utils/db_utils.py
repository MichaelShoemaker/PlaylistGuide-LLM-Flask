import os
import psycopg2
from datetime import datetime
    
postgres_user = os.getenv("POSTGRES_USER", "db_user")
postgres_password = os.getenv("POSTGRES_PASSWORD", "admin")
postgres_db = os.getenv("POSTGRES_DB", "feedback")
postgres_host = os.getenv("POSTGRES_HOST", "db")
postgres_port = os.getenv("POSTGRES_PORT", "5432")

 # Initialize PostgreSQL database with required table
def init_db():
    conn = psycopg2.connect(
        dbname=postgres_db,
        user=postgres_user,
        password=postgres_password,
        host=postgres_host,
        port=postgres_port
    )
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback VARCHAR(10) CHECK (feedback IN ('positive', 'negative')) NOT NULL,
            comments TEXT,
            title TEXT,
            link TEXT NOT NULL,
            date_time TIMESTAMP NOT NULL,
            UNIQUE (question, link) -- Ensure unique feedback per question and link
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bad_responses (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            error TEXT,
            date_time TIMESTAMP NOT NULL
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()

def db_conn():
    conn = psycopg2.connect(
        dbname=postgres_db,
        user=postgres_user,
        password=postgres_password,
        host=postgres_host,
        port=postgres_port
    )
    return conn

def insert_feedback(question, response_summary, feedback, comments, response_title, response_link):
            return """
                INSERT INTO feedback (question, answer, feedback, comments, title, link, date_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (question, link)
                DO UPDATE SET 
                    feedback = EXCLUDED.feedback,
                    comments = EXCLUDED.comments,
                    title = EXCLUDED.title,
                    answer = EXCLUDED.answer,
                    date_time = EXCLUDED.date_time;
            """, (
                question,
                response_summary,
                feedback,
                comments,
                response_title,
                response_link,
                datetime.now()
            )