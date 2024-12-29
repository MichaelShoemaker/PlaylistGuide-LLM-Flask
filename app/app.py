import os
import json
from flask import Flask, request, jsonify, render_template
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import openai
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
elasticsearch_host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
elasticsearch_port = os.getenv("ELASTICSEARCH_PORT", "9200")
elasticsearch_url = f"http://{elasticsearch_host}:{elasticsearch_port}"
openai_api_key = os.getenv("OPENAI_API_KEY", "your_default_openai_key")
postgres_user = os.getenv("POSTGRES_USER", "db_user")
postgres_password = os.getenv("POSTGRES_PASSWORD", "admin")
postgres_db = os.getenv("POSTGRES_DB", "feedback")
postgres_host = os.getenv("POSTGRES_HOST", "db")
postgres_port = os.getenv("POSTGRES_PORT", "5432")

# Initialize connections
es = Elasticsearch(elasticsearch_url)
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
openai.api_key = openai_api_key

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
            elasticsearch_response JSONB NOT NULL, -- New column for Elasticsearch response
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

# Call init_db function to set up the table
init_db()

app = Flask(__name__)

# Helper functions
def knn_query(question):
    return  {
        "field": "text_vector",
        "query_vector": model.encode(question),
        "k": 5,
        "num_candidates": 10000,
        "boost": 0.5,
    }

def keyword_query(question):
    return {
        "bool": {
            "must": {
                "multi_match": {
                    "query": f"{question}",
                    "fields": ["description^3", "text", "title"],
                    "type": "best_fields",
                    "boost": 0.5,
                }
            },
        }
    }

def multi_search(key_word):
    response = es.search(
        index='video-content',
        query=keyword_query(key_word),
        knn=knn_query(key_word),
        size=10
    )
    return [
        {
            'title': record['_source']['title'],
            'timecode_text': record['_source']['timecode_text'],
            'link': record['_source']['link'],
            'text': record['_source']['text'] 
        }
        for record in response["hits"]["hits"]
    ]

def ask_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7
    )
    return response.choices[0].message['content']

def make_context(question, records):
    return f"""
        You are a helpful program which is given a quesion and then the results from video transcripts which should answer the question given.
        This is an online course about using Retreival Augmented Generation (RAGS) and LLMs as well as how to evaluate the results of elasticsearch
        and the answers from the LLMs, how to monitor the restults etc. In the Course we used MAGE as an Orchestrator and PostgreSQL to capture user
        feedback. Given the question below, look at the records in the RECORDS section and return the best matching video link and a short summary
        and answer if you are able to which will answer the students question. Again your response should be a link from the records in the RECORDS
        section below. 

        Please return your answer as a dictionary without json```. Just have summary, title and link as as keys in the dictionary.

        QUESTION:
        {question}

        RECORDS:
        {records}
        """

def get_answer(question):
    search_results = multi_search(question)
    prompt = make_context(question, search_results)
    try:
        answer = ask_openai(prompt)
    except Exception as e:
        answer = f"Error: {e}"
    return answer

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    question = request.form['question']
    try:
        search_results = multi_search(question)
        elasticsearch_response = json.dumps(search_results)  # Serialize Elasticsearch response to JSON
        print("Elasticsearch Response:", elasticsearch_response)  # Log the response

        results = get_answer(question)
        response = json.loads(results) if isinstance(results, str) else results
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Raw Results:", results)
        return render_template('error.html', error=f"Error decoding JSON: {e} Results were {results}", results=results)
    except Exception as e:
        print("Error during search or OpenAI processing:", e)
        return render_template('error.html', error=str(e))

    return render_template(
        'response.html',
        response=response,
        question=question,
        elasticsearch_response=elasticsearch_response
    )

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback = request.form.get('feedback')
        question = request.form.get('question')
        response_summary = request.form.get('summary')
        response_title = request.form.get('title')
        response_link = request.form.get('link')
        comments = request.form.get('comments')

        # Log the received data
        print("Received Feedback Data:")
        print(f"Feedback: {feedback}")
        print(f"Question: {question}")
        print(f"Summary: {response_summary}")
        print(f"Title: {response_title}")
        print(f"Link: {response_link}")
        print(f"Comments: {comments}")

        conn = psycopg2.connect(
            dbname=postgres_db,
            user=postgres_user,
            password=postgres_password,
            host=postgres_host,
            port=postgres_port
        )
        cursor = conn.cursor()

        # Insert or update feedback in the database
        cursor.execute("""
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
        ))
        conn.commit()
        cursor.close()
        conn.close()

        # Render a success page
        return render_template('feedback_success.html', question=question)

    except Exception as e:
        print("Error during feedback submission:", e)
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
