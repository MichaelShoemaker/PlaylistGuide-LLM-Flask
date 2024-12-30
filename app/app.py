import os
import json
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

load_dotenv()
from utils.db_utils import init_db, db_conn, insert_feedback
from utils.elasticsearch_utils import multi_search
from utils.openai_utils import make_context, ask_openai
from utils.redis_utils import make_redis_client

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Create feedback table if it does not already exist
init_db()
redis_client = make_redis_client()

app = Flask(__name__)

def get_answer(question):
    # Get records from Elasticsearch
    search_results = multi_search(question, model)
    # Make a prompt along with the elasticsearch records
    prompt = make_context(question, search_results)
    try:
        answer = ask_openai(prompt, test=True)
    except Exception as e:
        answer = f"Error: {e}"
    return answer


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    question = request.form['question']
    try:
        #The below was just for troubleshooting
        # search_results = multi_search(question)
        # elasticsearch_response = json.dumps(search_results, indent=2)  # Serialize Elasticsearch response to JSON for debugging
        # print("Elasticsearch Response:", elasticsearch_response)  # Log the response

        results = get_answer(question)

        # Ensure `results` is a dictionary
        if isinstance(results, str):
            try:
                response = json.loads(results)  # Decode JSON string to Python dictionary
            except json.JSONDecodeError as e:
                print("Error decoding JSON response:", e)
                print("Raw Results (string):", results)
                return render_template(
                    'error.html',
                    error=f"Error decoding JSON: {e}",
                    results=results
                )
        elif isinstance(results, dict):
            response = results
        else:
            print("Unexpected type for results:", type(results))
            return render_template(
                'error.html',
                error="Unexpected response type.",
                results=str(results)
            )

    except Exception as e:
        print("Error during search or OpenAI processing:", e)
        return render_template('error.html', error=str(e))

    return render_template(
        'response.html',
        response=response,
        question=question
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


        with db_conn() as conn:
            with conn.cursor() as cursor:
                # Unpack the tuple into the query and parameters
                query, params = insert_feedback(question, response_summary, feedback, comments, response_title, response_link)
                cursor.execute(query, params)  # Pass the query and parameters separately
            conn.commit()

            return render_template('feedback_success.html', question=question)

    except Exception as e:
        print("Error during feedback submission:", e)
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
