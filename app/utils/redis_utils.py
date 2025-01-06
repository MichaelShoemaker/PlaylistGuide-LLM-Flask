import json
import os

import numpy as np
import redis
from sklearn.metrics.pairwise import cosine_similarity


# Initialize Redis connection
def make_redis_client():
    return redis.StrictRedis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )


def check_redis(question):
    # Encode the current question
    current_embedding = model.encode(question).reshape(1, -1)

    # Retrieve stored questions and their embeddings from Redis
    stored_questions = redis_client.hgetall(
        "questions")  # {question: embedding_json}

    for stored_question, embedding_json in stored_questions.items():
        stored_embedding = np.array(json.loads(embedding_json)).reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            current_embedding, stored_embedding)[0][0]
        if similarity > 0.85:
            print(
                f"Found similar question in Redis: {stored_question} (Similarity: {similarity})"
            )
            # Return cached answer
            return redis_client.get(f"answer:{stored_question}")

    # If no similar question is found, proceed with Elasticsearch and OpenAI
    search_results = multi_search(question, model)
    prompt = make_context(question, search_results)

    try:
        answer = ask_openai(prompt)
        # Cache the new question and its answer in Redis
        redis_client.hset("questions", question,
                          json.dumps(current_embedding.tolist()))
        redis_client.set(f"answer:{question}", answer)
    except Exception as e:
        answer = f"Error: {e}"

    return answer
