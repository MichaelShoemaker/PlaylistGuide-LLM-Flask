import os
import openai

openai_api_key = os.getenv("OPENAI_API_KEY", "your_default_openai_key")
openai.api_key = openai_api_key

def ask_openai(prompt, test=False):
    if test == True:
        return '''{
    "summary": "In this video, the presenter discusses how to calculate costs associated with using the OpenAI API. They explain the formula for calculating costs based on the number of input and output tokens, providing a practical example of how to implement this in code.",
    "title": "LLM Zoomcamp 7.5 - Monitoring and containerization",
    "link": "https://www.youtube.com/watch?v=nQda9etJWW8&t=3135s"}
    '''
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
