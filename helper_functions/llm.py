import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv('.env')

client = OpenAI(os.getenv('OPENAI_API_KEY'))

#%%

def get_embedding(input, model = 'text-embedding-3-small'):
    response = client.embedding.create(
        input = input,
        model = model
    )
    return [x.embedding for x in response.data]

# This is the "Updated" helper function for calling LLM
def get_completion(prompt, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1, json_output=False):
    if json_output == True:
        output_json_structure ={"type": "json_object"}
    else:
        output_json_structure = None
        
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        response_format=output_json_structure
    )

    return response.choices[0].message.content


# Note that this function directly take in "messages" as the parameter.
def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content