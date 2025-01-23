import os
import openai
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

openai.api_key = 'sk-proj-deiKeghcMl8Nq6xXOjIST3BlbkFJbiBzDDo2SVHFX0PC6fXX'  # Replace with your actual API key

file_path = r'C:\Users\moonl\Documents\FatwaGpt\Fatwa_with_embeddings1.csv'
df = pd.read_csv(file_path)
df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return np.array(response['data'][0]['embedding'])

def semantic_search(query, df, top_n=5):
    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], list(df['embedding'].values))[0]
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_n_indices].copy()
    results['similarity'] = similarities[top_n_indices]
    return results[['Content', 'similarity']]

def summarize_content(content, max_tokens=500):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text in Malay."},
            {"role": "user", "content": f"Ringkaskan teks berikut dalam Bahasa Melayu dan hadkan hanya kepada 500 patah perkataan dan hanya menjawab apa yang ditanya sahaja:\n\n{content}"}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def split_text(text, max_length=2048):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    search_results = semantic_search(query, df, top_n=1)
    summarized_results = [summarize_content(result) for result in search_results['Content'].values]
    response_text = ' '.join(summarized_results)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
