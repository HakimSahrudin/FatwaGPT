import openai
import pandas as pd
import os

# Load the CSV file
file_path = r'C:\Users\moonl\Documents\Fatwagpt\Fatwa.csv'
df = pd.read_csv(file_path)

# Initialize OpenAI API key
openai.api_key = 'sk-proj-deiKeghcMl8Nq6xXOjIST3BlbkFJbiBzDDo2SVHFX0PC6fXX'  # Use your OpenAI API key

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Generate embeddings for each row in the dataframe
df['embedding'] = df['Content'].apply(lambda x: get_embedding(x))

# Save the dataframe with embeddings to a new CSV file
output_file_path = r'C:\Users\moonl\Documents\Fatwagpt\Fatwa_with_embeddings.csv'
df.to_csv(output_file_path, index=False)

print("Embeddings have been generated and saved to", output_file_path)
