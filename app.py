import ollama

# Define your sentence or word to embed
text_to_embed = "Artificial intelligence is transforming industries."

# Generate the embedding
response = ollama.embeddings(model='llama3.2', prompt=text_to_embed)

print(response['embedding'])
