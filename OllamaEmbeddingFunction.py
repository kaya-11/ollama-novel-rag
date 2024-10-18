from chromadb import Documents, EmbeddingFunction, Embeddings
import ollama as ollama
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEl = os.getenv("EMBEDDING_MODEl")

class OllamaEmbeddingFunction(EmbeddingFunction):

    def ollama_embed(self, text: str) -> List[float]:
        """
        Function to generate embeddings using the llama3 model from Ollama.

        :param text: Input text to embed
        :return: A list of floats representing the embedding
        """
        response = ollama.embeddings(model=EMBEDDING_MODEl, prompt=text)
        embedding = response['embedding']  # Assuming the embedding comes under 'embedding' key
        return embedding

    def __call__(self, input: Documents) -> Embeddings:
        return [self.ollama_embed(text=Documents.page_content) for chunk in Documents]
