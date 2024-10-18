from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import nltk
import shutil
import os
from dotenv import load_dotenv
from OllamaEmbeddingFunction import OllamaEmbeddingFunction

nltk.download('punkt_tab')

load_dotenv()

NOVEL_NAME = os.getenv("NOVEL_NAME")
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")

def main():
    print(f"Novel: {NOVEL_NAME}")
    print(f"Chroma-Path: {CHROMA_PATH}")
    print(f"Data-Path: {DATA_PATH}")
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob=f"{NOVEL_NAME}*.md")
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?"],
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def generate_ollama_embeddings(chunks):
    embeddingFuction = OllamaEmbeddingFunction()
    return [embeddingFuction.ollama_embed(text=chunk.page_content) for chunk in chunks]


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print(f"Start saving {len(chunks)} chunks to {CHROMA_PATH}.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.create_collection(name="docs")

    embeddings = generate_ollama_embeddings(chunks)

    documents = [chunk.json() for chunk in chunks]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
