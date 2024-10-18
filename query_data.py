from ollama import generate
from langchain.prompts import ChatPromptTemplate
import chromadb
import json
import os
from dotenv import load_dotenv
from OllamaEmbeddingFunction import OllamaEmbeddingFunction

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH")

PROMPT_TEMPLATE = """
You are an expert about the novel Catcher in the rye bye J.D. Salinger. Use the following context to answer the question:

{context}

---

Answer the following question with up to 3 sentences max using the context above : {question}
"""


def main():

    query_text = "What kind of character is Holdens roommate Stradlater?"

    # Prepare the DB.
    embedding_function = OllamaEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="docs", embedding_function=embedding_function)

    query_vector = embedding_function.ollama_embed(query_text)

    results = collection.query(
        query_embeddings=[query_vector],  # You can query multiple vectors
        n_results=6,                      # Number of results to return
        include=["distances", "documents"]  # Include distances and metadata in results
    )

    # Search the DB.
    if len(results) == 0 or len(results['distances'][0]) == 0 or results['distances'][0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([json.loads(result)["page_content"] for result in results["documents"][0]])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    response = generate('llama3.2', prompt)
    print(response['response'])


if __name__ == "__main__":
    main()