from ollama import generate
from langchain.prompts import ChatPromptTemplate
from OllamaEmbeddingFunction import OllamaEmbeddingFunction
import chromadb
import json

PROMPT_TEMPLATE = """
You are an expert about the novel Catcher in the rye bye J.D. Salinger. 

Answer with only maximum 3 sentences the following question: {question}
"""


def main():

    query_text = "Who is Stradlater?"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)
    print(prompt)

    response = generate('llama3.2', prompt)
    print(response['response'])

if __name__ == "__main__":
    main()