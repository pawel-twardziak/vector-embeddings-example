import chromadb
from sentence_transformers import SentenceTransformer
import openai
from termcolor import colored
import re

# 1. Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="openai_embeddings_guide")

# 2. Initialize the model for generating embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Load and process the document
with open('openai_embeddings_guide.md', 'r', encoding='utf-8') as file:
    text = file.read()

# 4. Split the Markdown into fragments
fragments = re.split('## |### ' , text)  # Split into paragraphs
# Clean up fragments
fragments = [fragment.strip() for fragment in fragments if fragment.strip()]

# 5. Generate embeddings and add them to ChromaDB
for idx, fragment in enumerate(fragments):
    embedding = embedder.encode(fragment).tolist()
    collection.add(
        documents=[fragment],
        embeddings=[embedding],
        ids=[str(idx)]
    )

print(f"\n\n{colored('Embeddings created and saved to ChromaDB!', 'blue')} (length: {collection.count()})\n\n")

# 6. User query
query = "What are the advantages of using embeddings?"

# 7. Generate the query embedding
query_embedding = embedder.encode(query).tolist()

# 8. Search for the most similar fragments
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# 9. Prepare context for the LLM
context = "\n\n".join([doc for doc in results['documents'][0]])

# 10. Use a local LLM to generate a response
# Assuming the local LLM is available via the OpenAI API
clientLocal = openai.OpenAI(base_url="http://localhost:11434/v1/", api_key="None")

completion = clientLocal.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": "Answer the user's question based on the provided context. Provide the answer in bullet points. Use only context from the provided context, nthing else."
        },
        {
            "role": "user",
            "content": f"{query}\n\nContext:\n{context}"
        }
    ]
)

print(f"\n{colored('Local response to', 'light_blue')}\n\n"
      f"```\n{colored(query , 'light_grey')}\n```\n\n"
      f"{colored('based on context retrieved from ChromaDB:', 'light_blue')}\n\n"
      f"```\n{colored(context, 'light_grey')}\n```\n\n"
      f"{colored('is:', 'light_blue')}\n\n"
      f"```markdown\n{colored(completion.choices[0].message.content, 'black', 'on_white', ['bold'])}\n```")
