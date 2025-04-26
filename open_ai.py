# Mini-application: Sending multiple texts to OpenAI Embeddings,
# saving to a CSV file, and searching for the closest text.
import openai
import csv
import numpy as np
import sklearn.metrics.pairwise as skp
import dotenv
from termcolor import colored
import re

# 1. Set your OpenAI API key
dotenv.load_dotenv()

# 2. Use the OpenAI API to generate embeddings
clientOpenAI = openai.OpenAI()

# 3. Load and process the document
with open('openai_embeddings_guide.md', 'r', encoding='utf-8') as file:
    text = file.read()

# 4. Split the Markdown into fragments
texts = re.split('## |### ' , text)  # Split into paragraphs
# Clean up fragments
texts = [fragment.strip() for fragment in texts if fragment.strip()]

# 4. Sending texts to OpenAI and receiving embeddings
embeddings = []
for text in texts:
    response1 = clientOpenAI.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response1.data[0].embedding
    embeddings.append((text, embedding))

# 5. Saving embeddings to a CSV file
with open('embeddings.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text'] + [f'dim_{i}' for i in range(len(embeddings[0][1]))])
    for text, emb in embeddings:
        writer.writerow([text] + emb)

print(f"\n\n{colored('Embeddings created saved to the file \'embeddings.csv\'!', 'blue')} (length: {len(embeddings)})\n\n")

# 6. Function to search for the most similar text
def find_most_similar(query_text):
    # 6.1 First, encode the query
    response2 = clientOpenAI.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(response2.data[0].embedding).reshape(1, -1)

    # 6.2 Load embeddings from the CSV file
    database_texts = []
    database_embeddings = []

    with open('embeddings.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            database_texts.append(row[0])
            database_embeddings.append(list(map(float, row[1:])))

    database_embeddings = np.array(database_embeddings)

    # 6.3 Calculate cosine similarity
    similarities = skp.cosine_similarity(query_embedding, database_embeddings)[0]
    most_similar_idx = np.argmax(similarities)

    print(f"\n{colored('The most similar text to the query:', 'light_blue')}\n\n"
          f"```\n{colored(query_text, 'light_grey')}\n```\n\n"
          f"{colored('among the texts in the content of the file:' , 'light_blue')}\n\n"
          f"```\n{colored('openai_embeddings_guide.md', 'light_grey')}\n```\n\n"
          f"{colored('converted to embeddings is:', 'light_blue')}\n\n"
          f"```\n{colored(texts[most_similar_idx], 'black', 'on_white', ['bold'])}\n```\n\n"
          f"(Similarity: {similarities[most_similar_idx]:.3f})")

# 7. Testing!
if __name__ == "__main__":
    find_most_similar("What are the advantages of using embeddings?")
