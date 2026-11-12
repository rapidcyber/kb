import psycopg
from pgvector .psycopg import register_vector
import ollama

# Connect to the knowledge database (password comes from PGPASSWORD)
conn = psycopg.connect(dbname="knowledge", user="postgres", host="localhost")
register_vector(conn)
print("Database: connected")

with conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM chunks")
    print("Chunks stored", cur.fetchone()[0])

# Ask the local embedding model to embed a test secntence
result = ollama.embed(model="nomic-embed-text", input="hello world")
print("Embedding size:", len(result["embeddings"][0]))

conn.close()

