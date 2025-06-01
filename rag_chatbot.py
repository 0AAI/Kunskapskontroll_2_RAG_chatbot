# rag_chatbot.py

import os
from dotenv import load_dotenv
from google.genai import types
import numpy as np
from pypdf import PdfReader
from google import genai


# 1) Initiera API-nyckeln och Gemini-klienten, läser in API_KEY via .env filen som finns i samma mapp. 

load_dotenv()  # Läser .env från aktuell mapp
api_key = os.getenv("API_KEY")
if api_key is None:
    raise RuntimeError("API_KEY saknas i .env!")

client = genai.Client(api_key=api_key)

# 2) Läs in PDF filen och printa hela texten

def load_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() or ""
    return all_text

# pdf filen
text = load_pdf("Tapper_info.pdf")

# 3) Chunkning, fixed-length

def chunk_text(text: str, n: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i : i + n])
    return chunks

chunks = chunk_text(text, n=500, overlap=100)



# 4) Skapa embeddings‐funktion 

def create_embeddings(text_list, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"):
    return client.models.embed_content(
        model=model,
        contents=text_list,
        config=types.EmbedContentConfig(task_type=task_type)
    )

# Generera embeddings för alla chunks i ett anrop
embeddings = create_embeddings(chunks)

# 5) Cosinus‐funktion

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 6) Semantisk sökning

def semantic_search(query, chunks, embeddings_obj, k=5):
    # Hämta ut embedding för frågan
    response_q = create_embeddings([query])  # Skicka som lista
    query_emb = response_q.embeddings[0].values

    # Extrahera chunk‐embeddingar från embeddings_obj
    raw_embeddings = [emb_obj.values for emb_obj in embeddings_obj.embeddings]

    # Beräkna cosinus‐likheter
    similarity_scores = []
    for i, chunk_emb in enumerate(raw_embeddings):
        sim = cosine_similarity(query_emb, chunk_emb)
        similarity_scores.append((i, sim))

    # Sortera och plocka topp k-index
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarity_scores[:k]]

    # Returnera chunk­text för högst relevanta index
    return [chunks[i] for i in top_indices]

# 7) System-prompt och generering av användarprompt

system_prompt = """Jag kommer ställa dig en fråga, och jag vill
att du svarar baserat bara på kontexten jag skickar med, och ingen annan information.
Om det inte finns nog med information i kontexten för att svara på frågan,
säg "Det vet jag inte". Försök inte att gissa.
Formulera dig enkelt och dela upp svaret i fina stycken."""

def generate_user_prompt(query):
    context = "\n".join(semantic_search(query, chunks, embeddings, k=5))
    return f"Frågan är {query}. Här är kontexten:\n{context}."


def generate_response(system_prompt, user_message, model="gemini-2.0-flash"):
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        contents=generate_user_prompt(user_message)
    )
    return response

# 8) Huvudloopen (terminal‐läge)

def main():
    print("Välkommen till RAG-chatboten! Vad har du för frågor om Tapper trädfällning? (skriv 'avsluta' för att avsluta)\n")
    while True:
        user_input = input("Du: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Chatbot avslutar. Hej då!")
            break

        resp = generate_response(system_prompt, user_input)
        print("Bot:", resp.text, "\n")


if __name__ == "__main__":
    main()
