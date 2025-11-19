# 🤖 AI Support API (FastAPI + RAG + Embeddings + Mock Mode)

En moderne **AI-drevet kundestøtte-API-tjeneste** bygget med **FastAPI**, **vector search**, og en RAG-pipeline.  
Prosjektet støtter både:

- ✅ **Real Mode:** Bruker OpenAI embeddings + LLM (krever API-nøkkel)  
- 🧪 **Mock Mode (default):** Ingen API-nøkkel nødvendig, ingen kostnader — hele systemet fungerer med simulerte embeddings + simulerte RAG-svar  

Dette gjør prosjektet perfekt som **portfolio-showcase**, og trygt å kjøre for hvem som helst.

---

## 🌟 Funksjonalitet

### 🔍 Embeddings & Retrieval
- Vector search mot intern kunnskapsbase (`knowledge_base.csv`)
- Cosine similarity for dokumentrangering
- Egen retriever-klasse (`EmbeddingRetriever`)

### 🧠 RAG-system
- Bygger prompt fra de mest relevante dokumentene
- Gir et formulert AI-svar basert på kontekst  
- **Mock Mode:** Genererer realistiske, men kostnadsfrie "RAG-style" svar  

### 🚀 API (FastAPI)
- `POST /query` – spør chatboten  
- `GET /health` – helsesjekk  
- Full **Swagger UI** på `/docs`

---

## 🗂️ Mappestruktur

```text
ai-support-api/
├─ data/
│  ├─ knowledge_base.csv
│  ├─ kb_embeddings_openai.npz        # auto-generated
│  └─ kb_index.csv                    # auto-generated
│
├─ src/
│  ├─ api.py                          # FastAPI application
│  ├─ embeddings.py                   # builds embeddings (mock or real)
│  ├─ retriever.py                    # vector search
│  ├─ rag.py                          # RAG generation + mock RAG
│  ├─ utils.py                        # logging, env, paths, mock detection
│  ├─ models.py                       # Pydantic request/response schemas
│  └─ __init__.py
│
├─ client.py                          # Simple CLI API tester
├─ .env                               # (optional) Your OpenAI API key
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## 🧪 Mock Mode (Default)
Mock Mode aktiveres automatisk når OPENAI_API_KEY ikke finnes.

### I mock-modus:

- Embeddings genereres deteministisk -> ingen API-kall
- Query embeddings genereres deteministisk -> ingen API-kall
- RAG-svaret er et simulert AI-svar som viser hvordan en ekte RAG-modell ville reagert

### Dette gjør prosjektet: 
- trygt
- gratis
- kjørbart for alle
- perfekt for portefølje og intervjuer

### 💡 Du trenger ikke API-nøkkel for å teste prosjektet.
Alt fungerer 100% i Mock Mode

---

## ▶️ Kom i gang

### 1. Opprett og aktiver virtuelt miljø
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Bygg embeddings (Mock Mode eller Real Mode)
```bash
python -m src.embeddings
```
- Hvis ingen .env -> mock embeddings genereres

- hvis .env inneholder en nøkkel:
```ini
OPENAI_API_KEY=sk-proj-xxxx
```
-> ekte embeddings genereres

### 3. Start API-server
```bash
uvicorn src.api:app --reload
```
APIet kjører på:
👉 http://127.0.0.1:8000
👉 Swagger UI: http://127.0.0.1:8000/docs

---

## ❓ Eksempel på API-request
Gjennom Swagger UI eller client.py
```json
{
  "question": "How can I cancel my contract?",
  "top_k": 3,
  "include_context": true
}
```

---

## 🧠 Eksempel på Mock RAG Svar
```text
[MOCK RAG RESPONSE]
Based on the knowledge base, the question is most similar to 'Cancel contract'.
This is a simulated RAG-style response. In production, an LLM would generate a precise answer.
```

---

## 🧩 Teknologier brukt
- FastAPI - moderne Python API-framework
- OpenAI Models (valgfritt) - embeddings + RAG
- Mock AI System - null kostnader, fullt fungerende
- Vector Search - cosine similarity + normalized vectors
- Pydantic - request/response validering
- Uvicorn - ASGI-server

---

## 📌 Real Mode (valgfritt)
For å aktivere ekte AI-svar:

1. Legg inn .env i prosjekt-roten:
```ini
OPENAI_API_KEY=sk-proj-xxxx
```

2. Kjør embedding-pipelinen på nytt:
```bash
python -m src.embeddings
```

3. Start API:
```bash
uvicorn src.api:app --reload
```
Nå bruker systemet ekte OpenAI embeddings + GPT-4o-mini for RAG.

## 🧭 Forfatter
### Runar Olsen
Data Analyst — Python • Machine Learning • FastAPI • Power BI