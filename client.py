import requests

API_URL = "http://127.0.0.1:8000/query"

def ask(question: str, top_k: int = 3):
    payload = {
        "question": question,
        "top_k": top_k,
        "include_context": True
    }

    print("\n📨 Sending request...")
    resp = requests.post(API_URL, json=payload)

    if resp.status_code != 200:
        print("❌ Error:", resp.text)
        return

    data = resp.json()

    print("\n🤖 AI ANSWER:")
    print(data["answer"], "\n")

    print("📄 Retrieved documents:")
    for d in data["docs"]:
        print(f"- {d['title']}  (score: {d['score']:.3f})")

if __name__ == "__main__":
    ask("How can I cancel my contract?")