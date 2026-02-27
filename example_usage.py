"""
Beispiel: AcademicAI verwenden.
Credentials werden aus .env geladen.
"""

import academicai

# Verfügbare Modelle abrufen
print("=== Verfügbare Modelle ===")
models = academicai.get_models()
for m in models["data"]:
    print(f"  - {m['id']} (context: {m.get('context_window', '?')})")

# Einfache Chat-Anfrage
print("\n=== Chat Completion ===")
response = academicai.completion(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Erkläre kurz den Begriff 'Micro-Credential'."}
    ],
)
print(response.choices[0].message.content)
print(f"\nToken-Usage: {response.usage}")

# Mit RAG / Tailored AI
# response_rag = academicai.completion(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Was steht in meiner Knowledge Base?"}],
#     extra_body={"tailoredAiId": "your-tailored-ai-id-here"},
# )
