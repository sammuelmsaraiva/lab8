import os
import json
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DOMINIO = "inteligencia artificial e aprendizado de maquina"
TOTAL   = 50
TREINO  = int(TOTAL * 0.9)

SYSTEM_PROMPT = (
    "Voce e um especialista em inteligencia artificial. "
    "Gere exatamente um par de instrucao e resposta no formato JSON com as chaves "
    "'prompt' e 'response'. A instrucao deve ser uma pergunta ou tarefa sobre "
    f"o dominio: {DOMINIO}. A resposta deve ser clara, precisa e didatica. "
    "Retorne apenas o JSON, sem texto adicional."
)

def gerar_par(indice):
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Gere o par numero {indice + 1}."},
        ],
        temperature=0.9,
        max_tokens=512,
    )
    conteudo = resposta.choices[0].message.content.strip()
    par = json.loads(conteudo)
    return par

pares = []
print(f"gerando {TOTAL} pares no dominio: {DOMINIO}\n")
for i in range(TOTAL):
    try:
        par = gerar_par(i)
        pares.append(par)
        print(f"  [{i + 1:02d}/{TOTAL}] prompt: {par['prompt'][:60]}...")
    except Exception as e:
        print(f"  [{i + 1:02d}/{TOTAL}] erro: {e}")

random.shuffle(pares)
treino = pares[:TREINO]
teste  = pares[TREINO:]

with open("dataset_treino.jsonl", "w", encoding="utf-8") as f:
    for par in treino:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")

with open("dataset_teste.jsonl", "w", encoding="utf-8") as f:
    for par in teste:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")

print(f"\ndataset gerado:")
print(f"  treino : {len(treino)} pares → dataset_treino.jsonl")
print(f"  teste  : {len(teste)} pares → dataset_teste.jsonl")