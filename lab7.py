import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

MODELO_BASE    = "meta-llama/Llama-2-7b-hf"
SAIDA_ADAPTADOR = "./adaptador_lora"
TREINO_JSONL   = "dataset_treino.jsonl"
TESTE_JSONL    = "dataset_teste.jsonl"

LORA_R         = 64
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1

MAX_SEQ_LEN    = 512
EPOCHS         = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LR             = 2e-4
WARMUP_RATIO   = 0.03


def carregar_jsonl(caminho):
    pares = []
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            pares.append(json.loads(linha.strip()))
    return pares

def formatar_instrucao(exemplo):
    return f"### Instrução:\n{exemplo['prompt']}\n\n### Resposta:\n{exemplo['response']}"


print("=" * 60)
print("PASSO 1 — Carregando Dataset")
print("=" * 60)

pares_treino = carregar_jsonl(TREINO_JSONL)
pares_teste  = carregar_jsonl(TESTE_JSONL)

dataset_treino = Dataset.from_list([
    {"text": formatar_instrucao(p)} for p in pares_treino
])
dataset_teste = Dataset.from_list([
    {"text": formatar_instrucao(p)} for p in pares_teste
])

print(f"exemplos de treino : {len(dataset_treino)}")
print(f"exemplos de teste  : {len(dataset_teste)}")
print(f"\nexemplo formatado:\n{dataset_treino[0]['text'][:200]}...")


print("\n" + "=" * 60)
print("PASSO 2 — Configurando Quantização (QLoRA / 4-bit NF4)")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("BitsAndBytesConfig configurado:")
print(f"  quantizacao  : 4-bit NF4")
print(f"  compute_dtype: float16")
print(f"  double_quant : True")


print("\n" + "=" * 60)
print("PASSO 3 — Arquitetura LoRA")
print("=" * 60)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
print("LoraConfig configurado:")
print(f"  rank (r)    : {LORA_R}")
print(f"  alpha       : {LORA_ALPHA}")
print(f"  dropout     : {LORA_DROPOUT}")
print(f"  task_type   : CAUSAL_LM")
print(f"  target      : q_proj, k_proj, v_proj, o_proj")


print("\n" + "=" * 60)
print("PASSO 4 — Carregando Modelo Base e Tokenizador")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
modelo.config.use_cache = False
modelo.config.pretraining_tp = 1

n_params = sum(p.numel() for p in modelo.parameters())
print(f"parametros totais do modelo base: {n_params:,}")


print("\n" + "=" * 60)
print("PASSO 5 — Training Loop (SFTTrainer)")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=SAIDA_ADAPTADOR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    optim="paged_adamw_32bit",
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=modelo,
    train_dataset=dataset_treino,
    eval_dataset=dataset_teste,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
    args=training_args,
)

n_treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
n_total      = sum(p.numel() for p in modelo.parameters())
print(f"parametros treinaveis (LoRA): {n_treinaveis:,} ({100 * n_treinaveis / n_total:.2f}%)")
print(f"iniciando treinamento por {EPOCHS} epocas...\n")

trainer.train()


print("\n" + "=" * 60)
print("SALVANDO ADAPTADOR")
print("=" * 60)

trainer.model.save_pretrained(SAIDA_ADAPTADOR)
tokenizer.save_pretrained(SAIDA_ADAPTADOR)
print(f"adaptador LoRA salvo em: {SAIDA_ADAPTADOR}")


print("\n" + "=" * 60)
print("INFERÊNCIA COM MODELO FINE-TUNED")
print("=" * 60)

modelo.eval()
frase_teste = pares_teste[0]["prompt"]
prompt_formatado = f"### Instrução:\n{frase_teste}\n\n### Resposta:\n"

inputs = tokenizer(prompt_formatado, return_tensors="pt").to(modelo.device)

with torch.no_grad():
    output = modelo.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

resposta = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nprompt  : {frase_teste}")
print(f"\nresposta gerada:\n{resposta[len(prompt_formatado):]}")