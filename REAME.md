# Lab 7 - Especialização de LLMs com LoRA e QLoRA

Pipeline completo de fine-tuning de um modelo de linguagem fundacional (LLaMA-2 7B) utilizando PEFT/LoRA com quantização QLoRA (4-bit NF4), viabilizando o treinamento em hardware limitado.

## Pré-requisitos

```
pip install torch transformers datasets peft trl bitsandbytes openai accelerate
```

> **Requisito de hardware:** GPU com mínimo 16 GB de VRAM (recomendado Google Colab Pro com A100) ou GPU consumer 24 GB com QLoRA.

## Estrutura do Projeto

```
lab7/
├── gerar_dataset.py       # Passo 1: geração do dataset via API OpenAI
├── lab7.py                # Passos 2-5: pipeline completo QLoRA + SFTTrainer
├── dataset_treino.jsonl   # 45 pares de instrução/resposta (domínio: IA/ML)
├── dataset_teste.jsonl    # 5 pares para avaliação
└── README.md
```

## Como Executar

### Passo 1 — Gerar dataset (opcional, já incluído)
```bash
export OPENAI_API_KEY="sua_chave_aqui"
python3 gerar_dataset.py
```
Os arquivos `dataset_treino.jsonl` e `dataset_teste.jsonl` já estão incluídos no repositório com 50 pares gerados sobre o domínio de Inteligência Artificial e Aprendizado de Máquina.

### Passo 2 a 5 — Fine-tuning QLoRA
```bash
# Necessário aceitar os termos do LLaMA-2 no Hugging Face e fazer login:
huggingface-cli login
python3 lab7.py
```

## Configurações Implementadas

### Quantização (Passo 2)
| Parâmetro           | Valor    |
|---------------------|----------|
| load_in_4bit        | True     |
| quant_type          | nf4      |
| compute_dtype       | float16  |
| double_quant        | True     |

### LoRA (Passo 3)
| Parâmetro           | Valor               |
|---------------------|---------------------|
| rank (r)            | 64                  |
| alpha               | 16                  |
| dropout             | 0.1                 |
| task_type           | CAUSAL_LM           |
| target_modules      | q_proj, k_proj, v_proj, o_proj |

### Otimizador e Scheduler (Passo 4)
| Parâmetro           | Valor               |
|---------------------|---------------------|
| otimizador          | paged_adamw_32bit   |
| lr_scheduler        | cosine              |
| warmup_ratio        | 0.03                |
| learning_rate       | 2e-4                |
| epochs              | 3                   |
| batch_size          | 4                   |
| grad_accumulation   | 4 (efetivo: 16)     |

## Resultado Esperado

O modelo adaptado deve responder perguntas sobre IA/ML com qualidade notavelmente superior ao modelo base para o domínio específico. O adaptador LoRA resultante ocupa apenas alguns MB, em contraste com os ~14 GB do modelo base em float16.

## Ferramentas utilizadas

- **OpenAI API**: geração do dataset sintético de instrução (Passo 1) — `gerar_dataset.py`
- **bitsandbytes**: quantização 4-bit NF4 para redução de memória (Passo 2)
- **peft**: configuração e aplicação do LoRA sobre o modelo base (Passo 3)
- **trl/SFTTrainer**: orquestração do training loop com PEFT integrado (Passo 4)
- **transformers**: carregamento do modelo LLaMA-2 e tokenizador