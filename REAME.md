# Lab 8 - Alinhamento Humano com DPO

Pipeline completo de alinhamento de um LLM utilizando Direct Preference Optimization (DPO), substituindo o complexo pipeline de RLHF. O objetivo é garantir que o modelo seja Útil, Honesto e Inofensivo (HHH — Helpful, Honest, Harmless), suprimindo respostas tóxicas ou maliciosas.

## Pré-requisitos

```
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

> **Hardware recomendado:** Google Colab Pro com GPU A100 (40 GB VRAM).

## Estrutura do Projeto

```
lab8/
├── lab8.py              # Pipeline completo DPO (4 passos)
├── dataset_hhh.jsonl    # Dataset de preferências HHH (31 pares)
└── README.md
```

## Como Executar

```bash
huggingface-cli login
python3 lab8.py
```

## Dataset de Preferências (Passo 1)

O arquivo `dataset_hhh.jsonl` contém 31 pares no formato obrigatório com as chaves `prompt`, `chosen` e `rejected`, cobrindo categorias de segurança como:

- Ataques a sistemas e bancos de dados
- Violação de privacidade e interceptação de comunicações
- Geração de malware e ransomware
- Fraude financeira e lavagem de dinheiro
- Discurso de ódio e incitação à violência
- Desinformação e deepfakes
- Manipulação psicológica e assédio

## Configurações Implementadas (Passos 2 e 3)

| Parâmetro           | Valor             |
|---------------------|-------------------|
| beta (DPO)          | 0.1               |
| LoRA rank           | 64                |
| LoRA alpha          | 16                |
| LoRA dropout        | 0.1               |
| otimizador          | paged_adamw_32bit |
| lr_scheduler        | cosine            |
| warmup_ratio        | 0.03              |
| quantização         | 4-bit NF4         |

## O Papel Matemático do Parâmetro Beta (β)

O DPO otimiza diretamente a política do modelo sem um modelo de recompensa explícito. A função objetivo é:

```
L_DPO(π_θ) = -E[ log σ( β · log(π_θ(y_w|x)/π_ref(y_w|x)) - β · log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

onde `y_w` é a resposta escolhida (*chosen*), `y_l` é a resposta rejeitada (*rejected*) e `π_ref` é o modelo de referência congelado.

O **β funciona como um imposto de regularização** sobre a divergência de Kullback-Leibler (KL) entre o modelo em treinamento (ator) e o modelo de referência. Matematicamente, ele controla o trade-off entre duas forças opostas: maximizar a probabilidade das respostas *chosen* em relação às *rejected* e manter a distribuição do modelo próxima à distribuição original pré-treinada.

Um **β próximo de zero** torna o modelo agressivo na otimização de preferências, podendo colapsar a fluência e diversidade da linguagem — o modelo "esquece" como escrever de forma natural ao focar apenas em preferir *chosen* sobre *rejected*. Um **β muito alto** mantém o modelo quase idêntico à referência, tornando o alinhamento ineficaz. O valor **β = 0.1** é o padrão da literatura (paper original DPO, Rafailov et al., 2023) por equilibrar alinhamento efetivo com preservação da fluência e capacidade geral do modelo base.