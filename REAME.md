# Lab 6 - Construindo um Tokenizador BPE e Explorando o WordPiece

Este laboratório implementa o algoritmo Byte Pair Encoding (BPE) do zero e explora o funcionamento do WordPiece na prática usando o tokenizador multilíngue do BERT.

## Pré-requisitos

```
pip install transformers
```

## Estrutura do Projeto

- `lab6.py`: Implementação completa do BPE e integração com o WordPiece via Hugging Face.

## Como Executar

```
python3 lab6.py
```

## Funcionalidades Implementadas

- **get_stats(vocab)**: Varre o corpus e conta a frequência de todos os pares adjacentes de símbolos. Validado: o par `('e', 's')` retorna frequência 9.
- **merge_vocab(pair, v_in)**: Substitui todas as ocorrências do par mais frequente pela versão fundida, retornando o vocabulário atualizado.
- **Loop de Treinamento BPE (K=5)**: Executa 5 rodadas de fusão, imprimindo o par fundido e o estado do vocabulário a cada iteração. Após as 5 rodadas, é possível observar a formação do sufixo morfológico `est</w>`.
- **WordPiece via Hugging Face**: Tokenização da frase de teste com `bert-base-multilingual-cased`, exibindo os tokens resultantes e os IDs correspondentes.

## Resultado das 5 Iterações BPE

| Iteração | Par Fundido         | Token Resultante |
|----------|---------------------|------------------|
| 1        | ('e', 's')          | es               |
| 2        | ('es', 't')         | est              |
| 3        | ('est', '</w>')     | est</w>          |
| 4        | ('l', 'o')          | lo               |
| 5        | ('lo', 'w')         | low              |

## Sobre o sinal `##` no WordPiece

No WordPiece (usado no BERT), o prefixo `##` indica que aquele token é uma **continuação** de uma palavra, ou seja, não está no início dela. Por exemplo, a palavra `inconstitucionalmente` pode ser segmentada em `inconstitucional` + `##mente`, onde `##mente` sinaliza que esse fragmento se une ao token anterior para formar a palavra completa.

Esse mecanismo resolve o problema do vocabulário desconhecido (OOV — *Out-of-Vocabulary*): em vez de tratar uma palavra rara como um único token `[UNK]`, o modelo a decompõe em sub-palavras conhecidas. Assim, mesmo palavras que nunca apareceram no treinamento podem ser representadas como sequências de sub-palavras, garantindo que o modelo sempre receba uma representação numérica válida e nunca trave diante de vocabulário novo.