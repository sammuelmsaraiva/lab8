import re
from collections import defaultdict

vocab = {
    'l o w </w>':       5,
    'l o w e r </w>':   2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}


def get_stats(vocab):
    pares = defaultdict(int)
    for palavra, freq in vocab.items():
        simbolos = palavra.split()
        for i in range(len(simbolos) - 1):
            pares[(simbolos[i], simbolos[i + 1])] += freq
    return pares


def merge_vocab(pair, v_in):
    v_out = {}
    bigrama = re.escape(' '.join(pair))
    padrao  = re.compile(r'(?<!\S)' + bigrama + r'(?!\S)')
    for palavra in v_in:
        nova_palavra = padrao.sub(''.join(pair), palavra)
        v_out[nova_palavra] = v_in[palavra]
    return v_out


print("=" * 60)
print("TAREFA 1 — Motor de Frequências")
print("=" * 60)

stats = get_stats(vocab)
print("\nfrequencia de todos os pares adjacentes:")
for par, freq in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {par} → {freq}")

par_max = max(stats, key=stats.get)
print(f"\npar mais frequente: {par_max} → {stats[par_max]}")
assert stats[('e', 's')] == 9, "ERRO: par ('e', 's') deveria ter frequencia 9"
print("validacao ok: ('e', 's') = 9")


print("\n" + "=" * 60)
print("TAREFA 2 — Loop de Fusão (K=5 iterações)")
print("=" * 60)

NUM_MERGES = 5
merges     = []

for i in range(1, NUM_MERGES + 1):
    stats      = get_stats(vocab)
    melhor_par = max(stats, key=stats.get)
    vocab      = merge_vocab(melhor_par, vocab)
    merges.append(melhor_par)
    print(f"\n  iteracao {i}")
    print(f"  par fundido: {melhor_par} → {''.join(melhor_par)}")
    print(f"  vocab atual:")
    for palavra, freq in vocab.items():
        print(f"    '{palavra}': {freq}")

print(f"\nmerges aprendidos: {merges}")
tokens_finais = set()
for palavra in vocab:
    tokens_finais.update(palavra.split())
print(f"tokens no vocabulario final: {sorted(tokens_finais)}")


print("\n" + "=" * 60)
print("TAREFA 3 — Integração Industrial e WordPiece")
print("=" * 60)

try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    frase = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."
    tokens = tokenizer.tokenize(frase)

    print(f"\nfrase original:\n  {frase}")
    print(f"\ntokens WordPiece ({len(tokens)} tokens):")
    print(f"  {tokens}")

    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nids correspondentes:")
    print(f"  {ids}")

    tokens_com_hash = [t for t in tokens if t.startswith("##")]
    print(f"\ntokens com ## (continuacao de sub-palavra):")
    print(f"  {tokens_com_hash}")

except ImportError:
    print("\nbiblioteca 'transformers' nao instalada.")
    print("execute: pip install transformers")
    print("e rode novamente para ver o resultado do WordPiece.")