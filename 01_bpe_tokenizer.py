"""
01_bpe_tokenizer.py
-------------------
Build a Byte Pair Encoding (BPE) tokenizer FROM SCRATCH.
No libraries doing the heavy lifting — you see every merge happen.

RUN:
    python 01_bpe_tokenizer.py

EXPERIMENT FLAGS (change these and re-run):
    NUM_MERGES   — how many BPE merge operations to perform
    CORPUS       — change this to any text and watch vocab change
"""

import re
from collections import defaultdict, Counter
from data.corpus import CORPUS_TECH
CORPUS = CORPUS_TECH

# ─────────────────────────────────────────────
# EXPERIMENT KNOB 1: How many merges to perform
# Try: 5, 20, 100 — watch the vocab grow
# ─────────────────────────────────────────────
NUM_MERGES = 30

# ─────────────────────────────────────────────
# EXPERIMENT KNOB 2: Your training corpus
# Try adding more domain-specific text and see
# which tokens get merged first
# ─────────────────────────────────────────────
CORPUS = """
the cat sat on the mat
the cat ate the rat
the rat ran from the cat
a fat cat sat on a flat mat
cats are not rats and rats are not cats
tokenization is the first step in natural language processing
natural language models learn from tokens not words
byte pair encoding merges the most frequent pairs first
"""


# ─────────────────────────────────────────────────────────────
# STEP 1: Build initial vocabulary
# Every word is split into characters + end-of-word marker </w>
# This is how BPE starts — everything is individual characters
# ─────────────────────────────────────────────────────────────
def get_vocab(corpus: str) -> dict:
    """
    Convert corpus into a frequency dict of space-separated characters.
    'cat' → ('c', 'a', 't', '</w>') : frequency

    WHY </w>?
    It marks word boundaries so 'est' inside 'test' and
    'est</w>' at end of 'longest' are treated differently.
    """
    vocab = defaultdict(int)
    for line in corpus.strip().split("\n"):
        for word in line.strip().split():
            # Represent word as tuple of chars + end marker
            char_tuple = tuple(list(word) + ["</w>"])
            vocab[char_tuple] += 1
    return dict(vocab)


# ─────────────────────────────────────────────────────────────
# STEP 2: Count all adjacent symbol pairs in current vocab
# ─────────────────────────────────────────────────────────────
def get_pair_frequencies(vocab: dict) -> Counter:
    """
    Count how often each adjacent pair of symbols appears.
    ('c','a','t','</w>') → pairs: (c,a), (a,t), (t,</w>)

    The MOST frequent pair is what BPE will merge next.
    This is the entire core algorithm.
    """
    pair_freq = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freq[pair] += freq
    return pair_freq


# ─────────────────────────────────────────────────────────────
# STEP 3: Merge the best pair everywhere in the vocab
# ─────────────────────────────────────────────────────────────
def merge_pair(best_pair: tuple, vocab: dict) -> dict:
    """
    Replace every occurrence of best_pair with merged symbol.
    ('t','h') → 'th' everywhere it appears.
    """
    merged = "".join(best_pair)
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                new_symbols.append(merged)
                i += 2  # skip both symbols, replaced by merged
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


# ─────────────────────────────────────────────────────────────
# STEP 4: Run BPE training for NUM_MERGES steps
# ─────────────────────────────────────────────────────────────
def train_bpe(corpus: str, num_merges: int):
    """
    Full BPE training loop.
    Returns: final vocab, list of merge rules (in order)
    """
    vocab = get_vocab(corpus)
    merge_rules = []

    print("=" * 60)
    print(f"BPE TRAINING — {num_merges} merges")
    print("=" * 60)

    # Show initial state
    all_symbols = set(s for word in vocab for s in word)
    print(f"\nInitial symbol count : {len(all_symbols)}")
    print(f"Initial symbols      : {sorted(all_symbols)}\n")

    for step in range(num_merges):
        pair_freq = get_pair_frequencies(vocab)

        if not pair_freq:
            print("No more pairs to merge. Stopping early.")
            break

        # The most frequent pair is always merged next
        best_pair = pair_freq.most_common(1)[0][0]
        best_count = pair_freq[best_pair]
        merged_symbol = "".join(best_pair)

        merge_rules.append(best_pair)
        vocab = merge_pair(best_pair, vocab)

        print(
            f"Step {step+1:3d} | Merge: {best_pair[0]!r:8s} + {best_pair[1]!r:8s}"
            f" → {merged_symbol!r:12s}  (appeared {best_count}x)"
        )

        # ── EXPERIMENT HOOK ──────────────────────────────────────
        # After each merge, notice:
        # - High frequency pairs get merged early (common subwords)
        # - Later merges are rarer combos
        # - </w> pairs tell you about word endings
        # ─────────────────────────────────────────────────────────

    # Final vocab summary
    final_symbols = set(s for word in vocab for s in word)
    print(f"\nFinal symbol count   : {len(final_symbols)}")
    print(f"Final symbols        : {sorted(final_symbols)}")

    return vocab, merge_rules


# ─────────────────────────────────────────────────────────────
# STEP 5: Tokenize new text using learned merge rules
# ─────────────────────────────────────────────────────────────
def tokenize(text: str, merge_rules: list) -> list:
    """
    Apply learned BPE merge rules to new text.
    Start with characters, apply merges in order.

    KEY INSIGHT: merge rules are applied in the SAME order as training.
    If a pair wasn't frequent enough to be learned, it stays split.
    """
    tokens = []
    for word in text.strip().split():
        symbols = list(word) + ["</w>"]
        for pair in merge_rules:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        tokens.append(symbols)
    return tokens


# ─────────────────────────────────────────────────────────────
# EXPERIMENTS — run after training to build intuition
# ─────────────────────────────────────────────────────────────
def run_experiments(merge_rules: list):
    print("\n" + "=" * 60)
    print("TOKENIZATION EXPERIMENTS")
    print("=" * 60)

    test_cases = [
        # Seen words — should tokenize cleanly
        "cat",
        "rat",
        # Unseen but similar — watch how BPE handles them
        "cats",
        "rats",
        "catfish",
        # Totally new — will stay fragmented
        "blockchain",
        "transformer",
        # Long compound — lots of subword splits
        "tokenization",
        # ── ADD YOUR OWN WORDS HERE and observe ──
    ]

    for word in test_cases:
        result = tokenize(word, merge_rules)
        token_list = result[0] if result else []
        n_tokens = len(token_list)
        print(f"  {word:20s} → {token_list}  ({n_tokens} tokens)")

    print("\n── INSIGHT ─────────────────────────────────────────────")
    print("Words from training corpus → fewer tokens (merged well)")
    print("Unknown/rare words         → more tokens (less merging)")
    print("This is why GPT-4 charges more for non-English text!")
    print("Try: change CORPUS to technical/domain text and retokenize")


# ─────────────────────────────────────────────────────────────
# WHAT TO CHANGE AND WHAT TO WATCH
# ─────────────────────────────────────────────────────────────
def print_experiment_guide():
    print("\n" + "=" * 60)
    print("WHAT TO CHANGE — YOUR EXPERIMENT GUIDE")
    print("=" * 60)
    experiments = [
        ("NUM_MERGES = 5",   "Very fragmented tokens. Even 'cat' may stay as c-a-t"),
        ("NUM_MERGES = 100", "Over-merged. Common words become single tokens"),
        ("Add Kannada text", "Watch token explosion — script not seen in training"),
        ("Add code snippets","def/return/class may become single tokens if frequent"),
        ("Repeat one word 50x","That word's subwords get merged earlier than others"),
    ]
    for change, effect in experiments:
        print(f"\n  Change: {change}")
        print(f"  Effect: {effect}")

if __name__ == "__main__":
    vocab, merge_rules = train_bpe(CORPUS, NUM_MERGES)
    run_experiments(merge_rules)
    print_experiment_guide()

    # Save merge rules for the next script to load
    import json
    with open("merge_rules.json", "w") as f:
        json.dump([list(pair) for pair in merge_rules], f, indent=2)
    print("\n✓ Saved merge_rules.json — used by 02_embedding_explorer.py")