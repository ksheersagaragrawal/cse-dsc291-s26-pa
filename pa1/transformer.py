from typing import Callable, Tuple, List, Dict
import argparse

import math
import numpy as np
import torch

import auto_diff as ad

# ============================================================
# DATA & TOKENIZER (provided — do not modify)
# ============================================================
#
# We use a simple **word-level tokenizer**.  Every sentence is split on
# whitespace and each unique word is assigned an integer index.  This is
# conceptually the same as what real LLM tokenizers (BPE, SentencePiece,
# etc.) do, but far simpler: our "sub-word vocabulary" is just whole
# words.  A special PAD token (index 0) exists for use during generation
# but never appears in the training data.
#
# All 10 training sentences contain exactly NUM_WORDS words, so after
# the next-token shift (input = tokens[:-1], target = tokens[1:]) every
# sequence has the same length and no padding is required.

SENTENCES = [
    "attention is all you need",       # All sentences have exactly 5 words.
    "the model learns very fast",
    "we will overfit this data",
    "gradients flow through the network",
    "tokens predict the next word",
    "i love deep learning systems",
    "neural nets are really cool",
    "train the model on data",
    "good models learn from examples",
    "loss goes down every epoch",
]

NUM_WORDS = 5  # every sentence has exactly this many words

# Build word-level vocabulary: PAD = 0, then sorted unique words starting at 1
_all_words = sorted(set(w for s in SENTENCES for w in s.split()))
WORD_TO_IDX: Dict[str, int] = {"<pad>": 0}
for _i, _w in enumerate(_all_words, start=1):
    WORD_TO_IDX[_w] = _i
IDX_TO_WORD: Dict[int, str] = {v: k for k, v in WORD_TO_IDX.items()}
VOCAB_SIZE = len(WORD_TO_IDX)  # 46 (45 words + 1 PAD)

# ============================================================
# HYPERPARAMETERS (provided — you may tune these)
# ============================================================

MODEL_DIM = 64
FF_DIM = 128
SEQ_LEN = NUM_WORDS - 1  # input/target length after shifting (4)
EPS = 1e-5
LR = 0.05
NUM_EPOCHS = 500
BATCH_SIZE = 10     # all 10 sentences in one batch (full-batch SGD)


def encode(sentence: str) -> List[int]:
    """Encode a sentence into a list of word-level token indices.

    Example
    -------
    >>> encode("attention is all you need")
    [3, 18, 1, 45, 26]
    """
    return [WORD_TO_IDX[w] for w in sentence.split()]


def decode(indices: List[int]) -> str:
    """Decode token indices back to a space-separated string.

    Stops at the first PAD token (index 0) if present.
    """
    words = []
    for idx in indices:
        if idx == 0:
            break
        words.append(IDX_TO_WORD.get(idx, "<unk>"))
    return " ".join(words)


def prepare_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input/target pairs for next-token prediction.

    Every sentence has exactly NUM_WORDS words.  After encoding:
      - input  x = tokens[:-1]   (first NUM_WORDS-1 tokens)
      - target y = tokens[1:]    (last  NUM_WORDS-1 tokens)
    No padding is needed because all sentences are the same length.

    Returns
    -------
    X : torch.Tensor of shape (num_sentences, SEQ_LEN), dtype long
    Y : torch.Tensor of shape (num_sentences, SEQ_LEN), dtype long
    """
    X_list, Y_list = [], []
    for s in SENTENCES:
        words = s.split()
        assert len(words) == NUM_WORDS, f"Sentence must have {NUM_WORDS} words: {s!r}"
        tokens = encode(s)
        X_list.append(tokens[:-1])
        Y_list.append(tokens[1:])
    return torch.tensor(X_list, dtype=torch.long), torch.tensor(Y_list, dtype=torch.long)


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer indices to one-hot float64 vectors.

    Parameters
    ----------
    indices : torch.Tensor of shape (...) with integer values
    num_classes : int

    Returns
    -------
    torch.Tensor of shape (..., num_classes), dtype float64
    """
    flat = indices.reshape(-1)
    oh = torch.zeros(flat.shape[0], num_classes, dtype=torch.float64)
    oh.scatter_(1, flat.unsqueeze(1), 1.0)
    return oh.reshape(*indices.shape, num_classes)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (autoregressive) attention mask.

    Returns a (seq_len, seq_len) tensor where:
      - allowed positions (i >= j) have value 0.0
      - masked positions (i < j, i.e. future tokens) have value -1e9

    This mask is added to attention scores before softmax so that
    each position can only attend to itself and earlier positions.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.float64)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i][j] = -1e9
    return mask


# ============================================================
# PART 1: Causal Self-Attention (10 pt)
# ============================================================

def causal_self_attention(
    X: ad.Node,
    W_Q: ad.Node,
    W_K: ad.Node,
    W_V: ad.Node,
    W_O: ad.Node,
    mask: ad.Node,
    model_dim: int,
) -> ad.Node:
    """Single-head causal self-attention.

    Parameters
    ----------
    X : ad.Node
        Input tensor of shape (batch, seq_len, model_dim).
    W_Q, W_K, W_V : ad.Node
        Projection weight matrices, each of shape (model_dim, model_dim).
    W_O : ad.Node
        Output projection weight matrix of shape (model_dim, model_dim).
    mask : ad.Node
        Causal mask of shape (batch, seq_len, seq_len).
        Contains 0 for allowed positions and -1e9 for masked (future) positions.
    model_dim : int
        Model dimension (used for scaling: 1/sqrt(model_dim)).

    Returns
    -------
    output : ad.Node
        Shape (batch, seq_len, model_dim).

    Steps
    -----
    1. Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
    2. scores = (Q @ K^T) / sqrt(model_dim)
    3. scores = scores + mask
    4. attn_weights = softmax(scores, dim=-1)
    5. attn_output = attn_weights @ V
    6. output = attn_output @ W_O

    Hints
    -----
    - Use ad.matmul for matrix multiplication.
    - Use ad.transpose(node, dim0, dim1) to transpose K.
      For a 3-D tensor (batch, seq_len, model_dim), swap dims 1 and 2.
    - Use ad.softmax(node, dim=-1).
    - Divide by sqrt(model_dim) using the / operator (divides by a constant).
    """
    Q = ad.matmul(X, W_Q)
    K = ad.matmul(X, W_K)
    V = ad.matmul(X, W_V)
    k_t = ad.transpose(K, 1, 2)
    scores = ad.matmul(Q, k_t) / math.sqrt(model_dim)
    scores = scores + mask
    attn = ad.softmax(scores, dim=-1)
    attn_out = ad.matmul(attn, V)
    return ad.matmul(attn_out, W_O)


# ============================================================
# PART 2: Decoder Layer (5 pt)
# ============================================================

def decoder_layer(
    X: ad.Node,
    W_Q: ad.Node,
    W_K: ad.Node,
    W_V: ad.Node,
    W_O: ad.Node,
    W_ff1: ad.Node,
    W_ff2: ad.Node,
    mask: ad.Node,
    model_dim: int,
    ff_dim: int,
    eps: float,
) -> ad.Node:
    """One transformer decoder layer with residual connections and layer norm.

    Parameters
    ----------
    X : ad.Node
        Input of shape (batch, seq_len, model_dim).
    W_Q, W_K, W_V, W_O : ad.Node
        Attention weight matrices, each (model_dim, model_dim).
    W_ff1 : ad.Node
        First feed-forward weight matrix of shape (model_dim, ff_dim).
    W_ff2 : ad.Node
        Second feed-forward weight matrix of shape (ff_dim, model_dim).
    mask : ad.Node
        Causal mask of shape (batch, seq_len, seq_len).
    model_dim, ff_dim : int
        Dimensions for the model and feed-forward layer.
    eps : float
        Epsilon for layer normalization.

    Returns
    -------
    output : ad.Node
        Shape (batch, seq_len, model_dim).

    Steps
    -----
    1. attn_out = causal_self_attention(X, W_Q, W_K, W_V, W_O, mask, model_dim)
    2. h = layernorm(X + attn_out, normalized_shape=[model_dim], eps=eps)
    3. ff_out = relu(h @ W_ff1) @ W_ff2
    4. output = layernorm(h + ff_out, normalized_shape=[model_dim], eps=eps)

    Hints
    -----
    - Use ad.layernorm(node, normalized_shape=[model_dim], eps=eps).
    - Use ad.relu(node) for the ReLU activation.
    - Use ad.matmul for matrix multiplication.
    - The + operator on nodes performs element-wise addition (residual connection).
    """
    attn_out = causal_self_attention(X, W_Q, W_K, W_V, W_O, mask, model_dim)
    h = ad.layernorm(X + attn_out, normalized_shape=[model_dim], eps=eps)
    ff = ad.matmul(ad.relu(ad.matmul(h, W_ff1)), W_ff2)
    return ad.layernorm(h + ff, normalized_shape=[model_dim], eps=eps)


# ============================================================
# PART 3: Transformer LM Forward Pass (5 pt) + Cross-Entropy Loss (5 pt)
# ============================================================

def transformer_lm(
    X_onehot: ad.Node,
    W_embed: ad.Node,
    pos_embed: ad.Node,
    W_Q: ad.Node,
    W_K: ad.Node,
    W_V: ad.Node,
    W_O: ad.Node,
    W_ff1: ad.Node,
    W_ff2: ad.Node,
    W_head: ad.Node,
    mask: ad.Node,
    model_dim: int,
    ff_dim: int,
    eps: float,
) -> ad.Node:
    """Full transformer language model forward pass.

    Parameters
    ----------
    X_onehot : ad.Node
        One-hot encoded input tokens of shape (batch, seq_len, vocab_size).
    W_embed : ad.Node
        Token embedding matrix of shape (vocab_size, model_dim).
    pos_embed : ad.Node
        Position embeddings, pre-tiled to shape (batch, seq_len, model_dim).
    W_Q, W_K, W_V, W_O : ad.Node
        Attention weight matrices.
    W_ff1, W_ff2 : ad.Node
        Feed-forward weight matrices.
    W_head : ad.Node
        Output projection matrix of shape (model_dim, vocab_size).
    mask : ad.Node
        Causal mask, pre-tiled to shape (batch, seq_len, seq_len).
    model_dim, ff_dim : int
    eps : float

    Returns
    -------
    logits : ad.Node
        Shape (batch, seq_len, vocab_size) — unnormalized log-probabilities
        for the next token at each position.

    Steps
    -----
    1. token_emb = X_onehot @ W_embed        → (batch, seq_len, model_dim)
    2. h = token_emb + pos_embed             (element-wise, same shape)
    3. h = decoder_layer(h, W_Q, ..., mask, model_dim, ff_dim, eps)
    4. logits = h @ W_head                   → (batch, seq_len, vocab_size)
    """
    emb = ad.matmul(X_onehot, W_embed)
    h = emb + pos_embed
    h = decoder_layer(
        h, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, mask, model_dim, ff_dim, eps
    )
    return ad.matmul(h, W_head)


def cross_entropy_loss(
    logits: ad.Node,
    targets_onehot: ad.Node,
    num_tokens: int,
) -> ad.Node:
    """Compute average cross-entropy loss for next-token prediction.

    Parameters
    ----------
    logits : ad.Node
        Model output of shape (batch, seq_len, vocab_size).
    targets_onehot : ad.Node
        One-hot encoded target tokens of shape (batch, seq_len, vocab_size).
    num_tokens : int
        Total number of target tokens (= BATCH_SIZE * SEQ_LEN, for averaging).

    Returns
    -------
    loss : ad.Node
        Average cross-entropy loss.

    Steps
    -----
    1. probs = softmax(logits, dim=-1)
    2. log_probs = log(probs)
    3. loss = -sum(targets_onehot * log_probs) / num_tokens

    Hints
    -----
    - Use ad.softmax(node, dim=-1).
    - Use ad.log(node). To avoid log(0), you may add a small constant:
      ad.log(probs + 1e-10)  (use ad.add_by_const).
    - Use ad.sum_op(node, dim=(0, 1, 2), keepdim=True) to sum over all dimensions.
    - Multiply by -1 and divide by num_tokens using the * and / operators.

    Note
    ----
    You do NOT need to implement a numerically stable version (log-sum-exp trick).
    The simple softmax → log approach is fine for this assignment.
    """
    probs = ad.softmax(logits, dim=-1)
    logp = ad.log(ad.add_by_const(probs, 1e-10))
    ce = ad.mul(targets_onehot, logp)
    s = ad.sum_op(ce, dim=(0, 1, 2), keepdim=True)
    return ad.div_by_const(ad.mul_by_const(s, -1.0), float(num_tokens))


# ============================================================
# PART 4: Training & Generation (15 pt)
# ============================================================

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    Y: torch.Tensor,
    model_weights: List[torch.Tensor],
    lr: float,
) -> Tuple[List[torch.Tensor], float]:
    """Run one epoch of SGD training.

    Since we have only 10 sentences and BATCH_SIZE=10, each epoch is a
    single gradient step over all data (full-batch gradient descent).

    Parameters
    ----------
    f_run_model : Callable
        Function that takes (X_onehot, Y_onehot, pos_embed_tiled, mask_tiled,
        model_weights) and returns a list:
        [logits, loss, grad_W_embed, grad_pos, grad_W_Q, grad_W_K, grad_W_V,
         grad_W_O, grad_W_ff1, grad_W_ff2, grad_W_head].
    X : torch.Tensor
        Input token indices of shape (num_sentences, SEQ_LEN), dtype long.
    Y : torch.Tensor
        Target token indices of shape (num_sentences, SEQ_LEN), dtype long.
    model_weights : List[torch.Tensor]
        [W_embed, W_pos, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, W_head]
    lr : float
        Learning rate.

    Returns
    -------
    model_weights : List[torch.Tensor]
        Updated weights after this epoch.
    loss_val : float
        The training loss for this epoch.
    """
    batch = X.shape[0]

    # --- Data preparation (provided) ---
    X_oh = one_hot(X, VOCAB_SIZE)   # (batch, SEQ_LEN, VOCAB_SIZE)
    Y_oh = one_hot(Y, VOCAB_SIZE)   # (batch, SEQ_LEN, VOCAB_SIZE)

    W_pos = model_weights[1]
    pos_tiled = W_pos.unsqueeze(0).expand(batch, SEQ_LEN, MODEL_DIM).clone()
    mask_tiled = create_causal_mask(SEQ_LEN).unsqueeze(0).expand(batch, SEQ_LEN, SEQ_LEN).clone()

    # --- Forward + backward (provided) ---
    result = f_run_model(X_oh, Y_oh, pos_tiled, mask_tiled, model_weights)
    loss_val = result[1]
    grads = result[2:]  # one gradient per weight, in the same order as model_weights

    # --- TODO: Update weights (your code here) ---
    # grads[i] corresponds to model_weights[i].
    # Each gradient has an extra leading batch dimension (dim 0).
    # You must sum over dim 0 before subtracting:
    #     new_W = W - lr * grad.sum(dim=0)
    # Return the updated model_weights list and float(loss_val).
    new_weights: List[torch.Tensor] = []
    for w, g in zip(model_weights, grads):
        new_weights.append((w - lr * g.sum(dim=0)).clone())
    lv = float(loss_val.item()) if isinstance(loss_val, torch.Tensor) else float(loss_val)
    return new_weights, lv


def generate(
    prompt: str,
    model_weights: List[torch.Tensor],
    max_new_tokens: int = 10,
) -> str:
    """Generate text autoregressively using greedy decoding.

    This function performs full forward passes (no KV cache).
    At each step, the entire sequence is fed through the model,
    and only the logits at the last valid position are used to
    predict the next token (word).

    Parameters
    ----------
    prompt : str
        The initial text (space-separated words) to continue from.
        Example: "attention is"
    model_weights : List[torch.Tensor]
        [W_embed, W_pos, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, W_head]
    max_new_tokens : int
        Maximum number of new tokens (words) to generate.

    Returns
    -------
    generated_text : str
        The prompt followed by the generated continuation (space-separated).
    """
    W_embed, W_pos, W_Q, W_K, W_V, W_O, W_ff1, W_ff2, W_head = model_weights

    # --- Graph construction for inference (provided) ---
    X_var = ad.Variable("X_gen")
    W_embed_var = ad.Variable("We_gen")
    pos_var = ad.Variable("pos_gen")
    W_Q_var = ad.Variable("Wq_gen")
    W_K_var = ad.Variable("Wk_gen")
    W_V_var = ad.Variable("Wv_gen")
    W_O_var = ad.Variable("Wo_gen")
    W_ff1_var = ad.Variable("Wf1_gen")
    W_ff2_var = ad.Variable("Wf2_gen")
    W_head_var = ad.Variable("Wh_gen")
    mask_var = ad.Variable("mask_gen")

    logits_node = transformer_lm(
        X_var, W_embed_var, pos_var,
        W_Q_var, W_K_var, W_V_var, W_O_var,
        W_ff1_var, W_ff2_var, W_head_var,
        mask_var, MODEL_DIM, FF_DIM, EPS,
    )
    gen_evaluator = ad.Evaluator([logits_node])

    # Pre-compute mask and pos for batch_size=1 (provided)
    mask_1 = create_causal_mask(SEQ_LEN).unsqueeze(0)   # (1, SEQ_LEN, SEQ_LEN)
    pos_1 = W_pos.unsqueeze(0)                           # (1, SEQ_LEN, MODEL_DIM)

    def run_forward(token_ids: List[int]) -> torch.Tensor:
        """Run the model on a single sequence padded to SEQ_LEN.
        Returns logits of shape (1, SEQ_LEN, VOCAB_SIZE)."""
        padded = (token_ids + [0] * SEQ_LEN)[:SEQ_LEN]
        X_oh = one_hot(torch.tensor([padded], dtype=torch.long), VOCAB_SIZE)
        result = gen_evaluator.run({
            X_var: X_oh, W_embed_var: W_embed, pos_var: pos_1,
            W_Q_var: W_Q, W_K_var: W_K, W_V_var: W_V, W_O_var: W_O,
            W_ff1_var: W_ff1, W_ff2_var: W_ff2, W_head_var: W_head,
            mask_var: mask_1,
        })
        return result[0]

    # --- TODO: Generation loop (your code here) ---
    # 1. Encode the prompt into token indices using encode().
    # 2. Loop up to max_new_tokens times:
    #    a. Call run_forward(tokens) to get logits (1, SEQ_LEN, VOCAB_SIZE).
    #    b. Read the logits at position len(tokens)-1, take argmax → next_token.
    #    c. If next_token == 0 (PAD) or len(tokens) > SEQ_LEN, stop.
    #    d. Append next_token to the token list.
    # 3. Return decode(tokens).
    tokens = list(encode(prompt))
    for _ in range(max_new_tokens):
        if len(tokens) > SEQ_LEN or len(tokens) >= NUM_WORDS:
            break
        logits = run_forward(tokens)
        pos = min(len(tokens), SEQ_LEN) - 1
        next_t = int(logits[0, pos].argmax().item())
        if next_t == 0:
            break
        tokens.append(next_t)
    return decode(tokens)


def save_weights(model_weights: List[torch.Tensor], path: str = "weights.pt") -> None:
    """Save trained model weights to disk."""
    torch.save(model_weights, path)


def load_weights(path: str = "weights.pt") -> List[torch.Tensor]:
    """Load trained model weights from disk."""
    return torch.load(path, map_location="cpu")


# ============================================================
# TRAIN MODEL (provided — do not modify)
# ============================================================

def train_model() -> List[torch.Tensor]:
    """Train the transformer LM to overfit 10 sentences.

    Returns
    -------
    model_weights : List[torch.Tensor]
        The trained model weights.
    """

    # --- Load data ---
    X, Y = prepare_data()
    num_tokens = BATCH_SIZE * SEQ_LEN

    # --- Define computational graph ---
    X_var = ad.Variable("X")
    Y_var = ad.Variable("Y")
    pos_var = ad.Variable("pos_embed")
    mask_var = ad.Variable("mask")

    W_embed_var = ad.Variable("W_embed")
    W_Q_var = ad.Variable("W_Q")
    W_K_var = ad.Variable("W_K")
    W_V_var = ad.Variable("W_V")
    W_O_var = ad.Variable("W_O")
    W_ff1_var = ad.Variable("W_ff1")
    W_ff2_var = ad.Variable("W_ff2")
    W_head_var = ad.Variable("W_head")

    logits = transformer_lm(
        X_var, W_embed_var, pos_var,
        W_Q_var, W_K_var, W_V_var, W_O_var,
        W_ff1_var, W_ff2_var, W_head_var,
        mask_var, MODEL_DIM, FF_DIM, EPS,
    )
    loss = cross_entropy_loss(logits, Y_var, num_tokens)

    weight_vars = [
        W_embed_var, pos_var,
        W_Q_var, W_K_var, W_V_var, W_O_var,
        W_ff1_var, W_ff2_var, W_head_var,
    ]
    grads = ad.gradients(loss, weight_vars)
    evaluator = ad.Evaluator([logits, loss, *grads])

    # --- Initialize weights ---
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(MODEL_DIM)
    W_embed_val = torch.tensor(np.random.uniform(-stdv, stdv, (VOCAB_SIZE, MODEL_DIM)))
    W_pos_val   = torch.tensor(np.random.uniform(-stdv, stdv, (SEQ_LEN, MODEL_DIM)))
    W_Q_val     = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, MODEL_DIM)))
    W_K_val     = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, MODEL_DIM)))
    W_V_val     = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, MODEL_DIM)))
    W_O_val     = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, MODEL_DIM)))
    W_ff1_val   = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, FF_DIM)))
    W_ff2_val   = torch.tensor(np.random.uniform(-stdv, stdv, (FF_DIM, MODEL_DIM)))
    W_head_val  = torch.tensor(np.random.uniform(-stdv, stdv, (MODEL_DIM, VOCAB_SIZE)))

    model_weights = [
        W_embed_val, W_pos_val, W_Q_val, W_K_val, W_V_val,
        W_O_val, W_ff1_val, W_ff2_val, W_head_val,
    ]

    def f_run_model(X_oh, Y_oh, pos_tiled, mask_tiled, weights):
        """Run forward + backward and return [logits, loss, *gradients]."""
        W_e, W_p, W_q, W_k, W_v, W_o, W_f1, W_f2, W_h = weights
        return evaluator.run({
            X_var: X_oh, Y_var: Y_oh,
            pos_var: pos_tiled, mask_var: mask_tiled,
            W_embed_var: W_e, W_Q_var: W_q, W_K_var: W_k,
            W_V_var: W_v, W_O_var: W_o,
            W_ff1_var: W_f1, W_ff2_var: W_f2, W_head_var: W_h,
        })

    # --- Training loop ---
    for epoch in range(NUM_EPOCHS):
        model_weights, loss_val = sgd_epoch(
            f_run_model, X, Y, model_weights, LR
        )
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS}:  loss = {loss_val:.4f}")

    # --- Test overfitting ---
    print("\n--- Generation test ---")
    num_correct = 0
    for s in SENTENCES:
        prompt = " ".join(s.split()[:2])
        generated = generate(prompt, model_weights, max_new_tokens=NUM_WORDS)
        gen_words = generated.split()[:NUM_WORDS]
        ref_words = s.split()
        match = gen_words == ref_words
        num_correct += int(match)
        status = "OK" if match else "FAIL"
        print(f"  [{status}] prompt='{prompt}' -> '{generated}'")
    print(f"\nOverfit accuracy: {num_correct}/{len(SENTENCES)}")

    return model_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PA1 transformer LM.")
    parser.add_argument(
        "--playground",
        action="store_true",
        help="Launch the interactive playground after training.",
    )
    args = parser.parse_args()

    weights = train_model()
    save_weights(weights)
    print("\nSaved trained weights to weights.pt")

    if args.playground:
        from library.cli import playground

        playground(generate, weights, WORD_TO_IDX)
