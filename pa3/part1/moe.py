"""Mixture-of-Experts: reference, tensor-parallel, and expert-parallel variants.

You will implement `ShardedLinear`, `MoE_TP`, and `MoE_EP` in this file. The
reference `SimpleMoE` and a pre-built `Router` are provided.
"""
import pickle

import numpy as np
from mpi4py import MPI

from mpi_wrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b."""

    def __init__(self, in_features, out_features):
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Two-layer MLP expert with ReLU."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Softmax-gated top-k router (replicated across ranks)."""

    def __init__(self, input_dim, num_experts):
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)
        gates = gates / np.sum(gates, axis=1, keepdims=True)
        return indices, gates


# ---------------------------------------------------------------------------
# Reference implementation: not parallel. Use this to verify correctness.
# ---------------------------------------------------------------------------
class SimpleMoE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
            ]

    def forward(self, x):
        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]
        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.1 — Tensor Parallel MoE.
# ---------------------------------------------------------------------------
class ShardedLinear:
    """Linear layer whose weight is column-sharded across MPI ranks.

    Each rank stores a `(in_features, out_features // world_size)` slice of the
    weight matrix. The forward pass produces the *full* output of shape
    `(batch, out_features)` on every rank, which means a collective is required
    to reassemble the columns each rank computed.

    Requires that `out_features` is evenly divisible by the world size.
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert out_features % self.world_size == 0, (
            f"Output features ({out_features}) must be evenly divisible by "
            f"world size ({self.world_size})"
        )

        self.in_features = in_features
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size
        self.output_offset = self.rank * self.local_out_features

        # Initialize local weights and bias
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global), dtype=np.float32)

        # Compute local partial output: (batch_size, local_out_features)
        local_output = np.dot(x, self.weight) + self.bias

        # Use Allgather to gather all local outputs along the output dimension
        # Each rank sends its (batch_size, local_out_features) slice
        gathered = mpi.allgather(local_output)
        
        # Concatenate along the output dimension to get full (batch_size, out_features_global)
        result = np.concatenate(gathered, axis=1)
        return result


class ShardedExpert:
    """Expert whose weights are sharded along the hidden / output dim."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        with rng_context("expert"):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)
        return self.fc2(hidden)


class MoE_TP:
    """Mixture-of-Experts with tensor-parallel experts.

    Every rank holds a slice of every expert. Routing is replicated. After
    each expert's forward pass, ranks need a collective to reassemble the
    full output of that expert before applying the gate.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        with rng_context("router"):
            self.router = Router(input_dim, num_experts)

        with rng_context("expert"):
            self.experts = [
                ShardedExpert(input_dim, hidden_dim, output_dim)
                for _ in range(num_experts)
            ]

        if self.rank == 0:
            print(
                f"[MoE_TP] world_size={self.world_size}, num_experts={num_experts}, topk={self.topk}"
            )

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        # Get routing indices and gates from self.router(x, self.topk).
        indices, gates = self.router(x, self.topk)

        # For each token and each expert it was routed to, run it through the expert
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]  # shape (1, input_dim)
                expert_output = self.experts[expert_idx](item)  # shape (1, output_dim)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


# ---------------------------------------------------------------------------
# Part 1.2 — Expert Parallel MoE.
# ---------------------------------------------------------------------------
class MoE_EP:
    """Mixture-of-Experts with expert-parallel experts.

    Each rank owns *exactly one* expert. After routing, tokens that have been
    assigned to expert `e` must be sent to the rank that owns expert `e`. The
    expert computes its forward pass on the tokens it received and the results
    are sent back to the originating ranks.

    The natural collective for this pattern is **all-to-all**: each rank
    builds `world_size` buckets (one per destination rank) and exchanges them.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # == world size
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.Get_rank()
        self.world_size = mpi.Get_size()

        assert num_experts == self.world_size, (
            "MoE_EP assumes one expert per rank; got "
            f"num_experts={num_experts}, world_size={self.world_size}"
        )

        with rng_context("router"):
            self.router = Router(input_dim, self.num_experts)

        # Each rank initializes its own expert independently — we want the
        # experts to be different, so this rng is rank-specific.
        with rng_context("expert_with_rank"):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

        # Bonus path (+5): route the EP all-to-all through the point-to-point
        # `myAlltoall` copied from PA2 instead of the pickle-based collective.
        # Set False to fall back to `mpi.alltoall`.
        self.use_my_alltoall = True

    def _alltoall(self, send_buckets):
        """All-to-all of one Python payload per destination rank.

        Bonus path: when `use_my_alltoall` is set, the exchange is routed
        through the PA2 point-to-point `myAlltoall`, which requires equal-sized
        segments. We pickle each destination's payload to bytes, prefix a
        4-byte length header, and zero-pad every per-rank segment to a common
        length agreed via an all-reduce(MAX). The receiver reads the header and
        unpickles only the valid bytes, so the result is identical to
        `mpi.alltoall(send_buckets)`.
        """
        if not getattr(self, "use_my_alltoall", False):
            return mpi.alltoall(send_buckets)

        ws = self.world_size
        payloads = [pickle.dumps(send_buckets[p]) for p in range(ws)]
        local_max = max(len(p) for p in payloads)
        cap = int(mpi.allreduce(local_max, op=MPI.MAX))  # common payload capacity
        seg = 4 + cap  # 4-byte length header + zero-padded payload

        send = np.zeros(ws * seg, dtype=np.uint8)
        for p in range(ws):
            base = p * seg
            n = len(payloads[p])
            send[base : base + 4] = np.frombuffer(np.uint32(n).tobytes(), dtype=np.uint8)
            send[base + 4 : base + 4 + n] = np.frombuffer(payloads[p], dtype=np.uint8)

        recv = np.empty_like(send)
        mpi.myAlltoall(send, recv)

        received = []
        for p in range(ws):
            base = p * seg
            n = int(np.frombuffer(recv[base : base + 4].tobytes(), dtype=np.uint32)[0])
            received.append(pickle.loads(recv[base + 4 : base + 4 + n].tobytes()))
        return received

    def forward(self, x):
        """
        Args:
            x: `(batch_size, input_dim)` — replicated on every rank.

        Returns:
            `(batch_size, output_dim)` — replicated on every rank.
        """
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))

        # Get routing indices and gates from self.router(x, self.topk).
        indices, gates = self.router(x, self.topk)

        # Build send buckets: buckets[dest_rank] = list of (token_idx, item, gate, k)
        # for tokens destined to that rank
        buckets = [[] for _ in range(self.world_size)]
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i : i + 1]  # shape (1, input_dim)
                buckets[expert_idx].append((i, item, gate, k))

        # Use all-to-all to send buckets to their destination ranks and receive buckets from others
        received_buckets = self._alltoall(buckets)

        # Run this rank's expert on all received tokens and store results
        # received_buckets[src_rank] = list of (token_idx, item, gate, k)
        token_results = {}  # (src_rank, token_idx, k) -> output
        for src_rank, bucket in enumerate(received_buckets):
            for token_idx, item, gate, k in bucket:
                expert_output = self.expert(item)  # shape (1, output_dim)
                token_results[(src_rank, token_idx, k)] = expert_output[0]

        # Send results back: results[dest_rank] = list of (token_idx, k, output)
        result_buckets = [[] for _ in range(self.world_size)]
        for (src_rank, token_idx, k), output in token_results.items():
            result_buckets[src_rank].append((token_idx, k, output))

        # Use all-to-all again to send results back
        received_results = self._alltoall(result_buckets)

        # Reconstruct outputs by combining results from all experts
        for expert_rank, results in enumerate(received_results):
            for token_idx, k, output in results:
                # Retrieve the gate weight that was used for this expert
                expert_idx = indices[token_idx, k]
                gate = gates[token_idx, k]
                outputs[token_idx] += gate * output

        return outputs

    def __call__(self, x):
        return self.forward(x)
