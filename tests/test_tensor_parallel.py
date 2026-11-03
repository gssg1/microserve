from unittest.mock import patch

import torch
import torch.nn.functional as F

from microserve.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from microserve.layers.linear import (MergedColumnParallelLinear,
                                      QKVParallelLinear, RowParallelLinear)
from microserve.models.qwen3 import Qwen3Attention


def linear_layer(cls, rank, world_size, *args):
    with patch("microserve.layers.linear.get_tensor_model_parallel_rank",
               return_value=rank), patch(
            "microserve.layers.linear.get_tensor_model_parallel_world_size",
            return_value=world_size):
        return cls(*args)


def vocab_layer(cls, rank, world_size, *args):
    with patch("microserve.layers.embed_head.get_tensor_model_parallel_rank",
               return_value=rank), patch(
            "microserve.layers.embed_head.get_tensor_model_parallel_world_size",
            return_value=world_size):
        return cls(*args)


def test_row_parallel():
    torch.manual_seed(0)
    weight = torch.randn(3, 8)
    x = torch.randn(5, 8)
    partials = []
    for rank in range(2):
        layer = linear_layer(RowParallelLinear, rank, 2, 8, 3, False)
        layer.weight_loader(layer.weight, weight)
        local_x = x[:, rank * 4:(rank + 1) * 4]
        partials.append(F.linear(local_x, layer.weight))
    torch.testing.assert_close(sum(partials), F.linear(x, weight))


def test_merged_column_parallel():
    torch.manual_seed(1)
    x = torch.randn(5, 4)
    gate = torch.randn(6, 4)
    up = torch.randn(6, 4)
    gate_parts = []
    up_parts = []
    for rank in range(2):
        layer = linear_layer(MergedColumnParallelLinear, rank, 2, 4, [6, 6], False)
        layer.weight_loader(layer.weight, gate, 0)
        layer.weight_loader(layer.weight, up, 1)
        output = layer(x)
        local_gate, local_up = output.chunk(2, dim=-1)
        gate_parts.append(local_gate)
        up_parts.append(local_up)
    torch.testing.assert_close(torch.cat(gate_parts, dim=-1), F.linear(x, gate))
    torch.testing.assert_close(torch.cat(up_parts, dim=-1), F.linear(x, up))


def test_qkv_parallel():
    torch.manual_seed(2)
    x = torch.randn(5, 4)
    q = torch.randn(8, 4)
    k = torch.randn(4, 4)
    v = torch.randn(4, 4)
    q_parts = []
    k_parts = []
    v_parts = []
    for rank in range(2):
        layer = linear_layer(QKVParallelLinear, rank, 2, 4, 2, 4, 2, False)
        layer.weight_loader(layer.weight, q, "q")
        layer.weight_loader(layer.weight, k, "k")
        layer.weight_loader(layer.weight, v, "v")
        local_q, local_k, local_v = layer(x).split([4, 2, 2], dim=-1)
        q_parts.append(local_q)
        k_parts.append(local_k)
        v_parts.append(local_v)
    torch.testing.assert_close(torch.cat(q_parts, dim=-1), F.linear(x, q))
    torch.testing.assert_close(torch.cat(k_parts, dim=-1), F.linear(x, k))
    torch.testing.assert_close(torch.cat(v_parts, dim=-1), F.linear(x, v))


def test_vocab_parallel_weights():
    torch.manual_seed(3)
    weight = torch.randn(8, 4)
    token_ids = torch.tensor([0, 3, 4, 7])
    embedding_parts = []
    embedding_outputs = []
    head_parts = []
    head_outputs = []
    hidden = torch.randn(3, 4)
    for rank in range(2):
        embedding = vocab_layer(VocabParallelEmbedding, rank, 2, 8, 4)
        embedding.weight_loader(embedding.weight, weight)
        embedding_parts.append(embedding.weight.detach())
        embedding_outputs.append(embedding(token_ids))
        head = vocab_layer(ParallelLMHead, rank, 2, 8, 4, False)
        head.weight_loader(head.weight, weight)
        head_parts.append(head.weight.detach())
        head_outputs.append(head(hidden))
    torch.testing.assert_close(torch.cat(embedding_parts), weight)
    torch.testing.assert_close(torch.cat(head_parts), weight)
    torch.testing.assert_close(sum(embedding_outputs), F.embedding(token_ids, weight))
    torch.testing.assert_close(torch.cat(head_outputs, dim=-1), F.linear(hidden, weight))


def test_attention_uses_global_qkv_sizes():
    with patch(
            "microserve.models.qwen3.get_tensor_model_parallel_world_size",
            return_value=2), patch(
            "microserve.layers.linear.get_tensor_model_parallel_rank",
            return_value=0), patch(
            "microserve.layers.linear.get_tensor_model_parallel_world_size",
            return_value=2):
        attention = Qwen3Attention(
            hidden_size=8,
            num_heads=4,
            num_kv_heads=2,
            max_position=32,
            head_dim=2,
        )
    assert attention.q_size == 4
    assert attention.kv_size == 2
    assert attention.qkv_proj.weight.shape == (8, 8)
    assert attention.o_proj.weight.shape == (8, 4)


def main():
    test_row_parallel()
    test_merged_column_parallel()
    test_qkv_parallel()
    test_vocab_parallel_weights()
    test_attention_uses_global_qkv_sizes()
    print("PASS")


if __name__ == "__main__":
    main()
