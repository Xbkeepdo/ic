import unittest

import torch
from torch import nn

from features.attention_writes import (
    attention_write_source_distribution,
    visual_attention_writes,
)


class AttentionWriteSourceTests(unittest.TestCase):
    def test_visual_writes_reconstruct_attention_update_without_bias(self):
        torch.manual_seed(9)
        layer = nn.Module()
        layer.input_layernorm = nn.LayerNorm(4)
        layer.self_attn = nn.Module()
        layer.self_attn.num_heads = 2
        layer.self_attn.num_key_value_heads = 1
        layer.self_attn.head_dim = 2
        layer.self_attn.v_proj = nn.Linear(4, 2)
        layer.self_attn.o_proj = nn.Linear(4, 4)
        states = torch.randn(1, 3, 4)
        attention = torch.softmax(torch.randn(1, 2, 1, 3), dim=-1)
        capture = {
            "h_prev": states,
            "attn_weights": attention,
            "attention_query_positions": (2,),
        }
        writes = visual_attention_writes(layer, capture, [2], 0, 3)
        values = layer.self_attn.v_proj(layer.input_layernorm(states[0]))
        weighted = (attention[0, :, 0, :, None] * values[None, :, :]).sum(dim=1)
        expected = torch.nn.functional.linear(
            weighted.reshape(1, 4), layer.self_attn.o_proj.weight,
        )
        self.assertTrue(torch.allclose(writes.sum(dim=1), expected, atol=1e-6))

    def test_packed_internvl_value_slot(self):
        torch.manual_seed(10)
        layer = nn.Module()
        layer.attention_norm = nn.LayerNorm(4)
        layer.attention = nn.Module()
        layer.attention.num_heads = 2
        layer.attention.num_key_value_heads = 1
        layer.attention.head_dim = 2
        layer.attention.wqkv = nn.Linear(4, 8)
        layer.attention.wo = nn.Linear(4, 4)
        states = torch.randn(1, 3, 4)
        attention = torch.softmax(torch.randn(1, 2, 1, 3), dim=-1)
        capture = {
            "h_prev": states,
            "attn_weights": attention,
            "attention_query_positions": (2,),
        }
        writes = visual_attention_writes(layer, capture, [2], 0, 3)
        packed = layer.attention.wqkv(layer.attention_norm(states[0]))
        values = packed.view(3, 1, 4, 2)[:, :, -1, :].squeeze(1)
        weighted = (attention[0, :, 0, :, None] * values[None, :, :]).sum(dim=1)
        expected = torch.nn.functional.linear(weighted.reshape(1, 4), layer.attention.wo.weight)
        self.assertTrue(torch.allclose(writes.sum(dim=1), expected, atol=1e-6))

    def test_signed_scalar_projection_and_cosine_temperature(self):
        ffn = torch.tensor([[3.0, 4.0]])
        writes = torch.tensor([[[0.0, 2.0], [1.0, 0.0], [0.0, -3.0]]])
        projection_scores = torch.tensor([[4.0, 3.0, -4.0]])
        pam = attention_write_source_distribution(ffn, writes, "pam")
        cam = attention_write_source_distribution(ffn, writes, "cam")
        self.assertTrue(torch.allclose(pam, projection_scores.softmax(dim=-1)))
        self.assertTrue(torch.allclose(cam, (projection_scores / 5.0 / 0.07).softmax(dim=-1)))
        self.assertTrue(torch.allclose(
            pam,
            attention_write_source_distribution(ffn, writes * 10.0, "pam"),
        ))


if __name__ == "__main__":
    unittest.main()
