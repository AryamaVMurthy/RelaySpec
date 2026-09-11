import unittest
import torch
from experiments.auf_vllm.blocks import make_block, collate_blocks


class BlockTests(unittest.TestCase):
    def test_no_future_features_or_labels_in_input(self):
        features = torch.arange(24).view(8,3).float()
        block = make_block(features, list(range(8)), 3, 4, 99)
        self.assertTrue(torch.equal(block.context, features[:3]))
        self.assertEqual(block.noise_ids.tolist(), [3,99,99,99])
        self.assertEqual(block.labels.tolist(), [-100,4,5,6])
        self.assertEqual(block.valid.tolist(), [False,True,True,True])

    def test_eos_supervised_but_later_tokens_excluded(self):
        block = make_block(torch.randn(8,3), list(range(8)), 3, 5, 99, eos_ids=(5,))
        self.assertEqual(block.labels.tolist(), [-100,4,5,-100,-100])
        self.assertEqual(block.valid.tolist(), [False,True,True,False,False])
        with self.assertRaises(ValueError):
            make_block(torch.randn(8,3), list(range(8)), 5, 5, 99, eos_ids=(5,))

    def test_padding_positions_and_attention(self):
        features = torch.randn(8,3)
        batch = collate_blocks([make_block(features,list(range(8)),a,4,99) for a in [2,6]], "cpu")
        self.assertEqual(batch["position_ids"][0].tolist(), [0,1,0,0,0,0,2,3,4,5])
        self.assertEqual(batch["attention_mask"][0,0,0].tolist(), [True,True,False,False,False,False,True,True,True,True])
        self.assertEqual(batch["valid"][1].tolist(), [False,True,False,False])
        self.assertTrue(torch.equal(batch["context"][0,2:], torch.zeros(4,3)))


if __name__ == "__main__":
    unittest.main()
