import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from mapq_classify import run_inference

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_values = [0.1, 0.9]
        self.call_count = 0

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 1))
        for i in range(batch_size):
            # Alternate between 0.1 and 0.9
            logits[i] = self.logits_values[(self.call_count * batch_size + i) % 2]

        self.call_count += 1
        output = MagicMock()
        output.logits = logits
        return output

    def to(self, device):
        return self

class MockTokenizer:
    def __call__(self, text, **kwargs):
        max_length = kwargs.get('max_length', 30)
        return {
            'input_ids': torch.zeros((1, max_length), dtype=torch.long),
            'attention_mask': torch.ones((1, max_length), dtype=torch.long)
        }

def test_run_inference_progress():
    model = MockModel()
    tokenizer = MockTokenizer()
    sequences = ["A"] * 10

    # batch_size=2, update_interval=3
    # Expected updates:
    # Batch 1 (processed 2): (2//3)=0, (0//3)=0 -> No print
    # Batch 2 (processed 4): (4//3)=1, (2//3)=0 -> PRINT!
    # Batch 3 (processed 6): (6//3)=2, (4//3)=1 -> PRINT!
    # Batch 4 (processed 8): (8//3)=2, (6//3)=2 -> No print
    # Batch 5 (processed 10): last batch or (10//3)=3, (8//3)=2 -> PRINT!

    with patch('builtins.print') as mock_print:
        predictions = run_inference(model, tokenizer, sequences, batch_size=2, update_interval=3)

        assert len(predictions) == 10
        # low_mapq_count should be 5, high_mapq_count should be 5
        low_count = np.sum(predictions < 0.5)
        high_count = np.sum(predictions >= 0.5)
        assert low_count == 5
        assert high_count == 5

        print_calls = [call.args[0] for call in mock_print.call_args_list if call.args and isinstance(call.args[0], str)]

        assert any("Running predictions on 10 sequences" in s for s in print_calls)
        assert any("Processed 4/10" in s for s in print_calls)
        assert any("Processed 6/10" in s for s in print_calls)
        assert any("Processed 10/10" in s for s in print_calls)
        # Verify that "Processed 2/10" and "Processed 8/10" are NOT in print_calls
        assert not any("Processed 2/10" in s for s in print_calls)
        assert not any("Processed 8/10" in s for s in print_calls)
