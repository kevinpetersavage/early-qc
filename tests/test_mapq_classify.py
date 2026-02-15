import os
import gzip
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

# Use Agg backend for matplotlib to avoid issues in headless environments
import matplotlib
matplotlib.use('Agg')

from mapq_classify import parse_fastq, SequenceMAPQgt30Dataset

def test_parse_fastq_plain(tmp_path):
    fastq_content = "@seq1\nACGT\n+\n!!!!\n@seq2\nTGCA\n+\n????\n"
    fastq_file = tmp_path / "test.fastq"
    fastq_file.write_text(fastq_content)

    sequences = list(parse_fastq(str(fastq_file)))
    assert sequences == ["ACGT", "TGCA"]

def test_parse_fastq_gz(tmp_path):
    fastq_content = "@seq1\nACGT\n+\n!!!!\n@seq2\nTGCA\n+\n????\n"
    fastq_file = tmp_path / "test.fastq.gz"
    with gzip.open(fastq_file, "wt") as f:
        f.write(fastq_content)

    sequences = list(parse_fastq(str(fastq_file)))
    assert sequences == ["ACGT", "TGCA"]

def test_parse_fastq_empty(tmp_path):
    fastq_file = tmp_path / "empty.fastq"
    fastq_file.write_text("")

    sequences = list(parse_fastq(str(fastq_file)))
    assert sequences == []

class MockTokenizer:
    def __call__(self, text, **kwargs):
        # Return something that looks like what tokenizer returns
        max_length = kwargs.get('max_length', 30)
        return {
            'input_ids': torch.zeros((1, max_length), dtype=torch.long),
            'attention_mask': torch.ones((1, max_length), dtype=torch.long)
        }

def test_dataset():
    sequences = ["ACGT", "TGCA"]
    mapqs = [20, 40]
    tokenizer = MockTokenizer()
    dataset = SequenceMAPQgt30Dataset(sequences, mapqs, tokenizer, max_length=10)

    assert len(dataset) == 2

    item = dataset[0]
    assert item['input_ids'].shape == (10,)
    assert item['attention_mask'].shape == (10,)
    assert item['labels'] == torch.tensor(0.0)

    item = dataset[1]
    assert item['labels'] == torch.tensor(1.0)

def test_dataset_item_types():
    sequences = ["A"]
    mapqs = [35]
    tokenizer = MockTokenizer()
    dataset = SequenceMAPQgt30Dataset(sequences, mapqs, tokenizer)

    item = dataset[0]
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    assert isinstance(item['labels'], torch.Tensor)
    assert item['labels'].dtype == torch.float

def test_save_histogram(tmp_path):
    output_plot = tmp_path / "test_plot.png"
    predictions = np.array([0.1, 0.2, 0.8, 0.9])
    from mapq_classify import save_histogram
    save_histogram(predictions, str(output_plot))
    assert output_plot.exists()
