import argparse
import os
import gzip
import subprocess
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import PeftModel
from torch.utils.data import Dataset

class SequenceMAPQgt30Dataset(Dataset):
    def __init__(self, sequences, mapqs, tokenizer, max_length=30):
        self.sequences = sequences
        self.mapqs = mapqs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        mapq = self.mapqs[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(mapq >= 30, dtype=torch.float)
        }

def parse_fastq(filepath):
    """Parses FASTQ records and yields sequences."""
    # Check if file is gzipped based on extension
    is_gzipped = filepath.endswith('.gz')

    # In the original notebook, it was named .gz but was actually unzipped text
    # We will try to detect if it's actually gzipped.
    try:
        if is_gzipped:
            with gzip.open(filepath, 'rt') as f:
                # Try reading one line to see if it's really gzipped
                f.readline()
            open_func = gzip.open
        else:
            open_func = open
    except Exception:
        # If gzip.open fails or it's not actually gzipped, fallback to regular open
        open_func = open

    with open_func(filepath, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header: # EOF
                break
            sequence = f.readline().strip()
            plus = f.readline().strip()
            quality = f.readline().strip()
            if sequence:
                yield sequence

def download_sample_data(url, output_path, limit_lines=10000):
    """Downloads a portion of a FASTQ file using shell commands."""
    print(f"Downloading sample data from {url}...")
    command = f"curl -s {url} | gunzip | head -{limit_lines} > {output_path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading sample data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Classify DNA sequences from a FASTQ file.")
    parser.add_argument("--input", type=str, help="Path to the input FASTQ file.")
    parser.add_argument("--model", type=str, default="InstaDeepAI/nucleotide-transformer-500m-human-ref", help="Base model name.")
    parser.add_argument("--adapter", type=str, default="kpps/mapq", help="Peft adapter name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--download_sample", action="store_true", help="Download sample data if input is not provided.")
    parser.add_argument("--output_plot", type=str, default="prediction_histogram.png", help="Path to save the prediction histogram.")

    args = parser.parse_args()

    input_file = args.input
    if not input_file and args.download_sample:
        input_file = "sample_data.fastq"
        sample_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/SRR062634_1.filt.fastq.gz"
        download_sample_data(sample_url, input_file)
    elif not input_file:
        print("Please provide an input file using --input or use --download_sample.")
        return

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading base model: {args.model}")
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print(f"Parsing sequences from {input_file}...")
    test_sequences = list(parse_fastq(input_file))
    print(f"Total sequences parsed: {len(test_sequences)}")

    if not test_sequences:
        print("No sequences found in the input file.")
        return

    test_dataset = SequenceMAPQgt30Dataset(test_sequences, [0] * len(test_sequences), tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    print("Running predictions...")
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions

    # Calculate counts
    low_mapq_count = np.sum(predictions < 0.5)
    high_mapq_count = np.sum(predictions >= 0.5)
    total = len(predictions)

    print("\nClassification Summary:")
    print(f"Total sequences: {total}")
    print(f"MAPQ < 30 (predicted < 0.5): {low_mapq_count} ({low_mapq_count/total:.2%})")
    print(f"MAPQ >= 30 (predicted >= 0.5): {high_mapq_count} ({high_mapq_count/total:.2%})")

    # Plotting
    print(f"Saving histogram to {args.output_plot}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions)
    plt.title("MAPQ Prediction Distribution")
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.savefig(args.output_plot)
    print("Done!")

if __name__ == "__main__":
    main()
