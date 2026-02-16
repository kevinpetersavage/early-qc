import argparse
import os
import gzip
import subprocess
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader

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

def load_model_and_tokenizer(model_name, adapter_name):
    """Loads the tokenizer and the model with the adapter."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading base model: {model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    print(f"Loading adapter: {adapter_name}")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    return model, tokenizer

def run_inference(model, tokenizer, sequences, batch_size=16, update_interval=1000):
    """Runs inference on a list of sequences."""
    test_dataset = SequenceMAPQgt30Dataset(sequences, [0] * len(sequences), tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    low_mapq_count = 0
    high_mapq_count = 0
    total = len(sequences)

    print(f"Running predictions on {total} sequences...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Assuming the model returns logits for 1 label
            logits = outputs.logits.view(-1).cpu().numpy()
            all_predictions.append(logits)

            # Update and show progress
            low_mapq_count += np.sum(logits < 0.5)
            high_mapq_count += np.sum(logits >= 0.5)

            processed = min((i + 1) * batch_size, total)
            # Update output if interval is reached or it's the last batch
            if (processed // update_interval) > ((processed - len(logits)) // update_interval) or processed == total:
                print(f"Processed {processed}/{total} reads. MAPQ < 30: {low_mapq_count}, MAPQ >= 30: {high_mapq_count}", end='\r', flush=True)

    print() # New line after the progress updates
    return np.concatenate(all_predictions)

def save_histogram(predictions, output_plot):
    """Saves a histogram of the predictions."""
    print(f"Saving histogram to {output_plot}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions)
    plt.title("MAPQ Prediction Distribution")
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.savefig(output_plot)

def main():
    parser = argparse.ArgumentParser(description="Classify DNA sequences from a FASTQ file.")
    parser.add_argument("--input", type=str, help="Path to the input FASTQ file.")
    parser.add_argument("--model", type=str, default="InstaDeepAI/nucleotide-transformer-500m-human-ref", help="Base model name.")
    parser.add_argument("--adapter", type=str, default="kpps/mapq", help="Peft adapter name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--update_interval", type=int, default=1000, help="How often to update progress (in reads).")
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

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)

    print(f"Parsing sequences from {input_file}...")
    test_sequences = list(parse_fastq(input_file))
    print(f"Total sequences parsed: {len(test_sequences)}")

    if not test_sequences:
        print("No sequences found in the input file.")
        return

    predictions = run_inference(model, tokenizer, test_sequences, args.batch_size, args.update_interval)

    # Calculate counts
    low_mapq_count = np.sum(predictions < 0.5)
    high_mapq_count = np.sum(predictions >= 0.5)
    total = len(predictions)

    print("\nClassification Summary:")
    print(f"Total sequences: {total}")
    print(f"MAPQ < 30 (predicted < 0.5): {low_mapq_count} ({low_mapq_count/total:.2%})")
    print(f"MAPQ >= 30 (predicted >= 0.5): {high_mapq_count} ({high_mapq_count/total:.2%})")

    # Plotting
    save_histogram(predictions, args.output_plot)
    print("Done!")

if __name__ == "__main__":
    main()
