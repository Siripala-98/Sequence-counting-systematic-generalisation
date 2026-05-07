"""
Training and evaluation script for Dyck-1 sequence classification experiments.

Trains a specified model on Dyck-1 bracket sequences and evaluates it on
short (in-distribution) and long (out-of-distribution) test sets.
Long-sequence generalisation is the key metric — sequences are twice the
length of anything seen during training.

Usage:
    python train.py --model StackLSTM_DFRC --dataset large --hidden 2 --epochs 30
    python train.py --model VanillaLSTM --dataset large --hidden 2 --epochs 30
    python train.py --model StackRNN --dataset small --hidden 3 --epochs 10000

References:
    El-Naggar, N. (2023). What Really Counts: Theoretical and Practical Aspects
    of Counting Behaviour in Simple RNNs.
    https://github.com/nadineelnaggar/DNNC
"""

import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from models_scm_clf_1_cleaned import (
    VanillaRNN, VanillaGRU, VanillaLSTM,
    StackRNN, StackGRU, StackLSTM,
    StackRNN_DFRC, StackGRU_DFRC, StackLSTM_DFRC,
    RecurrentStack, RecurrentDFRC,
)

# ---------------------------------------------------------------------------
# Model family sets — used throughout to route forward pass logic cleanly
# ---------------------------------------------------------------------------

# Models that have a counter module (need stack.reset() between sequences)
COUNTER_MODELS = {
    'StackRNN', 'StackGRU', 'StackLSTM',
    'StackRNN_DFRC', 'StackGRU_DFRC', 'StackLSTM_DFRC',
    'RecurrentStack', 'RecurrentDFRC',
}

# Models that use the DFRC counter (2-element stack state)
DFRC_MODELS = {
    'StackRNN_DFRC', 'StackGRU_DFRC', 'StackLSTM_DFRC', 'RecurrentDFRC',
}

# Models that use an LSTM backbone (need tuple hidden state)
LSTM_MODELS = {'VanillaLSTM', 'StackLSTM', 'StackLSTM_DFRC'}

# Pure counter models (no RNN backbone — different forward signature)
PURE_COUNTER_MODELS = {'RecurrentStack', 'RecurrentDFRC'}

# Vanilla models (no counter at all)
VANILLA_MODELS = {'VanillaRNN', 'VanillaGRU', 'VanillaLSTM'}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on Dyck-1 classification.')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(COUNTER_MODELS | VANILLA_MODELS),
                        help='Model architecture to train.')
    parser.add_argument('--dataset', type=str, default='large', choices=['large', 'small'],
                        help='Dataset size (default: large).')
    parser.add_argument('--hidden', type=int, default=2,
                        help='Hidden size / number of recurrent units (default: 2).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Defaults to 30 for large, 10000 for small.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001).')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of training runs (default: 1).')
    parser.add_argument('--activation', type=str, default='Sigmoid',
                        choices=['Sigmoid', 'Softmax', 'Clipping'],
                        help='Output activation function (default: Sigmoid).')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing Dyck-1 dataset files (default: data/).')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to write results and logs (default: results/).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_dataset(data_dir, dataset_size):
    """Load Dyck-1 train/short-test/long-test splits from text files."""
    if dataset_size == 'large':
        train_file = os.path.join(data_dir, 'Dyck1_Dataset_Binary_train__clf.txt')
        test_file = os.path.join(data_dir, 'Dyck1_Dataset_Binary_test__clf.txt')
        long_file = os.path.join(data_dir, 'Dyck1_Dataset_6pairs_balanced_clf.txt')
    else:
        train_file = os.path.join(data_dir, 'Dyck1_Dataset_2pairs_balanced_clf.txt')
        test_file = os.path.join(data_dir, 'Dyck1_Dataset_2pairs_balanced_clf.txt')
        long_file = None

    def read_file(path):
        sentences, labels = [], []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sentences.append(parts[0])
                    labels.append(parts[1])
        return sentences, labels

    X_train_raw, y_train_raw = read_file(train_file)
    X_test_raw, y_test_raw = read_file(test_file)
    X_long_raw, y_long_raw = read_file(long_file) if long_file else ([], [])

    return (X_train_raw, y_train_raw), (X_test_raw, y_test_raw), (X_long_raw, y_long_raw)


def encode_token(token):
    """One-hot encode a single bracket token: '(' → [1,0], ')' → [0,1]."""
    if token == '(':
        return [1.0, 0.0]
    elif token == ')':
        return [0.0, 1.0]
    else:
        raise ValueError(f"Unknown token: {token}")


def encode_sentence(sentence):
    """Encode a bracket string into a sequence of one-hot tensors."""
    return torch.tensor([encode_token(t) for t in sentence], dtype=torch.float32)


def encode_label(label):
    """Encode a binary label: 'Valid' → [1,0], 'Invalid' → [0,1]."""
    if label == 'Valid':
        return torch.tensor([1.0, 0.0])
    else:
        return torch.tensor([0.0, 1.0])


class DyckDataset(Dataset):
    def __init__(self, sentences, labels):
        self.X = [encode_sentence(s) for s in sentences]
        self.y = [encode_label(l) for l in labels]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model initialisation helpers
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'VanillaRNN': VanillaRNN, 'VanillaGRU': VanillaGRU, 'VanillaLSTM': VanillaLSTM,
    'StackRNN': StackRNN, 'StackGRU': StackGRU, 'StackLSTM': StackLSTM,
    'StackRNN_DFRC': StackRNN_DFRC, 'StackGRU_DFRC': StackGRU_DFRC, 'StackLSTM_DFRC': StackLSTM_DFRC,
    'RecurrentStack': RecurrentStack, 'RecurrentDFRC': RecurrentDFRC,
}


def build_model(model_name, hidden_size, output_activation):
    cls = MODEL_REGISTRY[model_name]
    return cls(
        input_size=2,
        hidden_size=hidden_size,
        output_size=2,
        output_activation=output_activation,
    )


def init_hidden(model_name, hidden_size, num_layers=1):
    """Initialise hidden state appropriate for the model type."""
    if model_name in LSTM_MODELS:
        return (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
    elif model_name in VANILLA_MODELS or model_name in COUNTER_MODELS - PURE_COUNTER_MODELS:
        return torch.zeros(num_layers, 1, hidden_size)
    return None


def init_stack_state(model_name):
    """Initialise the counter stack state for models that use one."""
    return torch.tensor([0, 0], dtype=torch.float32)


def reset_counter(model, model_name):
    """Reset the counter module between sequences."""
    if model_name in COUNTER_MODELS:
        model.stack.reset()


def forward_step(model, model_name, x, hidden, stack_state):
    """
    Route a single token through the model, returning (output, hidden, stack_state).

    Abstracts the three different forward signatures:
        - Vanilla: (x, hidden) → (output, hidden)
        - Stack/DFRC with RNN: (x, hidden, stack_state) → (output, hidden, stack_state)
        - Pure counter: (x, stack_state) → (output, stack_state)
    """
    if model_name in VANILLA_MODELS:
        output, hidden = model(x, hidden)
    elif model_name in PURE_COUNTER_MODELS:
        output, stack_state = model(x, stack_state)
    else:
        output, hidden, stack_state = model(x, hidden, stack_state)
    return output, hidden, stack_state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, model_name, train_loader, train_raw, args, results_dir, run_id):
    criterion = nn.MSELoss()
    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = StepLR(optimiser, step_size=100000, gamma=0.9)

    num_epochs = args.epochs
    labels = ['Valid', 'Invalid']
    accuracies = []

    X_raw, y_raw = train_raw

    log_path = os.path.join(results_dir, f'train_log_{model_name}_run{run_id}.txt')

    for epoch in range(num_epochs):
        epoch_start = time.time()
        num_correct = 0

        for i, (input_tensor, class_tensor) in enumerate(train_loader):
            class_tensor = class_tensor.squeeze(0)
            class_category = y_raw[i]

            reset_counter(model, model_name)
            optimiser.zero_grad()

            hidden = init_hidden(model_name, args.hidden)
            stack_state = init_stack_state(model_name)

            # Process sequence token by token (many-to-one: use final output)
            output = None
            for j in range(input_tensor.size(0)):
                output, hidden, stack_state = forward_step(
                    model, model_name, input_tensor[j].squeeze(), hidden, stack_state
                )

            loss = criterion(output, class_tensor)
            loss.backward()
            optimiser.step()

            guess = 'Valid' if output[0] >= output[1] else 'Invalid'
            if guess == class_category:
                num_correct += 1

        scheduler.step()
        accuracy = num_correct / len(train_loader) * 100
        accuracies.append(accuracy)
        elapsed = (time.time() - epoch_start) / 60
        print(f"Epoch {epoch+1}/{num_epochs} | Accuracy: {accuracy:.2f}% | Time: {elapsed:.2f} min")

    # Save model
    os.makedirs(results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_dir, f'{model_name}_run{run_id}.pt'))

    with open(log_path, 'w') as f:
        for ep, acc in enumerate(accuracies):
            f.write(f"Epoch {ep+1}: {acc:.2f}%\n")

    return accuracies[-1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, model_name, dataset_raw, hidden_size, num_layers=1, split_name='test'):
    """
    Evaluate model on a dataset split. Returns accuracy (%).

    Args:
        split_name: Label for logging ('short_test' or 'long_test').
    """
    model.eval()
    X_raw, y_raw = dataset_raw
    labels = ['Valid', 'Invalid']
    num_correct = 0
    num_samples = len(X_raw)

    with torch.no_grad():
        for i in range(num_samples):
            input_tensor = encode_sentence(X_raw[i])
            class_category = y_raw[i]

            reset_counter(model, model_name)
            hidden = init_hidden(model_name, hidden_size, num_layers)
            stack_state = init_stack_state(model_name)

            output = None
            for j in range(input_tensor.size(0)):
                output, hidden, stack_state = forward_step(
                    model, model_name, input_tensor[j].squeeze(), hidden, stack_state
                )

            guess = 'Valid' if output[0] >= output[1] else 'Invalid'
            if guess == class_category:
                num_correct += 1

    accuracy = num_correct / num_samples * 100
    print(f"{split_name} accuracy: {accuracy:.2f}%")
    return accuracy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Defaults
    if args.epochs is None:
        args.epochs = 30 if args.dataset == 'large' else 10000

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nModel:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden:  {args.hidden}")
    print(f"Epochs:  {args.epochs}")
    print(f"LR:      {args.lr}\n")

    # Load data
    train_raw, test_raw, long_raw = load_dataset(args.data_dir, args.dataset)
    X_train_raw, y_train_raw = train_raw

    train_dataset = DyckDataset(X_train_raw, y_train_raw)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Run multiple training runs
    train_accuracies, test_accuracies, long_accuracies = [], [], []

    for run in range(1, args.runs + 1):
        print(f"\n--- Run {run}/{args.runs} ---")

        model = build_model(args.model, args.hidden, args.activation)

        train_acc = train_model(
            model, args.model, train_loader, train_raw,
            args, args.results_dir, run_id=run
        )
        train_accuracies.append(train_acc)

        test_acc = evaluate_model(
            model, args.model, test_raw, args.hidden, split_name='short_test'
        )
        test_accuracies.append(test_acc)

        if long_raw[0]:
            long_acc = evaluate_model(
                model, args.model, long_raw, args.hidden, split_name='long_test'
            )
            long_accuracies.append(long_acc)

    # Summary
    summary_path = os.path.join(args.results_dir, f'summary_{args.model}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.model} | Dataset: {args.dataset} | Hidden: {args.hidden}\n\n")
        f.write(f"Train accuracy:      avg={np.mean(train_accuracies):.2f}%  "
                f"std={np.std(train_accuracies):.2f}%  "
                f"min={min(train_accuracies):.2f}%  max={max(train_accuracies):.2f}%\n")
        f.write(f"Short test accuracy: avg={np.mean(test_accuracies):.2f}%  "
                f"std={np.std(test_accuracies):.2f}%  "
                f"min={min(test_accuracies):.2f}%  max={max(test_accuracies):.2f}%\n")
        if long_accuracies:
            f.write(f"Long test accuracy:  avg={np.mean(long_accuracies):.2f}%  "
                    f"std={np.std(long_accuracies):.2f}%  "
                    f"min={min(long_accuracies):.2f}%  max={max(long_accuracies):.2f}%\n")

    print(f"\nSummary written to {summary_path}")


if __name__ == '__main__':
    main()
