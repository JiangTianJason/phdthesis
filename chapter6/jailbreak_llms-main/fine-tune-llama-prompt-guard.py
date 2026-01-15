import torch
from datasets import load_dataset
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
prompt_injection_model_name = r'Llama-Prompt-Guard-2-22M'
tokenizer = AutoTokenizer.from_pretrained(prompt_injection_model_name)                     ###the original model is saved locally
model = AutoModelForSequenceClassification.from_pretrained(prompt_injection_model_name)
dataset = load_dataset("csv",data_files=r"./data/balanced_jailbreak_dataset_train_balanced.csv")
train_dataset = dataset['train']

def get_class_probabilities(text, temperature=1.0, device='cpu'):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.

    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    # Encode the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities

def get_jailbreak_score(text, temperature=1.0, device='cpu'):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(text, temperature, device)
    return probabilities[0, 1].item()

def train_model(train_dataset, model, tokenizer, batch_size=8, epochs=5, lr=5e-5, device='cpu'):
    """
    Train the model on the given dataset.

    Args:
        train_dataset (datasets.Dataset): The training dataset.
        model (transformers.PreTrainedModel): The model to train.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the texts.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate for the optimizer.
        device (str): The device to run the model on ('cpu' or 'cuda').
    """
    # Adjust the model's classifier to have two output labels
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.num_labels = 2

    model.to(device)
    model.train()

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare data loader
    def collate_fn(batch):
        texts = [item['prompt'] for item in batch]
        labels = torch.tensor([0 if item['type'] == 'benign' else 1 for item in batch])  # Convert string labels to integers
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Average loss in epoch {epoch + 1}: {total_loss / len(data_loader)}")

    torch.save(model, r'./fine-tuned-prompt-guard/5_epochs.pth')

# Example usage
train_model(train_dataset, model, tokenizer, device='cuda:0')