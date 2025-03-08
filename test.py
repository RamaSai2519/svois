import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'unk_token': '[UNK]'})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0  # Ensure divisibility
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape
        Q = self.query(x).view(batch_size, seq_length,
                               self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length,
                             self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length,
                               self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V).transpose(
            1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.fc_out(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, feedforward_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_size)
        )

    def forward(self, x):
        attention_out = self.multi_head_attention(x)
        x = self.norm1(x + attention_out)  # Residual Connection

        feedforward_out = self.feedforward(x)
        x = self.norm2(x + feedforward_out)  # Residual Connection

        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_dim, max_length):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_size=embed_size, num_heads=num_heads, feedforward_dim=ff_dim)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            batch_size, seq_length).to(x.device)
        x = self.embedding(x) + self.position_embedding(positions)

        for layer in self.encoder_layers:
            x = layer(x)

        return self.fc_out(x)


class TextDataset(data.Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long)
        }


def generate_text(model, tokenizer, start_text, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_id = torch.argmax(
                outputs[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


ff_dim = 256
num_heads = 4
num_layers = 2
embed_size = 128
max_length = 128
vocab_size = len(tokenizer)

print(
    f"Max token ID in dataset: {max(max(seq) for seq in tokenized_datasets['train']['input_ids'])}")
print(f"Vocab Size: {vocab_size}")

model = MiniTransformer(vocab_size, embed_size, num_heads,
                        num_layers, ff_dim, max_length)
model.embedding = nn.Embedding(vocab_size, embed_size)
model.position_embedding = nn.Embedding(max_length, embed_size)
train_dataset = TextDataset(tokenized_datasets["train"])
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()


epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["input_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Generate text
print(generate_text(model, tokenizer, "The future of AI is"))
