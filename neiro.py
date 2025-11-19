import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# 1. Эмбеддинги
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        return self.layer_norm(token_embeds + position_embeds)

# 2. Многоголовое внимание
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, embed_dim)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.fc(attn_output)

# 3. Фидфорвардный слой
class FeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

# 4. Слой энкодера
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardLayer(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

# 5. BERT энкодер
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, hidden_dim, num_layers):
        super(BERTEncoder, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 6. Классификатор на основе BERT
class BERTLanguageClassifier(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(BERTLanguageClassifier, self).__init__()
        self.bert = BERTEncoder(vocab_size, max_seq_len, embed_dim, num_heads, hidden_dim, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, mask=None):
        x = self.bert(input_ids, mask)
        cls_token_output = x[:, 0, :]  # Используем выход [CLS] токена
        return self.classifier(cls_token_output)

# 7. Пример использования с токенизацией и обучением
VOCAB = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "Bonjour": 3,  # French
    "Hello": 4,    # English
    "Hola": 5,     # Spanish
    "Hallo": 6,    # German
    "Ciao": 7,     # Italian
    "Salut": 8,    # French
    "Hi": 9,       # English
    "Holla": 10,   # Spanish
    "Guten": 11,   # German
    "Buongiorno": 12,  # Italian
    # Add more words for each language
}
def tokenize(text, max_len=10):
    tokens = [VOCAB.get(word, VOCAB["[PAD]"]) for word in text.split()]
    tokens = [VOCAB["[CLS]"]] + tokens[:max_len - 2] + [VOCAB["[SEP]"]]
    tokens += [VOCAB["[PAD]"]] * (max_len - len(tokens))
    return tokens

class LanguageDataset(Dataset):
    def __init__(self, texts, labels, max_len=10):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(tokenize(self.texts[idx], self.max_len))
        label = torch.tensor(self.labels[idx])
        return {"input_ids": input_ids, "label": label}

# Данные
texts = ["Bonjour", "Hello", "Hola", "Hallo", "Ciao"]
labels = [0, 1, 2, 3, 4]  # Метки для языков
language = ["French", "English", "Spanish", "German", "Italian"]

# Параметры модели
VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LEN = 100
EMBED_DIM = 768
NUM_HEADS = 12
HIDDEN_DIM = 3072
NUM_LAYERS = 2
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTLanguageClassifier(VOCAB_SIZE, MAX_SEQ_LEN, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)

dataset = LanguageDataset(texts, labels)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Обучение
model.train()
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# Тестирование
# Функция для предсказания языка текста
def predict_language(text):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenize(text)]).to(device)
        output = model(input_ids)
        predicted_label = torch.argmax(output, dim=1).item()
        return language[predicted_label]

# Чтение текста из файла
def read_text_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
        return text
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

# Главная функция
if __name__ == "__main__":
    text = read_text_from_file("G:/Учеба/Магистратура/1 семестр/Нейросеть/Text.txt")
    if text:
        language_detected = predict_language(text)
        print(f"Определённый язык: {language_detected}")
    else:
        print("Не удалось прочитать файл.")
