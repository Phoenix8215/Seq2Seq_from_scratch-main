import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from torch.utils.data import DataLoader, TensorDataset


# --------------------------
# 1. 数据预处理相关函数
# --------------------------
def pad_sequence(seq, desired_len, pad_char=" "):
    """确保字符串 seq 补齐到 desired_len 长度，使用空格填充"""
    return seq.ljust(desired_len, pad_char)


def load_addition_data(file_path, input_len=7, output_len=5):
    """加载数据并填充到固定长度"""
    data_pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "_" not in line:
                continue
            input_seq, output_seq = line.split("_")
            input_seq = pad_sequence(input_seq.strip(), input_len)  # 输入序列补齐
            output_seq = "_" + output_seq.strip()  # 输出序列以 `_` 开头
            output_seq = pad_sequence(output_seq, output_len)  # 输出序列补齐
            data_pairs.append((input_seq[::-1], output_seq)) # ❗此处使用反转的技巧
    return data_pairs


def build_vocab(data_pairs):
    """构建字符到索引的映射"""
    all_chars = sorted(set("".join(inp + outp for inp, outp in data_pairs)))
    char2idx = {ch: i for i, ch in enumerate(all_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


def seq_to_tensor(seq, char2idx):
    """将字符串转换为 Tensor"""
    return torch.tensor([char2idx[ch] for ch in seq], dtype=torch.long)


# --------------------------
# 2. 模型定义
# --------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_dim)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.size()
        h, c = torch.zeros(
            batch_size, hidden_dim, device=input_seq.device
        ), torch.zeros(batch_size, hidden_dim, device=input_seq.device)

        for t in range(seq_len):
            emb = self.embedding(input_seq[:, t])
            h, c = self.lstm_cell(emb, (h, c))

        return h, c


# class Decoder(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super(Decoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.lstm_cell = nn.LSTMCell(embed_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, target_seq, h, c, teacher_forcing_ratio=0.5):
#         batch_size, seq_len = target_seq.size()
#         outputs = []
#         input_token = target_seq[:, 0]

#         for t in range(seq_len):
#             emb = self.embedding(input_token)
#             h, c = self.lstm_cell(emb, (h, c))
#             logits = self.fc(h)
#             outputs.append(logits.unsqueeze(1))

#             if t < seq_len - 1:
#                 teacher_force = random.random() < teacher_forcing_ratio
#                 input_token = (
#                     target_seq[:, t + 1] if teacher_force else logits.argmax(dim=1)
#                 )

#         return torch.cat(outputs, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(self, target_seq, context, h, c, teacher_forcing_ratio=0.5):

        batch_size, seq_len = target_seq.size()
        outputs = []
        input_token = target_seq[:, 0]

        for t in range(seq_len):
            emb = self.embedding(input_token)  # (batch_size, embed_dim)
            lstm_input = torch.cat(
                [emb, context], dim=1
            )  # (batch_size, embed_dim + hidden_dim)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output_input = torch.cat(
                [h, context], dim=1
            )  # (batch_size, hidden_dim + hidden_dim)
            logits = self.fc(output_input)  # (batch_size, vocab_size)
            outputs.append(logits.unsqueeze(1))

            if t < seq_len - 1:
                teacher_force = random.random() < teacher_forcing_ratio
                input_token = (
                    target_seq[:, t + 1] if teacher_force else logits.argmax(dim=1)
                )
        return torch.cat(outputs, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        h, c = self.encoder(src)
        outputs = self.decoder(trg, h, h, c, teacher_forcing_ratio)
        return outputs


# --------------------------
# 3. 预测函数
# --------------------------
def predict_sample(input_str, model, char2idx, idx2char, input_len=7, output_len=5):
    """给定输入字符串，使用完整的 Seq2Seq 模型进行预测"""
    input_str = "_" + input_str.strip()
    input_str = pad_sequence(input_str, input_len)
    inp_tensor = seq_to_tensor(input_str, char2idx).unsqueeze(0).to(device)

    trg_tensor = torch.full(
        (1, output_len), char2idx[" "], dtype=torch.long, device=device
    )

    with torch.no_grad():
        outputs = model(inp_tensor, trg_tensor, teacher_forcing_ratio=0.0)
        predicted_indices = outputs.argmax(dim=-1).squeeze(0).tolist()

    return "".join(idx2char[idx] for idx in predicted_indices)


# --------------------------
# 4. 训练或测试
# --------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    file_path = "addition.txt"
    input_len, output_len = 7, 5
    num_epochs, batch_size = 100, 64
    embed_dim, hidden_dim = 32, 64
    learning_rate = 0.001
    checkpoint_path = "seq2seq_checkpoint.pth"

    data_pairs = load_addition_data(file_path, input_len, output_len)
    char2idx, idx2char = build_vocab(data_pairs)
    vocab_size = len(char2idx)

    encoder = Encoder(vocab_size, embed_dim, hidden_dim).to(device)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()
    else:
        print("No pre-trained model found. Starting training...")
        dataset = TensorDataset(
            torch.stack([seq_to_tensor(inp, char2idx) for inp, _ in data_pairs]),
            torch.stack([seq_to_tensor(out, char2idx) for _, out in data_pairs]),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for inp_tensor, out_tensor in dataloader:
                inp_tensor, out_tensor = inp_tensor.to(device), out_tensor.to(device)
                optimizer.zero_grad()
                outputs = model(inp_tensor, out_tensor, teacher_forcing_ratio=0.5)
                loss = criterion(outputs.view(-1, vocab_size), out_tensor.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
            )
        torch.save(model.state_dict(), checkpoint_path)

    print("Testing prediction...")
    test_input = "16+75"
    print(
        f"Input: {test_input}, Predicted: {predict_sample(test_input, model, char2idx, idx2char)}"
    )
