import random
import time
import math
import torch.optim as optim
import spacy
import datasets
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


# 데이터 로드
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

# 데이터 전처리


def tokenize_data(dataset):
    lower = True
    sos_token = "<sos>"
    eos_token = "<eos>"

    fn_kwargs = {"en_nlp": spacy_en, "de_nlp": spacy_de,
                 "lower": lower, "sos_token": sos_token, "eos_token": eos_token}

    def tokenizer(data, en_nlp, de_nlp, lower, sos_token, eos_token):
        en_tokens = [token.text for token in en_nlp.tokenizer(data["en"])]
        de_tokens = [token.text for token in de_nlp.tokenizer(data["de"])]
        if lower:
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        return {"en_tokens": en_tokens, "de_tokens": de_tokens}
    return dataset.map(tokenizer, fn_kwargs=fn_kwargs)


dataset = datasets.load_dataset('bentrevett/multi30k')
train_data, valid_data, test_data = tokenize_data(dataset['train']), tokenize_data(
    dataset['validation']),  tokenize_data(dataset['test'])

# vocab 생성


def build_vocab(data, min_freq=2):
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    special_tokens = [sos_token, eos_token, unk_token, pad_token]
    vocab = build_vocab_from_iterator(
        data, min_freq=min_freq, specials=special_tokens)
    vocab.set_default_index(vocab['<unk>'])
    return vocab


en_vocab = build_vocab((data["en_tokens"] for data in train_data))
de_vocab = build_vocab((data["de_tokens"] for data in train_data))

# 데이터 로더 생성
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print()

batch_size = 128


def get_collate_fn(pad_index, data):
    def collate_fn(batch):
        batch_en_ids = pad_sequence([data["en_ids"]
                                    for data in batch], batch_first=True, padding_value=pad_index)
        batch_de_ids = pad_sequence([data["de_ids"]
                                    for data in batch], batch_first=True, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids
        }
        return batch
    return collate_fn


def get_data_loader(data, batch_size):
    pad_index = en_vocab['<pad>']
    collate_fn = get_collate_fn(pad_index, data)
    data_loader = DataLoader(data, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    return data_loader


def numericalize_data(data, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(data["en_tokens"])
    de_ids = de_vocab.lookup_indices(data["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}


# 데이터 정수화
fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}
train_data = train_data.map(numericalize_data, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_data, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_data, fn_kwargs=fn_kwargs)

# 데이터 torch.Tensor로 변환
data_type = "torch"
format_columns = ["en_ids", "de_ids"]
train_data.set_format(
    type=data_type, columns=format_columns, output_all_columns=True)
valid_data.set_format(
    type=data_type, columns=format_columns, output_all_columns=True)
test_data.set_format(type=data_type, columns=format_columns,
                     output_all_columns=True)

train_dataloader = get_data_loader(train_data, batch_size)
valid_dataloader = get_data_loader(valid_data, batch_size)
test_dataloader = get_data_loader(test_data, batch_size)

for i, batch in enumerate(train_dataloader):
    src = batch["en_ids"]
    trg = batch["de_ids"]

    print(f"첫 번째 배치 크기: {src.shape}")

    # 현재 배치에 있는 하나의 문장에 포함된 정보 출력
    for i in range(src.shape[1]):
        print(f"인덱스 {i}: {src[0][i].item()}")  # 여기에서는 [Seq_num, Seq_len]

    # 첫 번째 배치만 확인
    break


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads  # head의 개수
        self.head_dim = hidden_dim // n_heads  # 각 head의 hidden_dim 크기

        # 이후 결과 dimension을 h개로 쪼개서 사용
        self.fc_q = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # hidden_dim -> n_heads X head_dim 형태로 변경
        # [batch_size, n_heads, length, head_dim]
        Q = Q.reshape(batch_size, -1, self.n_heads,
                      self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads,
                      self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads,
                      self.head_dim).permute(0, 2, 1, 3)

        # attention energy 계산(계산 추가이해)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # mask 적용(무한에 가깝게 만들어줌)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention score 계산
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous()  # 메모리 레이아웃을 연속적으로 변경
        x = x.reshape(batch_size, -1, self.hidden_dim)  # 다시 원래 형태로 변경
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask):
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(
            max_length, hidden_dim)  # 실제 논문과 달리 pos_embedding을 학습하

        self.layers = nn.ModuleList([EncoderLayer(
            hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):

        batch_size = src.shape[0]  # 문장의 개수
        src_len = src.shape[1]  # 가장 긴 문장의 길이

        # unsqueeze(0)(가장 바깥 차원추가), 각 문장마다 적용
        pos = torch.arange(0, src_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]
        # pos: [batch_size, src_len]
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # 모든 레이어를 차례대로 거치면서 순전파(forward) 수행
        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim, n_heads, dropout_ratio, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hidden_dim, n_heads, dropout_ratio, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim, pf_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(
            hidden_dim, n_heads, pf_dim, dropout_ratio, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # src -> mask값을 0으로 변경
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    # trg -> 각 단어는 다음 단어가 무엇인지 알 수 없도록 mask
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(
            1).unsqueeze(2).to(self.device)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


# Traning
INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1


SRC_PAD_IDX = en_vocab['<pad>']
TRG_PAD_IDX = de_vocab['<pad>']

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS,
              ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS,
              DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)


# Adam optimizer로 학습 최적화
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


# 모델 학습(train) 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        print(i, batch)
        src = batch['de_ids'].to(device)
        trg = batch['en_ids'].to(device)

        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스()는 제외
        # 입력을 할 때는 부터 시작하도록 처리
        output, _ = model(src, trg[:, :-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0()은 제외
        trg = trg[:, 1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward()  # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval()  # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch['de_ids'].to(device)
            trg = batch['en_ids'].to(device)

            # 출력 단어의 마지막 인덱스()는 제외
            # 입력을 할 때는 부터 시작하도록 처리
            output, _ = model(src, trg[:, :-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0()은 제외
            trg = trg[:, 1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()  # 시작 시간 기록

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_dataloader, criterion)

    end_time = time.time()  # 종료 시간 기록
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer_german_to_english.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'''\tTrain Loss: {train_loss:.3f} | Train PPL: {
          math.exp(train_loss):.3f}''')
    print(f'''\tValidation Loss: {valid_loss:.3f}
          | Validation PPL: {math.exp(valid_loss):.3f}''')


# 모델 저장
torch.save(model.state_dict(), 'transformer_german_to_english.pt')
