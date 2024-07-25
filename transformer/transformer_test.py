from torchtext.data.metrics import bleu_score
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import torch.optim as optim
import spacy
import datasets
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math

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

model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

model.load_state_dict(torch.load('transformer_german_to_english.pt'))

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

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


test_loss = evaluate(model, test_dataloader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')

# 번역(translation) 함수


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval()  # 평가 모드

    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    if logging:
        print(f"전체 소스 토큰: {tokens}")

    src_stoi = src_field.get_stoi()
    src_indexes = [src_stoi.get(token, src_stoi['<pad>']) for token in tokens]
    if logging:
        print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # 소스 문장에 따른 마스크 생성
    src_mask = model.make_src_mask(src_tensor)

    # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 처음에는  토큰 하나만 가지고 있도록 하기
    trg_stoi = trg_field.get_stoi()
    trg_indexes = [trg_field['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # 출력 문장에 따른 마스크 생성
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)

        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:, -1].item()

        # 를 만나는 순간 끝
        if pred_token == trg_field['<eos>']:
            break

        trg_indexes.append(pred_token)  # 출력 문장에 더하기

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_itos = trg_field.get_itos()
    trg_tokens = [trg_itos[i] for i in trg_indexes]

    # 첫 번째 는 제외하고 출력 문장 반환
    return trg_tokens[1:], attention


# 번역(translation) 함수 테스트
example_idx = 10
src = test_data[example_idx]['de_tokens']
trg = test_data[example_idx]['en_tokens']

print(f'소스 문장: {src}')
print(f'타겟 문장: {trg}')

translation, attention = translate_sentence(
    src, de_vocab, en_vocab, model, device, logging=True)

print("모델 출력 결과:", " ".join(translation))

# 어텐션(attention) 시각화 함수


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    assert n_rows * n_cols == n_heads

    # 출력할 그림 크기 조절
    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        # 어텐션(Attention) 스코어 확률 값을 이용해 그리기
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + [''] + [t.lower()
                           for t in sentence] + [''], rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


# display_attention(src, translation, attention)

# BLEU Score 계산


def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = datum['de_tokens']
        trg = datum['en_tokens'][1:-1]

        pred_trg, _ = translate_sentence(
            src, src_field, trg_field, model, device, max_len, logging=False)

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index += 1
        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(data)}]")
            print(f"예측: {pred_trg}")
            print(f"정답: {trg}")

    bleu = bleu_score(pred_trgs, trgs, max_n=4,
                      weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')

    individual_bleu1_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    individual_bleu2_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    individual_bleu3_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    individual_bleu4_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

    print(f'Individual BLEU1 score = {individual_bleu1_score*100:.2f}')
    print(f'Individual BLEU2 score = {individual_bleu2_score*100:.2f}')
    print(f'Individual BLEU3 score = {individual_bleu3_score*100:.2f}')
    print(f'Individual BLEU4 score = {individual_bleu4_score*100:.2f}')

    cumulative_bleu1_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    cumulative_bleu2_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    cumulative_bleu3_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    cumulative_bleu4_score = bleu_score(
        pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

    print(f'Cumulative BLEU1 score = {cumulative_bleu1_score*100:.2f}')
    print(f'Cumulative BLEU2 score = {cumulative_bleu2_score*100:.2f}')
    print(f'Cumulative BLEU3 score = {cumulative_bleu3_score*100:.2f}')
    print(f'Cumulative BLEU4 score = {cumulative_bleu4_score*100:.2f}')


show_bleu(test_data, de_vocab, en_vocab, model, device)
