
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pickle

# Model definitions (copied from main script)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, num_heads=8):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encode = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions, tgt_mask=None):
        memory = features.unsqueeze(1).transpose(0, 1)
        embed = self.embed(captions)
        embed = self.pos_encode(embed.transpose(0, 1))
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(embed.size(0)).to(embed.device)
        output = self.transformer(embed, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output.transpose(0, 1))
        return output

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3, num_heads=8):
        super(ImageCaptioner, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderTransformer(embed_size, hidden_size, vocab_size, num_layers, num_heads)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions[:, :-1])
        return outputs

# Vocabulary class (copied from main script)
class Vocabulary:
    def __init__(self, freq_threshold=0):  # Set to 0 for loading pre-built vocab
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_english(text):
        return [tok.lower() for tok in nltk.word_tokenize(text)]

    def build_vocabulary(self, sentence_list):
        counts = Counter()
        for sentence in sentence_list:
            counts.update(self.tokenizer_english(sentence))
        for word, freq in counts.items():
            if freq > self.freq_threshold:
                self.stoi[word] = len(self.itos)
                self.itos[len(self.itos)] = word

    def numericalize(self, text):
        tokenized_text = self.tokenizer_english(text)
        return [
            self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text
        ]

# Beam search decoding (copied from main script)
def beam_search_decode(decoder, features, vocab, beam_width=3, max_len=50):
    device = features.device
    start = [vocab.stoi["<start>"]]
    start_word = [(start, 0.0)]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            captions = torch.tensor([s[0]], dtype=torch.long).to(device)
            outputs = decoder(features, captions)
            probs = torch.log_softmax(outputs[:, -1, :], dim=1)
            top_probs, top_words = probs.topk(beam_width, dim=1)
            for i in range(beam_width):
                next_seq = s[0] + [top_words[0, i].item()]
                next_score = s[1] + top_probs[0, i].item()
                temp.append((next_seq, next_score))
        start_word = sorted(temp, reverse=True, key=lambda x: x[1])[:beam_width]
        if start_word[0][0][-1] == vocab.stoi["<end>"]:
            break
    seq = start_word[0][0]
    cap_list = [vocab.itos[idx] for idx in seq[1:-1]] if seq[-1] == vocab.stoi["<end>"] else [vocab.itos[idx] for idx in seq[1:]]
    return cap_list

# Streamlit app
st.title("Image Captioning App")
st.write("Upload an image to generate a caption using a pre-trained model.")

# Load vocabulary and model
try:
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
except FileNotFoundError:
    st.error("Vocabulary file (vocab.pkl) not found. Please ensure it is in the same directory.")
    st.stop()

try:
    embed_size = 256
    hidden_size = 512
    num_layers = 3
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageCaptioner(embed_size, hidden_size, vocab_size, num_layers, num_heads).to(device)
    model.load_state_dict(torch.load('model_epoch_10.pth', map_location=device))
    model.eval()
except FileNotFoundError:
    st.error("Model file (model_epoch_10.pth) not found. Please ensure it is in the same directory.")
    st.stop()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        features = model.encoder(image_tensor)
        caption = beam_search_decode(model.decoder, features, vocab)
    st.write(f"**Generated Caption**: {' '.join(caption)}")
    