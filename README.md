# Image Captioning with Transformer Architecture

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.25%2B-green?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

A state-of-the-art **Image Captioning** system that generates natural language descriptions for images using a **CNN-Transformer** architecture. The model combines ResNet-50 for visual feature extraction with a Transformer decoder for caption generation, trained on the COCO dataset.

## 🌟 Features

- **Modern Architecture**: CNN Encoder + Transformer Decoder
- **Pre-trained Backbone**: ResNet-50 for robust feature extraction  
- **Attention Mechanism**: Multi-head self-attention for better caption quality
- **Advanced Decoding**: Beam search for improved inference
- **Comprehensive Evaluation**: BLEU score metrics
- **Interactive Demo**: Live Streamlit web application
- **Easy Deployment**: Ready-to-use inference pipeline

## 🚀 Live Demo

Try the model live on our Streamlit app: **[Image Captioning Demo](https://image-captioning-78phdkkky32ytejfox5w6t.streamlit.app)**

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 5GB free disk space for COCO dataset

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/ManasMehta1110/Image-Captioning.git
   cd Image-Captioning
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install nltk pycocotools tqdm pillow
   pip install streamlit  # For web app
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

## ⚡ Quick Start

### Generate Caption for Single Image

```python
from generate_caption import generate_caption

# Generate caption for your image
caption = generate_caption('path/to/your/image.jpg')
print(f"Generated Caption: {caption}")
```

### Run Training (Optional)

```bash
python main.py  # This will download COCO dataset and start training
```

### Launch Web App

```bash
streamlit run app.py
```

## 🏗️ Model Architecture

### CNN Encoder
- **Backbone**: Pre-trained ResNet-50
- **Features**: 2048-dimensional image representations
- **Output**: 256-dimensional embeddings via linear projection

### Transformer Decoder
- **Layers**: 3 transformer decoder layers
- **Attention Heads**: 8 multi-head attention mechanisms
- **Embedding Size**: 256 dimensions
- **Vocabulary**: Built from COCO captions (5+ frequency threshold)

### Training Configuration
```python
{
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 3,
    "num_heads": 8,
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 10
}
```

## 🎯 Training

The model is trained on the COCO 2017 validation set with the following process:

1. **Data Preprocessing**
   - Image resize to 224x224
   - ImageNet normalization
   - Caption tokenization and padding

2. **Loss Function**
   - Cross-entropy loss with padding token ignored
   - Teacher forcing during training

3. **Optimization**
   - Adam optimizer with learning rate 3e-4
   - Gradient clipping for stability

```bash
# Start training
python main.py

# Monitor training progress
# Model checkpoints saved as model_epoch_{epoch}.pth
```

## 📊 Evaluation

### Metrics
- **BLEU Score**: Industry-standard metric for caption quality
- **Corpus-level evaluation**: Comprehensive assessment across validation set

### Run Evaluation
```python
from evaluate import evaluate

# Evaluate trained model
evaluate('model_epoch_10.pth')
```

## 💻 Usage Examples

### 1. Basic Inference
```python
import torch
from PIL import Image
from generate_caption import generate_caption

# Load and caption an image
caption = generate_caption('example.jpg', 'model_epoch_10.pth', 'vocab.pkl')
print(f"Caption: {caption}")
```

### 2. Batch Processing
```python
import os
from generate_caption import generate_caption

# Process multiple images
image_folder = 'test_images/'
for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, img_file)
        caption = generate_caption(img_path)
        print(f"{img_file}: {caption}")
```

### 3. Custom Model Loading
```python
from models import ImageCaptioner
import pickle
import torch

# Load custom trained model
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

model = ImageCaptioner(256, 512, len(vocab), 3, 8)
model.load_state_dict(torch.load('your_model.pth'))
model.eval()
```

## 📁 Dataset

### COCO 2017
- **Images**: ~40K validation images
- **Captions**: ~200K human-annotated captions
- **Download**: Automatic via training script
- **Size**: ~1.3GB total

### Data Structure
```
data/
├── val2017/              # Validation images
├── annotations/
│   └── captions_val2017.json
└── vocab.pkl             # Generated vocabulary
```

## 📈 Results

### Performance Metrics
- **BLEU-1**: ~0.65
- **BLEU-2**: ~0.45  
- **BLEU-3**: ~0.32
- **BLEU-4**: ~0.24

### Sample Outputs
```
Image: beach_scene.jpg
Caption: "a group of people sitting on a beach near the ocean"

Image: cat_photo.jpg  
Caption: "a black and white cat sitting on a wooden table"

Image: city_street.jpg
Caption: "a busy city street with cars and buildings"
```

## 🗂️ Repository Structure

```
Image-Captioning/
├── main.py                 # Training script with data download
├── models.py              # Model architectures
├── dataset.py             # Dataset and data loading utilities
├── evaluate.py            # Evaluation metrics and functions  
├── generate_caption.py    # Inference pipeline
├── app.py                 # Streamlit web application
├── vocab.pkl              # Saved vocabulary (generated)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── models/               # Saved model checkpoints
    ├── model_epoch_1.pth
    ├── model_epoch_2.pth
    └── ...
```

## 🔧 Advanced Configuration

### Hyperparameter Tuning
Modify these parameters in the training script:

```python
# Model architecture
EMBED_SIZE = 256          # Embedding dimensions
HIDDEN_SIZE = 512         # Hidden layer size
NUM_LAYERS = 3            # Transformer layers
NUM_HEADS = 8             # Attention heads

# Training settings
LEARNING_RATE = 3e-4      # Learning rate
BATCH_SIZE = 32           # Batch size
EPOCHS = 10               # Training epochs
MAX_LEN = 50              # Maximum caption length
```

### Custom Vocabulary
```python
# Build custom vocabulary with different threshold
vocab = Vocabulary(freq_threshold=10)  # Higher threshold = smaller vocab
vocab.build_vocabulary(your_captions)
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   batch_size = 16  # or 8
   ```

2. **Download Failures**
   ```bash
   # Manual COCO download
   wget http://images.cocodataset.org/zips/val2017.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   ```

3. **Import Errors**
   ```bash
   pip install --upgrade torch torchvision
   pip install pycocotools-windows  # For Windows users
   ```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black *.py
flake8 *.py
```

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Microsoft COCO Dataset](https://cocodataset.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Manas Mehta**
- GitHub: [@ManasMehta1110](https://github.com/ManasMehta1110)
- LinkedIn: [Connect with me](https://linkedin.com/in/manasmehta1110)

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- COCO dataset creators for high-quality annotations
- Streamlit for easy web app deployment
- ResNet authors for the pre-trained backbone

---

<p align="center">
  <strong>⭐ Star this repository if you found it helpful!</strong>
</p>

<p align="center">
  Made with ❤️ by Manas Mehta
</p>
