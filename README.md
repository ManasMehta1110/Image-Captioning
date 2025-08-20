
# Image Captioning with CNN + Transformer

This project implements an **image captioning model** using a **ResNet-50 CNN encoder** and a **Transformer decoder**. 
The model is trained and evaluated on the [MS COCO 2017](https://cocodataset.org/#home) dataset.

**Key highlights:**
- **Encoder:** Pretrained ResNet-50 extracts image features (frozen weights for stability).
- **Decoder:** Transformer-based caption generator with multi-head attention.
- **Training:** Cross-entropy loss with teacher forcing.
- **Evaluation:** BLEU score computation for caption quality.
- **Inference:** Greedy decoding and beam search supported.

> **Note:** This repository does **not** contain the full training code.  
> It contains only the trained model checkpoints saved after each epoch.  
> The model was developed and trained entirely on **Google Colab**.

---

## Features
- Custom vocabulary builder using NLTK tokenization  
- Image preprocessing with `torchvision.transforms`  
- COCO dataset integration via `pycocotools`  
- Easy model saving and loading with `torch.save()` / `torch.load()`  
- Beam search decoding for high-quality captions  

---

## Usage
Since this repo contains **only the trained weights**, you can:
1. Load the provided `.pth` checkpoint files into your own training/inference code.
2. Use the `generate_caption()` function (from the original code) to produce captions:
```python
caption = generate_caption("example.jpg", model_path="model_epoch_10.pth", vocab_path="vocab.pkl")
print("Generated caption:", caption)
```

If you need the full training pipeline (data loading, training loops, evaluation scripts), 
refer to the **project development notebook on Google Colab** (not included in this repository).

---

## Model Architecture
**1. Encoder (CNN)**
- Pretrained ResNet-50 (last layer removed)
- Linear projection to embedding space (256-D)
- Batch normalization for stability

**2. Decoder (Transformer)**
- Positional encoding
- Multi-head attention with 3 decoder layers, 8 heads each
- Output projection to vocabulary size

---

## Dataset
- **Dataset:** MS COCO 2017
- **Images used:** `val2017` (~1GB) for demonstration/training subset
- **Annotations:** `captions_val2017.json`

---

## Collaborators
- **Manas Mehta** *(GitHub: [ManasMehta1110](https://github.com/ManasMehta1110))*
- **Vishvesh Sharma** *(GitHub: [VishveshSharma2005](https://github.com/VishveshSharma2005))*

---

## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- pycocotools
- NLTK
- tqdm
- PIL

(Colab users can simply install dependencies using pip in the notebook.)

---

## Inference Example
Once you load the model and vocabulary:
```python
caption = generate_caption("sample_image.jpg", "model_epoch_10.pth", "vocab.pkl")
print("Caption:", caption)
```
