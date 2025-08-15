# Image Captioning Model

This repository contains an image captioning model built with PyTorch, utilizing a ResNet50 encoder and a Transformer decoder to generate descriptive captions for images. Trained on the COCO dataset, the model employs beam search decoding for high-quality captions. The project includes a Streamlit web app for real-time image captioning, making it accessible to users via a browser.

## Features
- **Deep Learning Model**: Combines ResNet50 for feature extraction and a Transformer decoder for caption generation, trained on the COCO 2017 validation dataset.
- **Training and Evaluation**: Scripts to train the model and evaluate performance using the BLEU score.
- **Streamlit App**: A user-friendly interface (`app.py`) for uploading images and generating captions.
- **Progress Tracking**: Uses `tqdm` for visual training progress.
- **Pre-trained Model**: Includes model weights (`model_epoch_10.pth`) and vocabulary (`vocab.pkl`) for immediate use.

## Prerequisites
- Python 3.10
- Git
- Dependencies listed in `requirements.txt`

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ManasMehta1110/Image-Captioning.git
   cd Image-Captioning
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights** (if not in repository):
   - The pre-trained model (`model_epoch_10.pth`) is hosted on Google Drive due to GitHub's file size limit.
   - Download it from: [Google Drive Link](https://drive.google.com/file/d/<your-model-file-id>/view?usp=sharing)
   - Place `model_epoch_10.pth` in the repository root directory.
   - Alternatively, `app.py` automatically downloads the model from Google Drive during runtime.

4. **Download NLTK Data**:
   - The Streamlit app requires NLTK's `punkt` tokenizer. Run:
     ```python
     import nltk
     nltk.download('punkt')
     ```

## Usage
### Running the Streamlit App
1. Ensure `model_epoch_10.pth` and `vocab.pkl` are in the repository root (or `app.py` is configured to download the model).
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in your browser, upload an image (JPG/PNG), and view the generated caption.

### Training the Model
1. Run the training script:
   ```bash
   python image_captioning_with_streamlit.py
   ```
   This downloads the COCO dataset, trains the model, and saves `model_epoch_10.pth` and `vocab.pkl`.

2. Evaluate the model:
   ```python
   from image_captioning_with_streamlit import evaluate
   evaluate()
   ```
   This computes the BLEU score on the COCO validation set.

### Generating Captions Programmatically
Use the `generate_caption` function:
```python
from image_captioning_with_streamlit import generate_caption
caption = generate_caption('data/val2017/000000039769.jpg')
print(f"Generated Caption: {caption}")
```

## Deployment on Streamlit Cloud
1. **Push to GitHub**:
   - Ensure all files (`app.py`, `requirements.txt`, `vocab.pkl`, `image_captioning_with_streamlit.py`, `.streamlit/config.toml`) are in the repository.
   - If `model_epoch_10.pth` is too large, configure `app.py` to download it from Google Drive (update the `model_url` with your Google Drive file ID).

2. **Deploy on Streamlit Cloud**:
   - Log in to [Streamlit Cloud](https://streamlit.io/cloud).
   - Create a new app, select your repository (`ManasMehta1110/Image-Captioning`), and set the main file to `app.py`.
   - Deploy the app and access it via the provided URL.

3. **Verify Deployment**:
   - Check Streamlit Cloud logs for dependency installation or runtime errors.
   - Ensure the app loads, allows image uploads, and generates captions.

## Repository Structure
- `app.py`: Streamlit app for real-time image captioning.
- `image_captioning_with_streamlit.py`: Main script for training, evaluation, and inference.
- `requirements.txt`: List of Python dependencies.
- `vocab.pkl`: Serialized vocabulary for caption decoding.
- `model_epoch_10.pth`: Pre-trained model weights (download from Google Drive if not in repository).
- `.streamlit/config.toml`: Specifies Python 3.10 for Streamlit Cloud.

## Notes
- **Model Size**: The `model_epoch_10.pth` file is large (>100 MB). If not included in the repository, `app.py` downloads it from Google Drive. Ensure the Google Drive link is public and the file ID is correct in `app.py`.
- **Performance**: The model achieves a competitive BLEU score on the COCO validation set (run `evaluate()` for exact metrics).
- **Customization**: Modify `image_captioning_with_streamlit.py` to adjust model architecture (e.g., use ResNet18) or retrain on other datasets.

## License
MIT License

## Acknowledgments
- Built with PyTorch, Streamlit, and the COCO dataset.
- Inspired by advancements in computer vision and natural language processing.
