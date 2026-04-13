# 🌿 PlantScan — Plant Disease Detection

A Flask web app for real-time plant disease detection using a Keras deep learning model.

## Project Structure

```
plant_disease_app/
├── app.py                  # Flask backend
├── requirements.txt
├── templates/
│   └── index.html          # Frontend UI
└── static/
    └── uploads/            # Uploaded images (auto-created)
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your model & classes
Open `app.py` and update:

- **`MODEL_PATH`** — path to your `.keras` model (already set to `/kaggle/working/best_model_phase1.keras`)
- **`CLASS_NAMES`** — the list of class labels **must match** the exact order used during training
- **`IMG_SIZE`** — set to `(224, 224)` by default; change if your model uses a different input size

### 3. Run the app
```bash
python app.py
```

Then open: **http://localhost:5000**

## How It Works

1. User uploads a leaf image (JPG/PNG/WEBP)
2. Flask receives the image, preprocesses it (resize → normalize → expand dims)
3. Model runs inference and returns class probabilities
4. Top-5 predictions are shown with confidence bars

## Customization

| Setting | Location | Default |
|---|---|---|
| Model path | `app.py → MODEL_PATH` | `/kaggle/working/best_model_phase1.keras` |
| Image size | `app.py → IMG_SIZE` | `(224, 224)` |
| Class labels | `app.py → CLASS_NAMES` | PlantVillage 38 classes |
| Max upload size | `app.py → MAX_CONTENT_LENGTH` | 16 MB |
| Port | `app.py → app.run(port=...)` | 5000 |

## Notes

- The app normalizes pixel values to `[0, 1]` (divides by 255). If your model was trained with a different normalization (e.g., `[-1, 1]`), update `preprocess_image()` in `app.py`.
- The default `CLASS_NAMES` list assumes the PlantVillage dataset with 38 classes. Replace with your actual class names if different.