# WriteFusion AI

WriteFusion AI is a Flask + TensorFlow language-model web app for next-word prediction, story continuation, paragraph generation, attention inspection, and model diagnostics. It serves a locally trained Keras Transformer model through a polished browser UI.

## What This App Does

- Predicts the next word while the user types.
- Shows ranked candidate words with probabilities.
- Lets the user accept the best suggestion with Tab on desktop or Enter on mobile.
- Generates story and paragraph continuations in the same editor field.
- Supports temperature, Top-K, Top-P, Top-K + Top-P, Typical, Min-P, Greedy, and Smart Anti-Repetition decoding.
- Reduces repeated generation with repetition penalty, no-repeat n-gram blocking, dynamic temperature, and dynamic Top-K.
- Shows attention focus tokens from the Transformer model.
- Provides a perplexity probe for confidence checking.
- Includes a model-development popup with architecture and training-process details.
- Supports light/dark UI and responsive mobile/desktop layout.

## Current Project Structure

```text
WriteFusion-AI/
|-- app.py                  # Flask backend, model loading, APIs, decoding logic
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
|-- render.yaml             # Render deployment config
|-- make_corpus_stats.py    # Optional corpus-stat helper
|-- best_model.keras        # Trained Keras model artifact
|-- tokenizer.pkl           # Saved tokenizer artifact
|-- model_config.json       # Model hyperparameter/config artifact
|-- training_log.csv        # Training history/log data
|-- static/                 # Images and static assets
|-- templates/
|   `-- index.html          # Frontend UI
`-- .venv/                  # Local Python virtual environment, ignored by Git
```

## Required Model Files

The app needs these files in the same folder as `app.py`:

```text
best_model.keras
tokenizer.pkl
model_config.json
```

Without these files, Flask may start importing dependencies but the model cannot load.

## Local Setup on Windows

Use Python 3.12. TensorFlow is not available for the Python 3.14 environment that was accidentally created earlier.

```powershell
cd C:\Users\skshi\Desktop\WriteFusion-AI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Correct PowerShell Commands

Use Windows-style paths in PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not use this PowerShell path style:

```powershell
/c:/Users/skshi/Desktop/WriteFusion-AI/.venv/Scripts/python.exe
```

That format is not recognized as a PowerShell executable path.

## Dependency Notes

The app currently uses:

```text
Flask
TensorFlow
NumPy
Gunicorn
Protobuf
```

`requirements.txt` pins TensorFlow-compatible protobuf:

```text
tensorflow>=2.21.0
protobuf>=6.31.1,<8.0.0
```

This avoids the protobuf runtime error:

```text
Detected incompatible Protobuf Gencode/Runtime versions
```

## Important Runtime Warnings

Some TensorFlow warnings are normal on Windows CPU:

- oneDNN custom operations warning
- CPU instruction optimization warning
- GPU not available on native Windows warning
- Keras custom-layer build warnings
- optimizer variable loading warning

These warnings do not mean the app failed. The app is healthy if it prints the local URL and the page returns HTTP 200.

## Backend API Routes

| Route | Method | Purpose |
|---|---:|---|
| `/` | GET | Loads the main UI |
| `/api/model-info` | GET | Returns model metadata |
| `/api/complete` | POST | Returns next-word candidates and attention data |
| `/api/generate` | POST | Generates story/paragraph continuation |
| `/api/perplexity` | POST | Computes perplexity for supplied text |

## Decoding Strategies

The generation endpoint supports multiple strategies:

- Greedy
- Temperature
- Top-K
- Top-P / nucleus sampling
- Top-K + Top-P
- Typical sampling
- Min-P sampling
- Smart Anti-Repetition

Smart Anti-Repetition applies:

- repetition detection
- dynamic temperature adjustment
- dynamic Top-K adjustment
- repetition penalty over recent tokens
- no-repeat n-gram suppression

## Model Snapshot

The UI displays:

- parameters
- vocabulary size
- layers
- attention heads
- context length
- embedding dimension
- feed-forward width
- corpus words: 2.7M
- sentences: 860,173
- max generation words: 500

## Training and Model Development

The model-development section explains the full pipeline:

1. Data collection
2. Text cleaning and normalization
3. Tokenization
4. Vocabulary design
5. Sequence creation
6. Transformer architecture
7. Training configuration
8. Evaluation
9. Inference pipeline
10. Decoding controls
11. Story/paragraph generation
12. Saving and deployment

Training plots are loaded from `static/` with light and dark variants.

## Git Cleanup Notes

The repository should not track generated environments, caches, or logs. These are ignored:

```text
.venv/
venv/
.venv*/
__pycache__/
*.pyc
*.log
```

Large model files are also ignored by default:

```text
best_model.keras
*.keras
*.h5
*.hdf5
```

Keep model files locally for running the app. If you need to publish the model, use Git LFS or external storage.

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

You are using the wrong or broken virtual environment. Recreate `.venv` with Python 3.12 and reinstall requirements.

### `No matching distribution found for tensorflow`

You are probably using Python 3.14. Use Python 3.12:

```powershell
py -3.12 -m venv .venv
```

### Protobuf runtime mismatch

Install compatible protobuf in the active environment:

```powershell
python -m pip install "protobuf>=6.31.1,<8.0.0"
```

### Port already in use

Stop the old Flask process with `Ctrl + C` in the terminal where it is running, then run again.

## Run Checklist

Before running, confirm:

- `.venv` is Python 3.12.
- `numpy`, `tensorflow`, and `flask` are installed.
- `best_model.keras`, `tokenizer.pkl`, and `model_config.json` exist.
- `python app.py` prints `Open: http://localhost:5000`.
