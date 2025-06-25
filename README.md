# Interview Analyzer (Whisper GUI)

A desktop tool for transcribing and analyzing interview videos or audio files using OpenAI Whisper, with a modern Flet-based GUI.

---

## Features

- **Audio/Video Upload**: Supports `.mp3`, `.wav`, `.m4a`, `.mp4`, `.avi`, `.mov`.
- **Automatic Speech-to-Text**: Extracts audio from videos and transcribes using OpenAI Whisper.
- **Chunk Processing**: Handles long recordings by splitting into chunks.
- **Model Selection**: Choose Whisper model size (base, small, medium, large).
- **Timestamped Transcript**: Outputs text with timecodes for each segment.
- **GUI**: Fast, user-friendly interface (built with [Flet](https://flet.dev/)).
- **Cache Management**: Delete Whisper model cache from the GUI.
- **Export**: Download your transcript as a `.txt` file.

---

## Installation

### ⚠️ Python Version
⚠️ Whisper requires Python 3.8 ~ 3.11.
> Please ensure you are using Python 3.11 (recommended for best compatibility).

You can check your version:
```sh
python --version
```

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/interview_analyzer.git
cd interview_analyzer
```
### 2. Create a Virtual Environment
```sh
python -m venv .venv
# Activate:
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Python Dependencies
```sh
pip install -r requirements.txt
```
⚠️ NOTE:
torch, torchvision, and torchaudio are NOT included in requirements.txt and must be installed manually according to your environment:

CUDA 12.1 + Python 3.11 Example:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Or use a direct link to the proper wheel, e.g.:
```sh
pip install torch @ https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp311-cp311-win_amd64.whl
```
See PyTorch official installation instructions for details.

Usage
Run the Application
```sh
python main.py
The GUI will open. Select your audio or video file, choose the model and chunk size, and click “Transcribe”.
```

Build as Standalone EXE (Windows)
You can use PyInstaller to package as a desktop app:

```sh
pip install pyinstaller
pyinstaller --onefile --noconsole main.py
The .exe will appear in the dist folder.
```

.gitignore Example
```sh
gitignore
.vscode/
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv/
dist/
build/
*.spec
audio/
transcript/
.cache/
*.log
*.tmp
```
License
MIT License

Acknowledgments
OpenAI Whisper

Flet UI

moviepy

pydub
