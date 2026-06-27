# SwarGAN Studio — Singing Voice Style Transfer

An interactive Streamlit application for singing-voice processing built on an
AutoVC-style voice-conversion model, a WORLD/librosa feature pipeline, a
Griffin-Lim vocoder, and Demucs-based vocal separation.

## Features

| Page | Description | Status |
|------|-------------|--------|
| **Voice Separation** | Extract vocals/stems from an upload or a URL (YouTube, SoundCloud, …) using Demucs. | Working (requires `ffmpeg`, `demucs`, `yt-dlp`). |
| **Audio Processing** | Upload source/target clips; extract mel-spectrogram + F0 features. | Working. |
| **Model Training** | Train the AutoVC model on uploaded clips. | Working (autoencoding + code-consistency objective). |
| **Voice Conversion** | Convert a source clip toward a target speaker's timbre, then vocode to audio. | Runs end-to-end. Quality depends on a trained model. |
| **Analysis & Visualization** | Mel-spectrogram and F0 plots + statistics. | Working. |
| **Model Information** | Model/parameter/system info. | Working. |

## Architecture

```
Source mel ──► ContentEncoder ──► content code ─┐
                                                 ├─► Decoder ─► Postnet ─► mel ─► Vocoder ─► audio
Target mel ──► SpeakerEncoder ─► speaker embed ──┘
```

- **ContentEncoder** — conv stack + bidirectional LSTM producing a low-dim bottleneck code.
- **SpeakerEncoder** — conv stack + BiLSTM + temporal mean-pool, L2-normalized timbre embedding.
- **Decoder + Postnet** — reconstruct and refine the mel-spectrogram.
- **AutoVCLoss** — `recon (pre-postnet) + recon (post-postnet) + L1 code-consistency`. The
  code-consistency term re-encodes the output and matches its content code to the
  source's, which is what makes content-preserving conversion possible.
- **Vocoder** — `SimpleVocoder` (Griffin-Lim, default) or an untrained `NeuralVocoder` placeholder.

Key parameters live in [`config.py`](config.py) (sample rate 22.05 kHz, 80 mel bins, etc.).

## Setup

Requires Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch (CPU build shown; pick the right index for your platform/GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Core audio/ML stack
pip install numpy scipy librosa soundfile pyworld matplotlib streamlit

# Optional: vocal separation + URL download
pip install demucs yt-dlp
# Demucs and yt-dlp also require ffmpeg on your PATH:
#   Debian/Ubuntu: sudo apt-get install ffmpeg
#   macOS:         brew install ffmpeg
```

## Running the app

```bash
streamlit run app.py --server.port 5000
```

Then open the URL Streamlit prints (default `http://localhost:5000`).

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -q
```

The suite covers audio validation, the feature pipeline, the AutoVC model/loss,
the vocoder, and an end-to-end integration test. Heavy/network-dependent paths
(Demucs separation, URL download) are not exercised by the unit tests.

## Logging

All modules log under the `swargan` namespace. Control verbosity with:

```bash
export SWARGAN_LOG_LEVEL=DEBUG   # or INFO (default), WARNING, ...
```

## Running the API service

The conversion pipeline is also exposed as an async HTTP API (FastAPI):

```bash
pip install fastapi uvicorn python-multipart
uvicorn service.app:create_app --factory --port 8000
```

Endpoints:
- `GET /health` — liveness + available engines/voices
- `GET /voices` — the offered (licensed) target voices
- `POST /convert` — multipart upload (`file`, `voice_id`, optional `separate`, `engine`) → returns a job
- `GET /jobs/{id}` — poll job status
- `GET /jobs/{id}/result` — download the converted audio when ready
- `POST /convert/sync` — synchronous conversion (blocking) that returns the audio directly; for short clips

Or run it in a container:

```bash
docker compose up --build   # serves on http://localhost:8000
```

The service enforces a **licensing guardrail**: conversion to a voice that is
not marked `licensed` is rejected (HTTP 403) unless `allow_unlicensed` is set.
The conversion backend is pluggable via `engine.ConversionEngine` — the default
is the in-repo AutoVC engine, and `engine.ExternalCommandEngine` wraps any CLI
SVC tool (RVC / so-vits-svc / DDSP-SVC) without changing the service layer.
See [`docs/SVC_BACKENDS.md`](docs/SVC_BACKENDS.md) for wiring in a real backend.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design,
model direction, roadmap, and data/rights strategy.

## Known limitations

- **No pretrained checkpoint is shipped.** Out of the box the model is randomly
  initialized, so conversion output will not resemble a real voice until you train
  a model (Model Training page) or load your own checkpoint into `checkpoints/final_model.pth`.
- The `NeuralVocoder` is an untrained placeholder; use `griffin_lim` for usable audio.
- Mel-spectrograms are kept in dB scale end-to-end; normalizing them is a reasonable
  future improvement for faster training convergence.
- Vocal separation requires `ffmpeg`, `demucs`, and `yt-dlp` to be installed.
