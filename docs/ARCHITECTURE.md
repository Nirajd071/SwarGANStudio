# SwarGAN Studio — Architecture & Strategy

This document describes the platform architecture, how the current code maps to
it, and the roadmap. It is intended to be read alongside the top-level
[`README.md`](../README.md).

## 1. Vision

> Upload a song → separate vocals → understand melody/lyrics/style → recreate the
> song in another (licensed) singer's voice → output a studio-quality track,
> preserving lyrics, melody, rhythm and emotion while changing only the singer
> identity.

The platform is built as four cooperating systems plus a serving layer.

## 2. System map

```
                 ┌─────────────────────────────────────────────────────────┐
 Upload / URL ──▶│ 1. Audio Understanding   (Demucs separation, features)   │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ clean vocals
                 ┌───────────────▼─────────────────────────────────────────┐
                 │ 2. Singer Intelligence   (target voice profile / embed)  │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ + target voice
                 ┌───────────────▼─────────────────────────────────────────┐
                 │ 3. Voice Transformation  (ConversionEngine: AutoVC/RVC…) │
                 └───────────────┬─────────────────────────────────────────┘
                                 │ converted vocals
                 ┌───────────────▼─────────────────────────────────────────┐
                 │ 4. Audio Reconstruction  (vocoder + remix + master)      │
                 └───────────────┬─────────────────────────────────────────┘
                                 ▼
                          Final converted song
```

## 3. How the code maps today

| System | Module(s) | Status |
|---|---|---|
| 1. Audio understanding | `utils/vocal_separator.py` (Demucs), `utils/feature_extractor.py`, `data_pipeline/` | Working; data pipeline is model-agnostic dataset prep. |
| 2. Singer intelligence | `engine/base.py::VoiceProfile`, `engine/registry.py::VoiceRegistry` | Per-voice profiles + reference audio. Zero-shot embedding is future work. |
| 3. Voice transformation | `engine/` (`ConversionEngine`, `AutoVCEngine`, `ExternalCommandEngine`) | Pluggable. AutoVC reference runs; RVC/so-vits-svc via external adapter. |
| 4. Audio reconstruction | `models/vocoder.py` (Griffin-Lim), `service/conversion.py` (`_remix`) | Baseline. NSF-HiFiGAN is the planned upgrade. |
| Serving | `service/` (FastAPI + async jobs) | `/convert` (async), `/convert/sync`, `/jobs`, `/voices`. |
| Data pipeline | `data_pipeline/` | download → separate → clean → segment → manifest. |

## 4. The engine abstraction (key design decision)

The product flow is **decoupled from the model** via `ConversionEngine`:

```python
class ConversionEngine:
    def convert(self, source_audio, sr, voice) -> (audio, sr): ...
```

- `AutoVCEngine` — in-repo baseline (runs today; quality needs a trained model).
- `ExternalCommandEngine` — wraps any CLI SVC tool (RVC, so-vits-svc, DDSP-SVC)
  out-of-process via a command template.

A stronger backend can be added without changing the API, jobs, or storage.

## 5. Recommended model direction (informed by current SOTA)

AutoVC (bottleneck disentanglement) is a 2019 speech technique and is the
*baseline*, not the target. Modern singing-voice conversion combines
self-supervised content features (ContentVec/HuBERT/Whisper) with **explicit
F0/pitch** and a **Neural Source-Filter HiFi-GAN** vocoder, optionally with a
diffusion or flow-matching decoder. For v1, fine-tune a mature open-source
stack (RVC / so-vits-svc) per voice rather than training zero-shot from scratch.

## 6. Serving architecture

- Stateless FastAPI app (`create_app` factory; DI for testing).
- Async jobs via an in-process thread pool (`service/jobs.py`) with a minimal
  create/submit/get interface — deliberately swappable for Celery/RQ/SQS +
  GPU workers and object storage for production.
- A synchronous `/convert/sync` endpoint exists for short clips / simple clients.
- Local filesystem storage today (`service/storage.py`); swap for S3/GCS in prod.

## 7. Roadmap

1. **Phase 0 (done)** — stabilize the baseline pipeline; build data pipeline;
   define engine abstraction + serving contract.
2. **Phase 1** — integrate a real SVC backend (RVC/so-vits-svc) + NSF-HiFiGAN
   via `ExternalCommandEngine`; assemble 3–5 licensed/original voices.
3. **Phase 2** — quality: de-reverb/cleaning, loudness mastering, robustness to
   messy uploads; production queue + object storage; web front-end.
4. **Phase 3** — singer-embedding model for few-shot voices; voice marketplace.

## 8. Data strategy

Per voice, target several hours of **clean, isolated** vocals. The bottleneck is
data quality, not model code. The `data_pipeline/` package automates
download → Demucs separation → cleaning → voiced-segment filtering → manifest,
and is reusable across model stacks.

## 9. Rights & licensing (non-negotiable)

Cloning real artists implicates sound-recording copyright, musical-composition
copyright, and personality/publicity rights (including estates of deceased
artists). The platform therefore enforces a **licensing guardrail in code**:
`VoiceProfile.licensed` gates conversion, and the API refuses unlicensed voices
(HTTP 403) unless explicitly overridden. v1 should ship only original/session
or licensed voices, with consent + provenance tracked from day one.
