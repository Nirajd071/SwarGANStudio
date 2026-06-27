# Plugging in a real SVC backend (RVC / so-vits-svc)

The in-repo `AutoVCEngine` is a runnable **baseline**. For production-quality
singing-voice conversion, swap in a mature open-source backend via the
`ExternalCommandEngine` abstraction — no changes to the API, jobs, or storage.

This guide shows how to wire in **so-vits-svc** or **RVC**.

> Prerequisites: a machine with a GPU, the backend tool installed, and a
> **trained model per target voice** (train/fine-tune on a dataset produced by
> `data_pipeline/`). Also ensure you hold the rights to each voice — the service
> only offers voices marked `licensed`.

## 1. Install a backend

Pick one (examples; follow each project's own install instructions):

- **so-vits-svc** — e.g. the `so-vits-svc-fork`, which exposes an `svc` CLI:
  ```bash
  pip install so-vits-svc-fork
  # inference: svc infer INPUT -m MODEL.pth -c config.json -s SPEAKER -o OUT.wav
  ```
- **RVC** — e.g. `rvc-python`, which exposes an `rvc` CLI:
  ```bash
  pip install rvc-python
  # inference: rvc infer -i INPUT -o OUT.wav -mp MODEL.pth -ip MODEL.index -me rmvpe
  ```

## 2. Register the backend + voices

```python
from engine.registry import default_registries
from engine.svc_templates import make_so_vits_svc_engine, make_rvc_engine
from engine.base import VoiceProfile
from service.app import create_app

voices, engines = default_registries()

# so-vits-svc backend
engines.register(make_so_vits_svc_engine(
    model_path="/models/rafi/G.pth",
    config_path="/models/rafi/config.json",
    speaker="rafi",
), default=True)

# (or) an RVC backend
engines.register(make_rvc_engine(
    model_path="/models/kishore/model.pth",
    index_path="/models/kishore/model.index",
    f0_method="rmvpe",
))

# Offer the (licensed!) voices
voices.register(VoiceProfile(id="rafi", name="Rafi (licensed)", licensed=True))
voices.register(VoiceProfile(id="kishore", name="Kishore (licensed)", licensed=True))

app = create_app(voices=voices, engines=engines)
```

Run it: `uvicorn service.app:create_app --factory` (or `swargan-serve`), then
`POST /convert` with `engine=so-vits-svc` (or `rvc`) and the chosen `voice_id`.

## 3. How the template mapping works

The factories leave two runtime placeholders that the engine fills per request:

| Placeholder | Filled with |
|---|---|
| `{source}` | the input vocals wav written by the engine |
| `{output}` | the path the tool must write the converted wav to |

For fully custom CLIs, `make_python_script_engine(..., extra_args=[...])` also
exposes `{target}` (voice reference audio), `{voice_id}`, `{model}`, and any key
from `VoiceProfile.metadata` (e.g. `{config}`, `{speaker}`, `{index}`).

## 4. Adjusting flags

CLI flags differ across forks/versions. If a command fails, run the tool's
`--help`, then tweak the template (or pass `extra_args=[...]`) so it matches your
installation. The engine surfaces the tool's stderr in the raised error to make
debugging straightforward.

## 5. Recommended quality stack

Per the architecture notes, the strongest current recipe is self-supervised
content features (ContentVec/Whisper) + explicit F0 (RMVPE) + an NSF-HiFiGAN
vocoder, fine-tuned per voice. so-vits-svc and RVC both follow this shape.
