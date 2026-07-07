# testdrive

**One CLI. Many vision foundation models.**

`testdrive` is a plugin-based command-line tool for trying out
open-vocabulary object-detection (and detection-adjacent) models. The
framework knows nothing about any specific model — every detector is a
self-contained plugin file in `testdrive/models/`. Drop in a new
adapter file and it shows up automatically; no framework code ever
needs to change.

> **Status:** v0.1.

## Active plugins

| id | model | notes |
|---|---|---|
| `groundingdino` | Grounding DINO | open-vocabulary detection |
| `owlv2` | OWLv2 | open-vocabulary detection, good multi-instance recall |
| `owlvit` | OWL-ViT | predecessor to OWLv2; needs a lower confidence threshold — see below |
| `florence2` | Florence-2 | phrase-grounding via `<CAPTION_TO_PHRASE_GROUNDING>` |
| `samgd` | Grounding DINO + SAM | GDINO box, refined to a tight mask-derived box by SAM |

Two plugins are parked in `testdrive/models/_inactive/` (not
discovered by the framework, so they don't appear in `-L`/`-I`/etc):

- **`seem`** — there is no `transformers`-compatible SEEM model. The
  only HF repo for it ships raw checkpoint files with no
  `config.json`/processor/modeling code — nothing for `AutoModel*` to
  load, `trust_remote_code=True` or not. Making it real would mean
  vendoring the original research repo's own loader, which is out of
  scope here.
- **`molmo` / `molmo7b`** — Molmo points at things rather than drawing
  boxes, so its output is approximated into small boxes centered on
  each point. The dense 7B model (`molmo7b.py`) didn't reliably finish
  a self-test on an 8-core CPU even in bfloat16; a lighter MoE variant
  was tried as the active `molmo.py` but is also currently parked
  pending further testing. See each file's module docstring for the
  full reasoning — both are kept, not deleted, in case your hardware
  (or a future transformers/molmo release) changes the calculus.

## The framework's own environment

testdrive insists on running from one dedicated virtual environment,
`cache/pyenv/framework` — not because of a preference, but as a
deliberate guard against dependency hell as more plugins accumulate
(this project already hit that once for real: a `transformers` upgrade
pulling in a TensorFlow chain that broke every plugin at once). If
you start testdrive from anywhere else, it transparently relaunches
itself from that environment if it already exists, or — on a fresh
checkout, before that environment exists yet — stops and tells you
exactly how to create it:

```bash
python3.12 -m venv cache/pyenv/framework            # py -3.12 ... ...\...\... on Windows
cache/pyenv/framework/bin/pip install -e .          # cache\...\Scripts\pip.exe on Windows
source cache/pyenv/framework/bin/activate           # cache\...\Scripts\activate on Windows
```

After that one-time setup, just run `testdrive` normally — the
self-relaunch is invisible from then on. (Set
`TESTDRIVE_SKIP_PYENV_CHECK=1` to bypass this entirely; used by CI's
core-only smoke test, not intended for normal use.)

Each plugin's manifest also carries a `pyenv` field (`-M <plugin>`),
currently always `"framework"` — the mechanism for a plugin to
actually get its own isolated environment (for a dependency set that's
incompatible with everything else) is tracked as follow-up work, not
built yet. See `testdrive/pyenv.py`'s module docstring for the reasoning.

## Installation

```bash
git clone https://github.com/<you>/testdrive.git
cd testdrive
pip install -e .

# Install backend deps for the plugin(s) you want to run:
pip install -e ".[owlv2]"          # one plugin
pip install -e ".[owlv2,owlvit]"   # a few
pip install -e ".[all]"            # every active plugin
```

`transformers` is pinned to exactly `4.50.3` in every extra — every
later release we've tried introduces a regression that pulls in a
TensorFlow dependency chain, which then breaks these plugins. Revisit
this once that's fixed upstream.

**The core framework needs only `Pillow`.** `-L`, `-I`, `-M` (including
`-M --json`) all work with nothing else installed — verified by running
them with `torch`/`transformers`/`huggingface_hub`/`numpy` all blocked.
Anything beyond that (`-T`, `-TT`, or plain detection) needs the
relevant plugin's own extra installed, and fails with a clear
"missing dependency" message rather than a traceback if it isn't.

## Usage

```bash
# List discovered plugins
testdrive -L

# Show full manifest (id, license, requirements, sample prompt, …)
testdrive -M owlv2
testdrive -M '*'          # every plugin
testdrive -M owlv2 --json

# Check what's installed, without loading any models
testdrive -I

# Populate the cache and verify a plugin end-to-end (synthetic image)
testdrive -T owlv2
testdrive -T '*'          # every plugin

# A stronger check: run each plugin's own example image and verify the
# expected number of matches, not just "did it run"
testdrive -TT owlv2       # same as: testdrive -T -T owlv2
testdrive -TT '*'

# Run detection
testdrive owlv2 photo.jpg "person, bicycle"

# <image> can also be a directory — every file in it is processed,
# except files matching our own *-matches.*/*-redacted.* output pattern
# (so re-running over an already-processed folder is safe)
testdrive owlv2 ./photos "person"

# <plugin> can be '*' — every discovered plugin runs against the same
# image/prompt (or every image, if <image> is also a directory: full
# cross product). Output filenames get "-<plugin id>" appended when more
# than one plugin is involved, so results never collide.
testdrive '*' photo.jpg "person"
testdrive '*' ./photos "person"

# Send all output somewhere else instead of next to the input image(s)
testdrive owlv2 ./photos "person" --output-dir ./results

# Cap parallel downloads (helpful on a slow link — Molmo-class models
# ship several multi-GB shards, and downloading them all at once just
# divides the same bandwidth further rather than finishing faster)
testdrive -T owlv2 --max-parallel-files 1

# Machine-readable output
testdrive -T owlv2 --json
testdrive owlv2 photo.jpg "cat" --json
```

Quote `'*'` on shells that glob it (most shells).

Output images are written alongside the source image by default (or
under `--output-dir` if given):

| File | Contents |
|---|---|
| `photo-matches.png` | Original with green bounding boxes + labels |
| `photo-redacted.png` | Original with black-bar redactions |

### Cache discipline

A plain detect run never silently downloads a model. If a plugin's
weights aren't cached yet, the run fails fast with a message pointing
you at `-T`/`-TT` instead of blocking on a multi-gigabyte download in
the middle of what looked like a normal command — which, stacked on a
naturally slow model's already-long init time, is easy to mistake for
a hang. Run `testdrive -T <plugin>` (or `-TT`) once first; after that,
plain detect runs against that plugin are unaffected either way.

Cached weights live under a plain, non-symlinked directory (default:
`./cache`, override with the `TESTDRIVE_CACHE` env var) rather than
`huggingface_hub`'s usual symlinked snapshot cache — the latter is
unreliable on some Windows installs and consistently broken under Wine.

## Prompt syntax

Separate multiple labels with commas:

```
"person"
"person, bicycle, car"
"cat, dog"
```

## Exit codes

| Code | Meaning              |
|------|----------------------|
| 0    | success              |
| 1    | CLI error            |
| 2    | plugin not found     |
| 3    | missing dependency (includes: model not cached, run -T/-TT first) |
| 4    | inference failed (includes: -TT match count mismatch) |
| 5    | image unreadable     |
| 6    | output write failed  |
| 7    | loop mode (`'*'`): at least one plugin/image/run in the loop failed |

## Examples directory

`testdrive/examples/<plugin>/image1-prompt-<slug>-<N>matches.png` — one
small synthetic (not real-photo, to avoid licensing questions) example
per active plugin, named so `-TT` can parse the prompt and expected
match count straight out of the filename. See
`testdrive/examples/README.md` for the full naming convention and how
to add your own (real-photo) examples for actual accuracy comparisons.

## Writing a new plugin

A plugin is one file in `testdrive/models/`:

```python
from ..detection import Detection
from ..plugin import DetectorPlugin

PLUGIN_API = 1

PLUGIN = {
    "id": "mymodel",
    "name": "My Model",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "...",
    "author": "...",
    "homepage": "...",
    "license": "Apache-2.0",
    "license_url": "...",
    "backend": "transformers",
    "hf_repo": "org/mymodel",
    "task": "...",
    "supports": ["text prompts"],
    "requirements": [
        {"pip": "torch",        "module": "torch"},
        {"pip": "transformers", "module": "transformers"},
        {"pip": "Pillow",       "module": "PIL"},
    ],
    "sample_prompt": "person",
    "test_threshold": "default",   # or e.g. "0.05" if -TT needs a
                                    # different threshold than plain
                                    # detect's 0.3 default for this model
}

class Plugin(DetectorPlugin):
    def initialize(self) -> None:
        # Load weights once. Use util.load_processor()/load_model()
        # (which route through the shared local-snapshot cache) rather
        # than calling transformers' from_pretrained() directly.
        ...

    def detect(self, image, prompt, threshold=0.3) -> list[Detection]:
        return [Detection(label="cat", score=0.93, bbox=(10, 10, 80, 90))]
```

`testdrive -L` picks it up automatically. `testdrive -T mymodel` runs
the self-test to verify the installation end-to-end. Add
`testdrive/examples/mymodel/image1-prompt-<slug>-<N>matches.png` to
also get `-TT mymodel` for free.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -v
```

The test suite doesn't require any plugin's backend deps (`torch`,
`transformers`, etc.) — everything that would otherwise touch a real
model is mocked.

## License

The testdrive framework is MIT-licensed. Each plugin wraps a
third-party model distributed under its own license — run
`testdrive -M <plugin>` to see it before use.
