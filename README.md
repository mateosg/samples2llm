<div align="center">

# 🎧 samples2llm

**Turn a folder of audio samples into AI-friendly JSON or Markdown.**

Structured context — for LLM-powered sound search, tagging, and production workflows.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
![Tests](https://img.shields.io/badge/tests-198%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
[![License](https://img.shields.io/badge/license-unlicensed-lightgrey)](#-license)

</div>

<hr />

`samples2llm` scans a sample library and exports a single structured file — JSON or Markdown — describing every sample's **measured audio characteristics** (dynamics, spectral, tonal, rhythmic, spatial) alongside **declared metadata** parsed from its path and filename (BPM, key, type).

Feed that file to any LLM and it can reason precisely about your library — including compound production queries like:

> *"Find 5 risers that fit a 128 BPM, A minor progressive house drop, with high energy growth and a wide stereo image in the final bar."*

No more auditioning 40 files by ear to find the one that fits.

## Table of Contents

- [Why this project](#-why-this-project)
- [Features](#-features)
- [Quickstart](#-quickstart)
- [CLI reference](#-cli-reference)
- [Example output](#-example-output)
- [Output model](#-output-model)
- [Cache behavior](#-cache-behavior)
- [Project structure](#-project-structure)
- [Known limitations & roadmap](#-known-limitations--roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🤔 Why this project

The bottleneck in production usually isn't *finding* samples — it's finding the right one inside a folder that already has hundreds of them. "A dark riser building into a 128 BPM drop, wide stereo, no clipping" normally means previewing file after file by ear until one fits.

`samples2llm` turns a sample folder into a single file you can hand to an LLM and simply ask which sample fits — and get an answer grounded in what the audio actually measures, not a guess based on filenames. Instead of auditioning a whole `Risers/` folder, you describe what you need and get candidates back.

## ✨ Features

- 📄 **JSON and Markdown output** — pick whichever fits your pipeline
- 🌳 **Directory tree** included once at the top level for spatial context
- 🔬 **Per-sample descriptors** across dynamics, spectral, tonal, rhythmic, and spatial dimensions
- ⚠️ **Confidence flags** whenever declared and measured values disagree
- ⚡ **Persistent hash cache** for fast repeated runs on large libraries
- 🧪 **Strong automated test suite** — synthetic and real-audio fixtures

## 🚀 Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Run on a sample folder

```bash
samples2llm ./my-sample-library --style json -o samples.json
```

Not on your `PATH`? Run it as a module instead:

```bash
python -m samples2llm.cli ./my-sample-library --style json -o samples.json
```

### 3. Markdown output

```bash
samples2llm ./my-sample-library --style markdown -o samples.md
```

### 4. Try it on the bundled examples

```bash
samples2llm examples/sample_library --style markdown
```

## 📖 CLI reference

```bash
samples2llm <directory> [--style json|markdown] [-o output_file] [--no-tonal] [--no-rhythmic]
```

| Option | Description |
|---|---|
| `directory` | Sample folder to process (required) |
| `--style` | Output format: `json` or `markdown` |
| `-o, --output` | Write output to a file instead of stdout |
| `--no-tonal` | Disable pitch/tonal extraction (faster) |
| `--no-rhythmic` | Disable tempo/rhythmic extraction |

## 🧾 Example output

Running `samples2llm examples/sample_library --style markdown` against the bundled example library produces:

```markdown
# Sample Library

## Directory Structure

Bass/
  Sub/
    Bass_Sub_Amin_128bpm_v2.wav
Drums/
  Kicks/
    Kick_808_hit.wav
FX_whoosh_descending.wav
Perc_multihit_layered.wav
Texture_wide_stereo.wav
loop_Cmaj_120bpm.wav

## Samples

| Path | Summary | Tags | Flags |
|---|---|---|---|
| Bass/Sub/Bass_Sub_Amin_128bpm_v2.wav | Loop (2.0s), medium attack, low brightness, tonal, pitch ~A2. | loop, tonal, low_frequency | declared_bpm_mismatch_with_detected_bpm |
| Drums/Kicks/Kick_808_hit.wav | One-shot (0.4s), fast attack, high brightness, percussive, no stable pitch. | one_shot, single_hit, percussive | - |
```

Note the `declared_bpm_mismatch_with_detected_bpm` flag on the bass sample: the filename claims 128 BPM, but the measured tempo disagreed — exactly the kind of signal an LLM (or you) can act on.

## 🗂 Output model

Each sample record includes:

- `path`, `filename`, `directory_context`
- `declared` — key/BPM/type hints parsed from names
- `file_metadata`
- `dynamics`, `spectral`, `tonal`, `rhythmic`, `spatial`
- `integrity`
- `derived_tags`
- `confidence_flags`
- `summary`

Full schema: [`src/samples2llm/config/schema.py`](src/samples2llm/config/schema.py)

## 💾 Cache behavior

Cache file: `.samples2llm.cache.json`, stored in the analyzed root folder.

A cached entry is reused only when **both** are unchanged:

- File fingerprint (size + mtime + SHA-256)
- Extraction config signature

Stale entries are pruned automatically on each run.

On a 24-file commercial subset, this took the run time from `111.5s` (cold) to `6.0s` (warm) — an ~18.6x speedup.

## 📁 Project structure

```text
src/samples2llm/
  cli.py
  config/
  core/
    cache.py
    context_parse.py
    extract.py
    packager.py
    sample_search.py
    summarize.py
    tree_generate.py
    features/
    output/
  shared/

tests/
examples/
```

## 🚧 Known limitations & roadmap

| Limitation today | Planned |
|---|---|
| Extraction is sequential (`extract.py`) | Parallel processing for large libraries |
| No `.gitignore`-style ignore parsing yet | `.gitignore` / `.samplesignore` support |
| Type-hint dictionary is intentionally compact | Richer context parser, more naming conventions |
| No embeddings/MFCC export | Optional embedding layer for similarity search |

## 🤝 Contributing

Contributions are welcome.

1. Fork the repo
2. Create a branch
3. Add or update tests
4. Run the checks below
5. Open a PR with a clear change summary

```bash
python -m pytest -q
python -m pytest --cov=src/samples2llm --cov-report=term-missing -q
```

The suite covers both synthetic fixtures and real audio (see `tests/` and `examples/esc50_real/`), currently at 198 tests / 98% coverage — run the commands above to reproduce.

## 📜 License

No `LICENSE` file is defined yet in this repository. If you plan to publish it publicly on GitHub, adding one (MIT is a common choice for tooling like this) before the first push is recommended — say the word and I'll add it.

## 🙏 Acknowledgements

- Built on the audio/DSP ecosystem: `librosa`, `soundfile`, `numpy`, `pydantic`, `typer`

<p align="center"><a href="#-samples2llm">⬆ Back to top</a></p>
