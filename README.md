# Learning face detection

Build a face detector step by step using
[the learning guide](Face_Detection_Codex_Learning_Guide.md).
The eventual input is an image; outputs will be an annotated copy and structured
face rectangles. Detection locates faces; it does not identify people.

## Current status: Session 2

Pretrained face detection now saves an annotated image and JSON predictions.
The manual rectangle exercise is complete. Inference is verified on a sample
photo and the empty practice grid; no training has run.
Read the [session documentation](docs/README.md) for detailed notes, code-flow
and architecture diagrams, exercises, and change records:

- [Session 0 — environment and the mental model](docs/session0/README.md)
- [Session 1 — images and manual rectangles](docs/session1/README.md)
- [Session 2 — pretrained face detection](docs/session2/README.md)

See [progress](docs/progress.md) for the current checkpoint. Session documents
are the detailed teaching reference; commands below are a quick-start guide.

## Environment

Verified on macOS 26.5.2, arm64, Apple M4, Python 3.14.5.
The local `.venv` contains Pillow 12.3.0, NumPy 2.5.3, and OpenCV Python
4.14.0.94, pinned in `requirements.txt`. Inference uses the CPU.

From this project folder, verify the interpreter:

```sh
.venv/bin/python -c 'import sys; print(sys.executable); print("Isolated environment:", sys.prefix != sys.base_prefix)'
```

Expected: this project's `.venv/bin/python` path and `Isolated environment: True`.
Both have been observed. Activation is optional with these explicit paths.

For a fresh setup only:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If environment creation ends with `^C` and `KeyboardInterrupt` during
`ensurepip`, setup was interrupted while preparing pip. Check the interpreter
and `.venv/bin/python -m pip --version` first. If both work, no recreation is
needed. This was verified after the Session 0 interruption.

## Run the image lesson

A synthetic grid is already available locally at `data/example.png`. To create
it in a fresh checkout, run this once (existing files are preserved):

```sh
.venv/bin/python src/make_example.py
```

Draw a box on a new copy:

```sh
.venv/bin/python src/image_basics.py data/example.png --box 100 80 300 240 --output outputs/my_rectangle.png
```

Expected and observed with the same box in `outputs/rectangle.png`:

```text
Width: 640; height: 480
Channels: ('R', 'G', 'B')
Array shape (height, width, channels): (480, 640, 3)
Data type: uint8 (whole numbers from 0 to 255)
Top-left pixel RGB: [200, 200, 200]
Center pixel RGB: [0, 100, 255]
Box xyxy: [100, 80, 300, 240]; area: 32000
```

The final line reports your output path. Open that PNG to see the red outline.
Each run needs a new output filename; the script refuses to overwrite files.
You can substitute a local photo and a box within its dimensions. The script
applies camera orientation and converts to RGB before reporting dimensions.

The example is a programmatically generated coordinate grid, not a photo or
training sample. The box is manually chosen, not detected.

## Completed Session 1 exercise

Add 80 to the left and right coordinates; keep top and bottom unchanged.
Predict how position and size will change before running:

```sh
.venv/bin/python src/image_basics.py data/example.png --box 180 80 380 240 --output outputs/experiment.png
```

The learner confirmed that the rectangle moved 80 pixels right without
changing size. Actual training follows labeled-data preparation later.

Local environments, inputs, model weights, and generated outputs are excluded
by `.gitignore`. The selected inference dependencies have been tested on Python 3.14.5.
Training-framework compatibility will be checked in its later session.


## Session 2: run pretrained face detection

The local model and public-domain sample photo have already been downloaded.
For a fresh checkout, install requirements as above, then run:

```sh
.venv/bin/python src/download_assets.py
```

This downloads the official YuNet model and scikit-image's public-domain NASA
sample, using the exact URLs and SHA-256 fingerprints in `docs/assets.json`.
Existing matching files are reused; differing files are left untouched.
See [model decisions and licenses](docs/decisions.md) and the retained
[YuNet MIT notice](docs/yunet-LICENSE.txt).

Run inference:

```sh
.venv/bin/python src/detect_faces.py data/astronaut.png --model models/face_detection_yunet_2023mar.onnx --output-dir outputs/my_first_detection
```

Expected: one face at the default threshold 0.8, plus these files:

- `outputs/my_first_detection/annotated.png`: original-size image with a box.
- `outputs/my_first_detection/predictions.json`: dimensions, detections, scores,
  and the settings used.

Observed with the same command using `outputs/astronaut_baseline`: one face,
box approximately `[178.17, 62.91, 268.07, 176.24]`, score `0.9323`.
The score is not a measured probability of correctness.

Use a new output directory each run; existing directories are preserved.
To use your own image, replace `data/astronaut.png` with its path. Everything
runs locally; the detector makes no network requests. Phone orientation is
applied before prediction and output; coordinates refer to that oriented image.
Large inputs shrink proportionally to a maximum side of 640 for inference;
boxes are mapped back to the original size. `--max-size` adjusts that limit.

To try an image with no faces:

```sh
.venv/bin/python src/detect_faces.py data/example.png --model models/face_detection_yunet_2023mar.onnx --output-dir outputs/my_grid_detection
```

Observed: zero faces, an unchanged-looking output image, and `"detections": []`.
This is a successful run. No face is not the same as a program failure.

### Your checkpoint

Run the photo command and inspect both saved files. Predict what increasing
its threshold to 0.95 will do, then try:

```sh
.venv/bin/python src/detect_faces.py data/astronaut.png --model models/face_detection_yunet_2023mar.onnx --threshold 0.95 --output-dir outputs/my_threshold_experiment
```

Explain the difference before we start Session 3. No weights are updated by
these commands. The later training workflow is described in the decisions doc.

### Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Five offline tests passed for coordinate mapping, clipping, and validation.
Actual inference was also verified at original and reduced sizes, with outputs
visually inspected. Missing/corrupt input, invalid model, and existing output
directories were checked separately. These are software checks, not an accuracy
benchmark. There is no labeled evaluation or training result yet.
# object-detection
