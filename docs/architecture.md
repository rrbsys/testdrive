# testdrive Architecture Proposal

## Vision

`testdrive` should act as a **"VLC for vision models"**: one CLI/API
capable of loading and running many computer vision and vision-language
models through a common plugin interface.

## Design goals

-   Zero core changes when adding a model.
-   One plugin = `models/<name>.py`
-   One metadata file = `configs/<name>.yaml`
-   Automatic discovery.
-   Unified CLI and Python API.
-   Backend agnostic (PyTorch, ONNX Runtime, OpenVINO, TensorRT,
    CoreML...).
-   Capability-driven rather than model-driven.

## Repository layout

``` text
testdrive/
  models/
    yolo11.py
    groundingdino.py
    florence2.py
    sam2.py

  configs/
    yolo11.yaml
    sam2.yaml

  backends/
    pytorch.py
    onnxruntime.py
    openvino.py

  capabilities/
    detection.py
    segmentation.py
    vlm.py
```

## Plugin contract

``` python
class Model:
    name = "yolo11"

    def load(self, cfg): ...
    def predict(self, image, **kwargs): ...
    def capabilities(self): ...
```

## YAML metadata

``` yaml
id: yolo11
backend: pytorch
weights: yolo11x.pt
tasks:
  - detection
classes: coco
```

## Capability model

A plugin advertises capabilities such as:

-   detection
-   segmentation
-   classification
-   OCR
-   pose
-   depth
-   embeddings
-   VLM
-   tracking

The CLI enables only options supported by the selected capability.

## Plugin discovery

At startup:

1.  Scan `models/`.
2.  Import each plugin.
3.  Read matching YAML.
4.  Register by model id and capabilities.

## Comparable projects

  Project                     Borrow
  --------------------------- ---------------------------
  ComfyUI                     Dynamic plugin loading
  OpenMMLab                   Registry + config
  Ultralytics                 Clean inference API
  Hugging Face Transformers   Unified model abstraction
  Diffusers                   Pipeline composition

## Suggested roadmap

1.  Stable plugin API.
2.  YAML schema.
3.  Automatic discovery.
4.  Capability registry.
5.  Backend abstraction.
6.  Model cache.
7.  Auto-generated CLI/help.
8.  Optional GUI.

## Guiding principle

Adding a new model should ideally require only: - one Python wrapper -
one YAML file - model weights
