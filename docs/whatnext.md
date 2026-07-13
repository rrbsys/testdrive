# whatnext: Projects related to testdrive

## Summary

Goal: a plugin-driven vision framework where adding `model.py` +
`model.yaml` makes a model immediately available.

## Comparable projects

  -----------------------------------------------------------------------
  Project                 Plugin mechanism        Closest idea
  ----------------------- ----------------------- -----------------------
  ComfyUI                 Python custom nodes     Dynamic plugin
                                                  discovery

  OpenMMLab               Registry + config       Configuration-driven
                                                  model registry

  Ultralytics             Python wrappers         Unified inference API

  Hugging Face            Model/config classes    Common abstraction over
  Transformers                                    many models

  Diffusers               Pipelines               Modular model loading

  MMDeploy                Backend plugins         Deployment abstraction
  -----------------------------------------------------------------------

## Suggested layout

``` text
models/
    yolo11.py
    florence2.py
    groundingdino.py
    sam2.py

configs/
    yolo11.yaml
    florence2.yaml
```

Each plugin implements:

``` python
class Model:
    def load(...)
    def predict(...)
    def classes(...)
```

Each YAML declares backend, weights, inputs, outputs and capabilities.

## Observation

The combination of automatic plugin discovery plus backend-agnostic
wrappers is still uncommon. The project is best thought of as a "VLC for
vision models".
