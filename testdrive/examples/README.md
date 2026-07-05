# Examples

One subdirectory per plugin, each holding one or more small sample images:

```
examples/
  <plugin id>/
    image1-prompt-<sample prompt, spaces as underscores>-<N>matches.png
    image2-prompt-<another sample prompt>-<M>matches.png
    ...
```

``<N>``/``<M>`` is the number of detections that plugin is expected to
find in that image at a low threshold — this is what makes `testdrive
-TT <plugin>` (see below) an actual pass/fail check rather than just a
"did it crash" smoke test.

Try one directly:

```
testdrive florence2 examples/florence2/image1-prompt-blue_triangle-1matches.png "blue triangle"
```

Or run the automated per-plugin example test:

```
testdrive -TT florence2      # single plugin
testdrive -TT '*'            # every plugin, one pass/fail line each
```

`-TT <plugin>` (equivalently `-T -T <plugin>`) finds
`examples/<plugin>/image1-prompt-...-<N>matches.*`, derives the prompt
from the filename, runs detection against it (output written to the
platform temp dir by default, or `--output-dir` if given), and passes
only if the number of detections found equals `<N>`. It's a real
correctness check, not just "did the model run" — see `-T` for that.

## About these images

The images shipped here are **synthetic** — small canvases with a few
flat-color geometric shapes, generated with PIL, not photographs. That's
a deliberate choice, for two reasons:

1. **No real-photo licensing question.** Shipping arbitrary photos
   sourced from the web in a repo raises copyright questions this
   project doesn't need to take on.
2. **They only need to prove the plugin runs end-to-end** — that the
   image loads, the prompt reaches the model, and the right *number* of
   boxes come back. They are *not* meant to say anything about a
   model's real-world accuracy. Vision-language detectors are trained
   on natural photographs; a flat-shaded shape on a plain background is
   an easy, out-of-distribution case that doesn't reflect how these
   models perform on real scenes — in practice this also means they can
   need a much lower confidence threshold than you'd use on a real
   photo (we saw OWL-ViT's true match land at confidence 0.07 on its
   example image, well under the CLI's normal 0.3 default — `-TT` uses
   a low 0.05 default threshold for exactly this reason, overridable
   with `--threshold`).

**For actual accuracy testing or comparing plugins (`'*'` loop mode),
swap in your own real-world photos.** Keep the same naming convention
(including the `-<N>matches` suffix if you want `-TT` to validate them)
so it stays clear what prompt each image is meant for and how many
hits are expected, and feel free to add more than one image per plugin
(`image1-...`, `image2-...`, etc.) — e.g. one easy case and one hard
case per model. Note `-TT` only looks at `image1-...`; additional
images are for manual use with a plain `testdrive <plugin> <image>
<prompt>` call.

## Per-plugin sample prompt

Each plugin's manifest also carries this same prompt as `sample_prompt`
(see `testdrive -M <plugin>`), and `-T <plugin>` uses it automatically
against its own synthetic self-test image — these example images give
you a corresponding *visual* file to point the real CLI (or `-TT`) at,
whereas `-T` generates its own image in memory and never writes it to
disk.

| plugin        | sample prompt      | why                                                                 |
|---------------|---------------------|----------------------------------------------------------------------|
| florence2     | `blue triangle`     | phrase-grounding via `<CAPTION_TO_PHRASE_GROUNDING>` handles descriptive noun phrases well |
| groundingdino | `red star`          | open-vocabulary detection — not limited to a fixed class list        |
| molmo         | `largest circle`    | strong at comparative / spatial reasoning over a scene               |
| owlv2         | `purple pentagon`   | zero-shot detection, good recall on multiple/uncommon shapes         |
| owlvit        | `orange square`     | baseline zero-shot open-vocabulary detection                         |
| samgd         | `green triangle`    | Grounding DINO box + SAM mask → a tight, boundary-accurate box       |
| seem          | `yellow circle`     | kept for structural completeness; **plugin is currently unimplemented** — see its manifest/`-T seem` output |

Note `seem`'s example exists only so the directory layout is uniform
across all 7 plugins; running `testdrive seem ...` or `-TT seem`
against it will fail fast with an explanatory message, by design (see
the plugin's own docstring for why).

