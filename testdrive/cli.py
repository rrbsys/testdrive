"""Command-line interface for testdrive.

Loop mode:
    Pass '*' as PLUGIN (for -M, -T, or the <plugin> positional) to repeat
    the action across every discovered plugin.

    <image> may also be a directory: every file in it is processed
    (except files matching our own output pattern, *-matches.* /
    *-redacted.*), so re-running over an already-processed directory
    doesn't try to detect objects in your own annotated output.

    '*' plugin and a directory <image> compose: every plugin runs
    against every image in the directory. In any loop (multi-plugin
    and/or multi-image), output files get '-<plugin id>' appended to
    their basename when more than one plugin is involved, so results
    don't collide. Quote '*' on shells that glob.

    --output-dir DIR sends all -matches/-redacted output there instead
    of next to each input image.

Cache discipline:
    A plain detect run (``testdrive <plugin> <image> <prompt>``) never
    downloads a model — if it isn't cached yet, the run fails fast with
    a clear message pointing at ``-T``/``-TT`` instead of silently
    blocking on a multi-gigabyte download in the middle of what looked
    like a normal command. Run ``-T <plugin>`` (or ``-TT <plugin>``)
    once first to populate the cache; after that, plain detect runs are
    unaffected either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import __version__
from .annotate import draw_boxes, redact
from .detection import DetectionResult, PluginManifest
from .imageio import derive_output_paths, load_image, save_image
from .pluginloader import PluginLoadError, iter_loadable_plugins, load_plugin
from .util import (
    ExitCode,
    setup_logging,
    set_max_parallel_files,
    set_downloads_allowed,
    set_auto_provision_enabled,
    set_pyenv_pip_upgrade,
    CacheNotPopulatedError,
)

log = logging.getLogger("testdrive.cli")

_DEFAULT_THRESHOLD = 0.3
# -TT's effective threshold, per plugin, is resolved by
# _resolve_test_threshold() below: an explicit --threshold always wins;
# otherwise each plugin's manifest 'test_threshold' field is used if it's
# a numeric string (some plugins are known to need a very different
# value on synthetic example images specifically — e.g. OWL-ViT's true
# match landed at confidence 0.07 there, while a blanket low threshold
# for every plugin just pulls in noise for others); "default"/anything
# non-numeric falls back to _DEFAULT_THRESHOLD, same as a plain detect
# run would use.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testdrive",
        description="One CLI. Many vision foundation models.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase verbosity (-v info, -vv debug)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"testdrive {__version__}",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-L", "--list", action="store_true", help="list discovered plugins")
    mode.add_argument(
        "-M", "--manifest", metavar="PLUGIN", help="show full manifest for PLUGIN ('*' = all)"
    )
    mode.add_argument(
        "-I", "--installed", action="store_true", help="installation status for all plugins"
    )
    mode.add_argument(
        "-T", "--selftest", metavar="PLUGIN", help="run self-test for PLUGIN ('*' = all)"
    )
    mode.add_argument(
        "-TT",
        "--example-test",
        metavar="PLUGIN",
        help="run PLUGIN's examples/<plugin>/image1-prompt-...-<N>matches.* through detection "
        "and check the match count ('*' = all). Also spelled '-T -T PLUGIN'.",
    )

    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=f"minimum confidence score (default: {_DEFAULT_THRESHOLD}; for -TT, overrides "
        "each plugin's manifest test_threshold instead)",
    )
    parser.add_argument(
        "--max-parallel-files",
        type=int,
        default=None,
        metavar="N",
        help="cap parallel file downloads per model snapshot (e.g. 1 on a slow link); "
        "default is the huggingface_hub library default (currently 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="write all -matches/-redacted output images here instead of next to "
        "each input image (created if it doesn't exist)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="override a plugin's default model variant, for plugins that declare more "
        "than one (see manifest 'models', e.g. yolo11n/s/m/l/x). No effect on plugins "
        "with a single fixed model.",
    )
    parser.add_argument(
        "--no-auto-provision",
        action="store_true",
        help="for plugins with their own environment (manifest 'pyenv' != \"framework\"): "
        "don't automatically create/pip-install it on first -T/-TT use — set it up by "
        'hand instead. No effect on plugins using the default "framework" environment.',
    )
    parser.add_argument(
        "--pyenv-pip-upgrade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when auto-provisioning a plugin's environment, upgrade pip in it first "
        "(default: on — use --no-pyenv-pip-upgrade to skip). No effect on the framework "
        "environment itself, which is always set up by hand.",
    )

    parser.add_argument("plugin", nargs="?", help="plugin id, e.g. owlv2 ('*' = all plugins)")
    parser.add_argument("image", nargs="?", help="path to an input image, or a directory of images")
    parser.add_argument("prompt", nargs="?", help='detection prompt (e.g. "person")')

    return parser


# ---------------------------------------------------------------------------
# -L
# ---------------------------------------------------------------------------


def cmd_list(as_json: bool) -> int:
    ids = sorted(p.manifest.id for p in iter_loadable_plugins())
    if as_json:
        print(json.dumps(ids, indent=2))
    else:
        if not ids:
            print("No plugins found in testdrive/models/.")
        for plugin_id in ids:
            print(plugin_id)
    return ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# -M
# ---------------------------------------------------------------------------


def _print_manifest_text(m: PluginManifest) -> None:
    print(f"Plugin ID      : {m.id}")
    if m.name:
        print(f"Name           : {m.name}")
    if m.version:
        print(f"Version        : {m.version}")
    print(f"API            : {m.api}")
    if m.description:
        print(f"Description    : {m.description}")
    print()
    if m.author:
        print(f"Author         : {m.author}")
    if m.homepage:
        print(f"Homepage       : {m.homepage}")
    if m.author or m.homepage:
        print()
    if m.license:
        print(f"License        : {m.license}")
    if m.license_url:
        print(f"License URL    : {m.license_url}")
    if m.license or m.license_url:
        print()
    if m.backend:
        print(f"Backend        : {m.backend}")
    if m.hf_repo:
        print(f"HF Repository  : {m.hf_repo}")
    if m.task:
        print(f"Task           : {m.task}")
    print(f"Pyenv          : {m.pyenv}")
    print()
    if m.supports:
        print("Supports")
        for item in m.supports:
            print(f"    \u2713 {item}")
        print()
    if m.requirements:
        print("Requirements")
        for req in m.requirements:
            print(f"    - {req['pip']}  (import {req['module']})")
        print()
    if m.models:
        print(f"Models         : {', '.join(m.models)}")
        print(f"Default model  : {m.model}  (override with --model)")
        print()
    if m.classes:
        print(f"Classes ({len(m.classes)})")
        # Wrap into readable rows rather than one giant line/one-per-line.
        row, width = [], 0
        for c in m.classes:
            row.append(c)
            width += len(c) + 2
            if width > 70:
                print("    " + ", ".join(row))
                row, width = [], 0
        if row:
            print("    " + ", ".join(row))
        print()
    if m.sample_prompt:
        print(f'Sample prompt  : "{m.sample_prompt}"')


def cmd_manifest(plugin_id: str, as_json: bool) -> int:
    try:
        loaded = load_plugin(plugin_id)
    except PluginLoadError as exc:
        log.error("plugin '%s' could not be loaded: %s", plugin_id, exc)
        return ExitCode.PLUGIN_NOT_FOUND

    m = loaded.manifest
    if as_json:
        print(json.dumps(m.to_dict(), indent=2))
        return ExitCode.SUCCESS

    _print_manifest_text(m)
    return ExitCode.SUCCESS


def cmd_manifest_loop(as_json: bool) -> int:
    """``-M '*'``: print manifests for every discovered plugin."""
    ids = sorted(p.manifest.id for p in iter_loadable_plugins())
    if not ids:
        print("No plugins found in testdrive/models/.")
        return ExitCode.SUCCESS

    any_failed = False
    json_out = []
    for i, plugin_id in enumerate(ids):
        try:
            loaded = load_plugin(plugin_id)
        except PluginLoadError as exc:
            any_failed = True
            if as_json:
                json_out.append({"plugin": plugin_id, "error": str(exc)})
            else:
                if i:
                    print()
                print(f"=== {plugin_id} ===")
                log.error("plugin '%s' could not be loaded: %s", plugin_id, exc)
            continue

        m = loaded.manifest
        if as_json:
            json_out.append(m.to_dict())
        else:
            if i:
                print()
            print(f"=== {plugin_id} ===")
            _print_manifest_text(m)

    if as_json:
        print(json.dumps(json_out, indent=2))

    return ExitCode.LOOP_PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# -I
# ---------------------------------------------------------------------------


def cmd_installed(as_json: bool) -> int:
    plugins = iter_loadable_plugins()
    results: list[dict[str, Any]] = []
    for loaded in sorted(plugins, key=lambda p: p.manifest.id):
        try:
            instance = loaded.instantiate()
            installed, missing = instance.is_installed()
        except Exception as exc:  # noqa: BLE001
            installed, missing = False, [f"error: {exc}"]
        results.append(
            {
                "id": loaded.manifest.id,
                "backend": loaded.manifest.backend,
                "installed": installed,
                "missing": missing,
            }
        )

    if as_json:
        print(json.dumps(results, indent=2))
        return ExitCode.SUCCESS

    if not results:
        print("No plugins found in testdrive/models/.")
        return ExitCode.SUCCESS

    for r in results:
        print(r["id"])
        print(f"    installed : {'yes' if r['installed'] else 'no'}")
        if r["backend"]:
            print(f"    backend   : {r['backend']}")
        if r["missing"]:
            print("    missing   :")
            for m in r["missing"]:
                print(f"        {m}")
        print()
    return ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# -T
# ---------------------------------------------------------------------------


def cmd_selftest(plugin_id: str, as_json: bool, model_override: str | None = None) -> int:
    from .selftest import run_selftest

    result = run_selftest(plugin_id, model_override=model_override)

    if as_json:
        print(
            json.dumps(
                {
                    "plugin": result.plugin_id,
                    "passed": result.passed,
                    "steps": result.steps,
                    "failures": result.failures,
                    "detection_count": result.detection_count,
                    "elapsed_ms": round(result.elapsed_ms, 1),
                },
                indent=2,
            )
        )
        return ExitCode.SUCCESS if result.passed else ExitCode.INFERENCE_FAILED

    print(result.summary())

    if result.passed:
        return ExitCode.SUCCESS
    elif result.failures and "missing" in result.failures[0]:
        return ExitCode.MISSING_DEPENDENCY
    elif result.failures and "environment set up" in result.failures[0]:
        return ExitCode.PYENV_NOT_CONFIGURED
    elif "load plugin" in (result.steps[0] if result.steps else ""):
        return ExitCode.PLUGIN_NOT_FOUND
    else:
        return ExitCode.INFERENCE_FAILED


def cmd_selftest_loop(as_json: bool, model_override: str | None = None) -> int:
    """``-T '*'``: run the self-test for every discovered plugin.

    Handy for a full cache rebuild (every plugin gets initialized, so
    every model gets downloaded) and for comparing detection behavior
    across plugins on the same synthetic image.
    """
    from .selftest import run_selftest

    ids = sorted(p.manifest.id for p in iter_loadable_plugins())
    if not ids:
        print("No plugins found in testdrive/models/.")
        return ExitCode.SUCCESS

    any_failed = False
    n_passed = 0
    json_out = []
    for i, plugin_id in enumerate(ids):
        result = run_selftest(plugin_id, model_override=model_override)
        if result.passed:
            n_passed += 1
        else:
            any_failed = True

        if as_json:
            json_out.append(
                {
                    "plugin": result.plugin_id,
                    "passed": result.passed,
                    "steps": result.steps,
                    "failures": result.failures,
                    "detection_count": result.detection_count,
                    "elapsed_ms": round(result.elapsed_ms, 1),
                }
            )
        else:
            if i:
                print()
            print(result.summary())

    if as_json:
        print(json.dumps(json_out, indent=2))
    else:
        print()
        print(f"{n_passed}/{len(ids)} plugin(s) passed")

    return ExitCode.LOOP_PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# detection  (<plugin> <image> <prompt>)
# ---------------------------------------------------------------------------


def _run_detect_one(
    plugin_id: str,
    image_path: Path,
    prompt: str,
    threshold: float,
    plugin_suffix: str | None = None,
    output_dir: Path | None = None,
    model_override: str | None = None,
) -> tuple[int, "DetectionResult | None", str | None]:
    """Run detection for a single (plugin, image) pair. Returns
    (exit_code, result, error).

    Does no printing itself, so it can be reused by both the
    single-run and loop-mode ('*' plugin and/or a directory of images)
    code paths, which need different presentation.
    """
    # 1. load plugin
    try:
        loaded = load_plugin(plugin_id)
    except PluginLoadError as exc:
        return ExitCode.PLUGIN_NOT_FOUND, None, f"plugin '{plugin_id}' could not be loaded: {exc}"

    plugin = loaded.instantiate()

    # 1.5. --model override, if given (a CLI-level user-input error, so
    # checked before dependencies — an invalid model name is wrong
    # regardless of what's installed). Safe to do here even for
    # worker-routed plugins below: this only touches static manifest
    # data (dataclasses.replace), no heavy imports.
    if model_override:
        try:
            plugin.set_model_override(model_override)
        except ValueError as exc:
            return ExitCode.CLI_ERROR, None, str(exc)

    # Plugins with a non-"framework" pyenv can't be dependency-checked,
    # initialized, or run in this process at all — their dependencies
    # live in a different environment entirely (that's the whole point
    # of declaring a different pyenv; see PluginManifest.pyenv). Those
    # three steps happen on the other side of a worker subprocess
    # instead — see pyenv.py / worker_main.py / worker_pool.py.
    uses_worker = plugin.manifest.pyenv != "framework"

    if not uses_worker:
        # 2. dependency check
        installed, missing = plugin.is_installed()
        if not installed:
            from .cache import cache_dir
            from .pyenv import install_hint

            msg = (
                f"plugin '{plugin_id}' is missing dependencies:\n    "
                + "\n    ".join(missing)
                + "\n\nInstall with:  "
                + install_hint(cache_dir(), plugin.manifest.pyenv, missing)
            )
            return ExitCode.MISSING_DEPENDENCY, None, msg

    # 3. load image (always in-process — core Pillow work, unaffected
    # by which pyenv a plugin declares)
    try:
        image = load_image(image_path)
    except FileNotFoundError as exc:
        return ExitCode.IMAGE_UNREADABLE, None, str(exc)
    except (OSError, ImportError) as exc:
        return ExitCode.IMAGE_UNREADABLE, None, f"could not load image '{image_path}': {exc}"

    width, height = image.size
    log.info("image loaded: %s (%dx%d)", image_path.name, width, height)

    if uses_worker:
        # 4 + 5 (worker path): a worker subprocess, kept alive across
        # this whole testdrive invocation, handles dependency checking,
        # initialize(), and detect() on the other side of the process
        # boundary — so a loop-mode run over many images only pays this
        # plugin's initialize() cost once, not once per image.
        from .cache import cache_dir
        from .worker_pool import WorkerError, get_pool

        try:
            worker = get_pool().get(plugin_id, plugin.manifest.pyenv, cache_dir())
            log.info(
                "running detection via '%s' worker: prompt=%r threshold=%.2f",
                plugin.manifest.pyenv,
                prompt,
                threshold,
            )
            t0 = time.perf_counter()
            detections = worker.detect(image_path, prompt, threshold, plugin.manifest.model or None)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except WorkerError as exc:
            if exc.kind == "missing_dependency":
                from .pyenv import install_hint

                msg = (
                    f"plugin '{plugin_id}' is missing dependencies in its "
                    f"'{plugin.manifest.pyenv}' environment:\n    "
                    + "\n    ".join(exc.missing)
                    + "\n\nInstall with:  "
                    + install_hint(cache_dir(), plugin.manifest.pyenv, exc.missing)
                )
                return ExitCode.MISSING_DEPENDENCY, None, msg
            if exc.kind == "cache_not_populated":
                return (
                    ExitCode.MISSING_DEPENDENCY,
                    None,
                    (
                        f"plugin '{plugin_id}' needs its model downloaded first ({exc}). "
                        f"Run `testdrive -T {plugin_id}` (or -TT {plugin_id}) once to populate the "
                        f"cache, then re-run this command."
                    ),
                )
            if exc.kind == "env_not_configured":
                return ExitCode.PYENV_NOT_CONFIGURED, None, str(exc)
            return ExitCode.INFERENCE_FAILED, None, f"plugin '{plugin_id}' (worker): {exc}"
    else:
        # 4. initialize plugin
        try:
            log.info("initializing plugin '%s' ...", plugin_id)
            plugin.initialize()
        except CacheNotPopulatedError as exc:
            return (
                ExitCode.MISSING_DEPENDENCY,
                None,
                (
                    f"plugin '{plugin_id}' needs its model downloaded first ({exc}). "
                    f"Run `testdrive -T {plugin_id}` (or -TT {plugin_id}) once to populate the "
                    f"cache, then re-run this command."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return (
                ExitCode.INFERENCE_FAILED,
                None,
                f"plugin '{plugin_id}' initialize() failed: {exc}",
            )

        # 5. run inference
        try:
            log.info("running detection: prompt=%r threshold=%.2f", prompt, threshold)
            t0 = time.perf_counter()
            detections = plugin.detect(image, prompt, threshold=threshold)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            return ExitCode.INFERENCE_FAILED, None, f"plugin '{plugin_id}' detect() raised: {exc}"

    # 6. annotate + save
    matches_path, redacted_path = derive_output_paths(
        image_path,
        plugin_suffix=plugin_suffix,
        output_dir=output_dir,
        match_count=len(detections),
    )
    try:
        save_image(draw_boxes(image, detections), matches_path)
        save_image(redact(image, detections), redacted_path)
    except (OSError, ImportError) as exc:
        return ExitCode.OUTPUT_WRITE_FAILED, None, f"could not save output images: {exc}"

    # 7. build result
    result = DetectionResult(
        image_path=image_path.resolve(),
        image_size=(width, height),
        plugin_id=plugin.manifest.id,
        plugin_name=plugin.manifest.name or plugin.manifest.id,
        plugin_version=plugin.manifest.version,
        prompt=prompt,
        threshold=threshold,
        model=plugin.manifest.model if plugin.manifest.models else "",
        detections=detections,
        elapsed_ms=elapsed_ms,
        matches_path=matches_path.resolve(),
        redacted_path=redacted_path.resolve(),
    )
    return ExitCode.SUCCESS, result, None


_OWN_OUTPUT_PATTERNS = ("*-matches*.*", "*-redacted.*")


def _is_own_output_file(path: Path) -> bool:
    """True if *path* matches our own output naming: ``*-redacted.*``,
    or ``*-matches.*``/``*-matches<N>.*`` (the match count embedded in
    the filename, e.g. ``photo-matches3.png`` — see
    imageio.derive_output_paths) — including plugin-suffixed variants
    like ``photo-owlv2-matches3.png``, which the glob's ``*`` covers too.
    """
    import fnmatch

    return any(fnmatch.fnmatch(path.name, pat) for pat in _OWN_OUTPUT_PATTERNS)


def _expand_image_arg(image_arg: str) -> tuple[list[Path], str | None]:
    """Resolve the ``<image>`` CLI argument to a list of image paths.

    A file resolves to itself. A directory resolves to every file
    directly inside it (non-recursive), except ones matching our own
    output pattern (``*-matches.*`` / ``*-redacted.*``) — so re-running
    detection over a directory you've already run detection in doesn't
    try to detect objects in your own annotated/redacted output images.

    Files with an unrecognized extension are still included rather than
    filtered out here: ``load_image()`` already produces a clean
    per-file "unreadable" error for anything Pillow can't decode, which
    is a better failure mode than silently skipping a file the person
    expected to be processed.
    """
    p = Path(image_arg)
    if not p.exists():
        return [], f"image path not found: {p}"

    if p.is_dir():
        candidates = sorted(x for x in p.iterdir() if x.is_file() and not _is_own_output_file(x))
        if not candidates:
            return [], (
                f"directory '{p}' has no input images "
                "(after excluding *-matches.*/*-redacted.* output files)"
            )
        return candidates, None

    return [p], None


def cmd_detect_dispatch(
    plugin_arg: str,
    image_arg: str,
    prompt: str,
    threshold: float,
    as_json: bool,
    output_dir: Path | None,
    model_override: str | None = None,
) -> int:
    """Handle ``<plugin> <image> <prompt>``, where ``<plugin>`` may be
    ``'*'`` (every discovered plugin) and ``<image>`` may be a directory
    (every file in it, minus our own prior output). Covers all four
    combinations with one code path; the plain single-plugin/single-image
    case prints exactly as before, everything else prints a per-run
    header plus a final tally, matching existing loop-mode conventions.
    """
    if plugin_arg == LOOP_ALL:
        plugin_ids = sorted(p.manifest.id for p in iter_loadable_plugins())
        if not plugin_ids:
            print("No plugins found in testdrive/models/.")
            return ExitCode.SUCCESS
    else:
        plugin_ids = [plugin_arg]

    image_paths, err = _expand_image_arg(image_arg)
    if err:
        log.error("%s", err)
        return ExitCode.IMAGE_UNREADABLE

    # The common case: exactly one plugin, one image — behave exactly as
    # a plain single detection run always has.
    if len(plugin_ids) == 1 and len(image_paths) == 1:
        exit_code, result, error = _run_detect_one(
            plugin_ids[0],
            image_paths[0],
            prompt,
            threshold,
            output_dir=output_dir,
            model_override=model_override,
        )
        if error:
            log.error("%s", error)
        if result:
            print(result.summary())
            if as_json:
                print()
                print(result.to_json())
        return exit_code

    # Loop mode: cross product of plugins x images.
    multi_plugin = len(plugin_ids) > 1
    any_failed = False
    n_passed = 0
    n_total = 0
    json_out = []

    for image_path in image_paths:
        for plugin_id in plugin_ids:
            n_total += 1
            header = image_path.name if not multi_plugin else f"{image_path.name} :: {plugin_id}"
            if not as_json:
                if n_total > 1:
                    print()
                print(f"=== {header} ===")

            plugin_suffix = plugin_id if multi_plugin else None
            exit_code, result, error = _run_detect_one(
                plugin_id,
                image_path,
                prompt,
                threshold,
                plugin_suffix=plugin_suffix,
                output_dir=output_dir,
                model_override=model_override,
            )
            if exit_code == ExitCode.SUCCESS:
                n_passed += 1
            else:
                any_failed = True

            if error:
                log.error("%s", error)

            if result:
                if as_json:
                    json_out.append({"exit_code": exit_code, **result.to_dict()})
                else:
                    print(result.summary())
            elif as_json:
                json_out.append(
                    {
                        "plugin_id": plugin_id,
                        "image_path": str(image_path),
                        "exit_code": exit_code,
                        "error": error,
                    }
                )

    if as_json:
        print(json.dumps(json_out, indent=2))
    else:
        print()
        print(f"{n_passed}/{n_total} run(s) succeeded")

    return ExitCode.LOOP_PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# -TT / example-test: run examples/<plugin>/image1-... through detection
# and check the match count encoded in its filename.
# ---------------------------------------------------------------------------

_EXAMPLE_FILENAME_RE = re.compile(
    r"^image1-prompt-(?P<slug>.+)-(?P<count>\d+)matches\.(?:png|jpe?g|bmp|webp)$",
    re.IGNORECASE,
)


def _examples_dir() -> Path:
    return Path(__file__).resolve().parent / "examples"


def _find_example_image(plugin_id: str) -> tuple[Path, str, int] | None:
    """Find ``examples/<plugin_id>/image1-prompt-<slug>-<N>matches.*`` and
    parse out the prompt and expected match count. Returns
    ``(path, prompt, expected_count)``, or ``None`` if there's no
    examples directory for this plugin, or no file in it matches the
    ``image1-prompt-...-<N>matches.<ext>`` naming convention.
    """
    plugin_dir = _examples_dir() / plugin_id
    if not plugin_dir.is_dir():
        return None

    for f in sorted(plugin_dir.iterdir()):
        if not f.is_file():
            continue
        m = _EXAMPLE_FILENAME_RE.match(f.name)
        if m:
            prompt = m.group("slug").replace("_", " ")
            return f, prompt, int(m.group("count"))

    return None


def _resolve_test_threshold(plugin_id: str, cli_threshold: float | None) -> float:
    """Resolve the effective threshold for a ``-TT`` run of *plugin_id*.

    Priority: an explicit ``--threshold`` on the CLI always wins.
    Otherwise, the plugin's manifest ``test_threshold`` is used if it's a
    numeric string (e.g. ``"0.05"``); ``"default"`` or any other
    non-numeric value falls back to ``_DEFAULT_THRESHOLD`` — the same
    threshold a plain detect run would use.
    """
    if cli_threshold is not None:
        return cli_threshold

    raw = None
    try:
        raw = load_plugin(plugin_id).manifest.test_threshold
    except PluginLoadError:
        pass  # unloadable plugin — _run_detect_one will report this properly

    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass  # "default" (or any other non-numeric string)

    return _DEFAULT_THRESHOLD


def _run_example_test_one(
    plugin_id: str,
    threshold: float | None,
    output_dir: Path,
    model_override: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run one plugin's example test. Returns ``(exit_code, info)`` where
    ``info`` has enough detail for both text and ``--json`` presentation.
    """
    found = _find_example_image(plugin_id)
    if found is None:
        return ExitCode.PLUGIN_NOT_FOUND, {
            "plugin": plugin_id,
            "passed": False,
            "error": (
                f"no usable example for '{plugin_id}': expected "
                f"examples/{plugin_id}/image1-prompt-<slug>-<N>matches.<ext>"
            ),
        }

    image_path, prompt, expected = found
    eff_threshold = _resolve_test_threshold(plugin_id, threshold)

    exit_code, result, error = _run_detect_one(
        plugin_id,
        image_path,
        prompt,
        eff_threshold,
        output_dir=output_dir,
        model_override=model_override,
    )

    if error:
        return exit_code, {
            "plugin": plugin_id,
            "image_path": str(image_path),
            "prompt": prompt,
            "expected": expected,
            "passed": False,
            "error": error,
        }

    # _run_detect_one's contract: exactly one of (result, error) is set.
    assert result is not None, "no error, but no result either — _run_detect_one contract violated"

    actual = len(result.detections)
    passed = actual == expected
    info = {
        "plugin": plugin_id,
        "image_path": str(image_path),
        "prompt": prompt,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "threshold": eff_threshold,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "matches_path": str(result.matches_path),
    }
    return (ExitCode.SUCCESS if passed else ExitCode.INFERENCE_FAILED), info


def cmd_example_test(
    plugin_arg: str,
    as_json: bool,
    threshold: float | None,
    output_dir: Path | None,
    model_override: str | None = None,
) -> int:
    """``-TT PLUGIN`` (or ``-T -T PLUGIN``, or ``'*'`` for every plugin):
    a real correctness check, not just "did it run" — finds
    ``examples/<plugin>/image1-prompt-...-<N>matches.*``, runs detection
    against it, and passes only if the number of detections found
    matches ``<N>``. Output defaults to the platform temp dir rather
    than the examples/ tree, unless --output-dir is given.
    """
    out_dir = output_dir if output_dir is not None else Path(tempfile.gettempdir())

    if plugin_arg == LOOP_ALL:
        plugin_ids = sorted(p.manifest.id for p in iter_loadable_plugins())
        if not plugin_ids:
            print("No plugins found in testdrive/models/.")
            return ExitCode.SUCCESS
    else:
        plugin_ids = [plugin_arg]

    multi = len(plugin_ids) > 1
    any_failed = False
    n_passed = 0
    json_out = []
    last_exit_code = ExitCode.SUCCESS

    for i, plugin_id in enumerate(plugin_ids):
        exit_code, info = _run_example_test_one(
            plugin_id, threshold, out_dir, model_override=model_override
        )
        last_exit_code = exit_code
        if info.get("passed"):
            n_passed += 1
        else:
            any_failed = True

        if as_json:
            info["exit_code"] = exit_code
            json_out.append(info)
            continue

        if multi:
            if i:
                print()
            print(f"=== {plugin_id} ===")

        if "error" in info:
            log.error("%s", info["error"])
            print("FAIL")
            continue

        status = "PASS" if info["passed"] else "FAIL"
        print(
            f"{status}  expected {info['expected']} match(es), got {info['actual']}  "
            f"(prompt={info['prompt']!r}  threshold={info['threshold']:.2f}  "
            f"{info['elapsed_ms']:.0f} ms)"
        )
        print(f"      {info['matches_path']}")

    if as_json:
        print(json.dumps(json_out, indent=2))
    elif multi:
        print()
        print(f"{n_passed}/{len(plugin_ids)} plugin(s) passed")

    if multi:
        return ExitCode.LOOP_PARTIAL_FAILURE if any_failed else ExitCode.SUCCESS
    return last_exit_code


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

LOOP_ALL = "*"


def _normalize_argv(argv: list[str]) -> list[str]:
    """Translate the '-T -T PLUGIN' spelling into '-TT PLUGIN' before
    argparse sees it, so both spellings of example-test mode work
    identically. (Plain argparse can't parse two consecutive '-T's as one
    doubled flag while '-T' also takes a value, without this.)
    """
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "-T" and i + 1 < len(argv) and argv[i + 1] == "-T":
            out.append("-TT")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _normalize_argv(argv)

    parser = build_parser()
    ns = parser.parse_args(argv)
    setup_logging(ns.verbose)

    try:
        return _dispatch(parser, ns)
    finally:
        # Guaranteed regardless of how dispatch exits (success, a
        # returned error code, or an uncaught exception) — safe/cheap
        # to call even if no worker was ever spawned this run (e.g.
        # every plugin used was "framework"-pyenv, running in-process).
        from .worker_pool import shutdown_all_workers

        shutdown_all_workers()


def _dispatch(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> int:
    if ns.max_parallel_files is not None:
        if ns.max_parallel_files < 1:
            log.error("--max-parallel-files must be >= 1 (got %d)", ns.max_parallel_files)
            return ExitCode.CLI_ERROR
        set_max_parallel_files(ns.max_parallel_files)

    set_auto_provision_enabled(not ns.no_auto_provision)
    set_pyenv_pip_upgrade(ns.pyenv_pip_upgrade)

    if ns.list:
        return cmd_list(ns.json)
    if ns.manifest:
        if ns.manifest == LOOP_ALL:
            return cmd_manifest_loop(ns.json)
        return cmd_manifest(ns.manifest, ns.json)
    if ns.installed:
        return cmd_installed(ns.json)
    if ns.example_test:
        set_downloads_allowed(True)  # -TT explicitly populates the cache
        output_dir = Path(ns.output_dir) if ns.output_dir else None
        return cmd_example_test(
            ns.example_test, ns.json, ns.threshold, output_dir, model_override=ns.model
        )
    if ns.selftest:
        set_downloads_allowed(True)  # -T explicitly populates the cache
        if ns.selftest == LOOP_ALL:
            return cmd_selftest_loop(ns.json, model_override=ns.model)
        return cmd_selftest(ns.selftest, ns.json, model_override=ns.model)

    if ns.plugin and ns.image and ns.prompt:
        # A plain detect run should never turn into a surprise multi-GB
        # download — run -T/-TT first to populate the cache (see
        # set_downloads_allowed's docstring for why).
        set_downloads_allowed(False)
        output_dir = Path(ns.output_dir) if ns.output_dir else None
        threshold = ns.threshold if ns.threshold is not None else _DEFAULT_THRESHOLD
        return cmd_detect_dispatch(
            plugin_arg=ns.plugin,
            image_arg=ns.image,
            prompt=ns.prompt,
            threshold=threshold,
            as_json=ns.json,
            output_dir=output_dir,
            model_override=ns.model,
        )

    positional = [x for x in (ns.plugin, ns.image, ns.prompt) if x]
    if positional:
        missing = 3 - len(positional)
        log.error(
            "detection requires <plugin> <image> <prompt>  (%d more argument%s needed)",
            missing,
            "s" if missing != 1 else "",
        )
        return ExitCode.CLI_ERROR

    parser.print_help()
    return ExitCode.CLI_ERROR


def entrypoint() -> int:
    """Real CLI entry point (installed ``testdrive`` command, and
    ``python -m testdrive``) — enforces the framework-environment guard
    (see ``pyenv.ensure_framework_env``) before doing anything else,
    then behaves exactly like ``main()``.

    Deliberately separate from ``main()``: the test suite calls
    ``main()`` directly so it's never subject to the environment guard
    (tests shouldn't need a real ``cache/pyenv/framework`` to run), and
    that's also why ``pyroject.toml``'s console script and
    ``__main__.py`` both point at this function instead of ``main``.
    """
    from .cache import cache_dir
    from .pyenv import ensure_framework_env

    ensure_framework_env(cache_dir())
    return main()


if __name__ == "__main__":
    sys.exit(entrypoint())
