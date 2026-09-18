#!/usr/bin/env python3
"""
myhttpd.py — tiny local control panel for the "testdrive" tool.

Run:
    pip install markdown pymdown-extensions
    python3 myhttpd.py [port]        # default 8000

Then open http://localhost:8000/

Layout of the page (top to bottom):
    1. Control panel: td_url, td_home, td_python, td_plugins entryboxes,
       three command entryboxes (td_install / td_test / td_config) each
       with a run button, a wide console output box, a return-code
       field, and "clear console" / "copy to clipboard" buttons.
    2. <hr>
    3. Plugin list: the FULL list comes from running '<td> -L'; only the
       ones also present in plugins.lst (the configured subset) are
       hyperlinked. Clicking one runs '<td> -M <plugin>' and streams the
       result into the console box above -- the rest render as plain
       text. Then <hr> + links.md rendered the way GitHub renders a
       README.

td_install / td_test / td_config are run **as typed** (after resolving
placeholders) -- they do NOT go through the testdrive executable. Their
defaults invoke autoconfig.py directly:
    "<td_python>" autoconfig.py --testdrive-home <td_home> --plugins <td_plugins>
        --python-path "<td_python>" --testdrive-remoteurl <td_url> 2>&1
where '<td_home>' is replaced with the current td_home value,
'<td_python>' with the current td_python value (default 'python'),
'<td_url>' with the current td_url value (default: same as autoconfig.py's
own --testdrive-remoteurl default), and '<td_plugins>' with the current
td_plugins value (default: the comma-separated contents of plugins.lst,
or '' if that file doesn't exist -- autoconfig.py's --plugins accepts a
comma separated list directly, same as it accepts a filename).
td_install additionally passes --remove-home; td_config additionally
passes --remove-cache; td_test
additionally passes --check-installed right after 'autoconfig.py'.

<td> — the real testdrive executable — is still used, but only for the
per-plugin links below the console (‘<td> -M <plugin>’):
    Windows : testdrive\\testdrive.bat
    Unix    : testdrive/testdrive
(both resolved relative to this script's directory)

Data files expected alongside this script:
    plugins.lst    -> the configured subset of plugin ids, one per line
                      (the full list comes from '<td> -L' at runtime)
    links.md       -> markdown fixture, rendered as a README
    testdrive/...  -> the testdrive executable (used for plugin links)
    autoconfig.py  -> run by the td_install/test/config buttons

SECURITY NOTE: the run buttons build and execute a shell command from
editable text fields. That's the whole point of the tool (it's a local
dev/test console), but it means this server should only ever be bound
to localhost / run on a trusted machine — never expose it on a public
network as-is.
"""

import json
import html
import mimetypes
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

import markdown

BASE_DIR = Path(__file__).resolve().parent
PLUGINS_LST = BASE_DIR / "plugins.lst"
LINKS_MD = BASE_DIR / "links.md"

# <td> — the testdrive executable, OS dependent.
TD_EXE = (
    BASE_DIR / "testdrive" / "testdrive.bat"
    if os.name == "nt"
    else BASE_DIR / "testdrive" / "testdrive"
)


def td_argv(*args):
    """Build the argv LIST to invoke <td> THROUGH testdrive.bat/testdrive
    with the given args -- never a hand-built shell command string, since
    Windows' argument quoting (list2cmdline) is what Python itself uses
    to build the underlying command line correctly; hand-assembling one
    ourselves is an easy way to get subtly wrong quoting. On Windows,
    TD_EXE is a .bat, which CreateProcess can't launch directly
    (shell=False would fail with "not a valid Win32 application"), so it
    still needs cmd.exe /c.

    This is the *fallback* path -- see resolve_td_target(), which tries
    to bypass cmd.exe/testdrive.bat entirely on Windows first and only
    falls back to this when that isn't possible."""
    if os.name == "nt":
        return ["cmd.exe", "/c", str(TD_EXE), *args]
    return [str(TD_EXE), *args]


# testdrive.bat is a small, fixed template we generate ourselves (see
# autoconfig.py's write_launcher()) -- reliably parseable back out:
#   set "TESTDRIVE_CACHE=<cache>"
#   pushd "<exe-dir>"
#   "<exe-path>" %*
#   ...
_TD_BAT_CACHE_RE = re.compile(r'set\s+"TESTDRIVE_CACHE=([^"]*)"', re.IGNORECASE)
_TD_BAT_EXE_RE = re.compile(r'^"([^"]+)"\s+%\*\s*$', re.IGNORECASE | re.MULTILINE)


def resolve_td_target(*args):
    """Build (argv, env, cwd) to invoke <td> with `args`. Returns a real,
    directly-launchable target that BYPASSES cmd.exe/testdrive.bat
    entirely when possible; only falls back to going through the .bat
    (td_argv(), the previous behaviour) if that isn't possible.

    Why bypass at all: this toolchain is commonly run under Wine on
    macOS (confirmed) -- Wine's bundled cmd.exe is its own from-scratch
    reimplementation, not Microsoft's binary, with known rough edges,
    particularly around piped/non-console I/O. That's exactly what our
    streaming console and manifest-popup features require
    (subprocess.PIPE for stdout/stderr), and it's the likely reason
    testdrive.bat runs perfectly typed at an interactive prompt but
    fails ("not recognized as an internal or external command") when
    the identical .bat is invoked the same way via Python's subprocess.
    Bypassing cmd.exe for our own automated calls removes Wine's cmd.exe
    from the equation entirely; testdrive.bat itself is left untouched
    and still works fine for manual/interactive use.

    Does this by parsing TD_EXE's own known-fixed template to extract
    the real framework testdrive.exe path and the TESTDRIVE_CACHE value
    it sets, then launching that exe directly with shell=False."""
    if os.name == "nt" and TD_EXE.exists():
        try:
            content = TD_EXE.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""
        cache_match = _TD_BAT_CACHE_RE.search(content)
        exe_match = _TD_BAT_EXE_RE.search(content)
        if cache_match and exe_match:
            real_exe = Path(exe_match.group(1))
            env = os.environ.copy()
            env["TESTDRIVE_CACHE"] = cache_match.group(1)
            return [str(real_exe), *args], env, str(real_exe.parent)

    # Fallback: couldn't parse (not Windows, TD_EXE missing, or its
    # content doesn't match the expected template) -- go through the
    # wrapper as before.
    return td_argv(*args), None, str(TD_EXE.parent)


MD_EXTENSIONS = [
    "fenced_code",
    "tables",
    "sane_lists",
    "nl2br",
    "pymdownx.tilde",
    "pymdownx.tasklist",
]
MD_EXT_CONFIG = {"pymdownx.tasklist": {"custom_checkbox": True}}

# Sentinel line used to smuggle the process's return code through the
# streamed stdout without it being mistaken for real program output.
RC_MARKER = "@@MYHTTPD_RC@@:"

# Sentinel used only for plugin runs ('<td> -M <plugin>'): stdout (the
# manifest) is base64-encoded onto one line so it can ride through the
# same chunked text stream as the live stderr lines, then get pulled out
# client-side and shown in a popup instead of the console.
STDOUT_MARKER = "@@MYHTTPD_STDOUT_B64@@:"

DEFAULT_TD_HOME = ".\\testdrive" if os.name == "nt" else "./testdrive"
DEFAULT_TD_PYTHON = "python3.11" if sys.platform == "darwin" else "python"
# Same default as autoconfig.py's own --testdrive-remoteurl.
DEFAULT_TD_URL = "https://github.com/rrbsys/testdrive/archive/refs/heads/main.zip"
DEFAULT_TD_INSTALL = (
    '"<td_python>" autoconfig.py --remove-home --skip-cache --testdrive-home <td_home> '
    '--plugins <td_plugins> --python-path "<td_python>" '
    "--testdrive-remoteurl <td_url> 2>&1"
)
DEFAULT_TD_TEST = (
    '"<td_python>" autoconfig.py --check-installed --testdrive-home <td_home> '
    '--plugins <td_plugins> --python-path "<td_python>" '
    "--testdrive-remoteurl <td_url> 2>&1"
)
DEFAULT_TD_CONFIG = (
    '"<td_python>" autoconfig.py --remove-cache --skip-models --skip-private-pyenvs '
    "--testdrive-home <td_home> "
    '--plugins <td_plugins> --python-path "<td_python>" '
    "--testdrive-remoteurl <td_url> 2>&1"
)


def load_configured_plugin_ids():
    """plugins.lst -- the subset of plugins actually configured/installed
    via td_install/td_test/td_config, one id per line."""
    if not PLUGINS_LST.exists():
        return []
    return [
        line.strip()
        for line in PLUGINS_LST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def default_td_plugins_value():
    """Default text for the td_plugins entrybox: the configured plugin
    ids from plugins.lst, comma separated (autoconfig.py's --plugins
    accepts a comma separated list directly, same as it accepts a
    filename) -- or '' if plugins.lst doesn't exist / is empty."""
    return ",".join(load_configured_plugin_ids())


def get_full_plugin_list():
    """Run '<td> -L' to get the *full* list of plugins testdrive knows
    about (a superset of the configured subset in plugins.lst)."""
    argv, env, cwd = resolve_td_target("-L")
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"error running '<td> -L': {exc}"
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        return [], f"'<td> -L' exited {proc.returncode}: {detail}"
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return names, None


def render_plugin_list_html():
    configured = set(load_configured_plugin_ids())
    full_list, err = get_full_plugin_list()

    if err:
        return f"  <li><em>{html.escape(err)}</em></li>"
    if not full_list:
        return "  <li><em>(&lt;td&gt; -L returned no plugins)</em></li>"

    items = []
    for plugin_id in full_list:
        safe_id = html.escape(plugin_id)
        if plugin_id in configured:
            # Configured -- clicking it runs '<td> -M <plugin>'.
            items.append(
                f'  <li><a href="#" class="plugin-link" data-plugin="{safe_id}">{safe_id}</a></li>'
            )
        else:
            # Known to testdrive but not in plugins.lst -- plain text.
            items.append(f"  <li>{safe_id}</li>")
    return "\n".join(items)


PLUGIN_POPUP_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #ffffff; }}
  pre {{ white-space: pre-wrap; margin: 0; padding: 14px;
         font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         font-size: 13px; }}
</style>
</head>
<body><pre>{body}</pre></body>
</html>
"""


def get_plugin_status_text(plugin_id):
    """Externally-fetchable equivalent of the in-page popup: classify
    `plugin_id` against '<td> -L' (does testdrive know it at all?) and
    plugins.lst (is it configured?), then either return that verdict or,
    if it's fully known+configured, run '<td> -M <plugin_id>' and return
    its manifest (stdout only -- this is a plain synchronous GET meant
    for external consumption, not the live console)."""
    full_list, err = get_full_plugin_list()
    if err:
        return f"error checking plugin '{plugin_id}': {err}"

    if plugin_id not in full_list:
        return f"Plugin {plugin_id} not installed"

    configured = set(load_configured_plugin_ids())
    if plugin_id not in configured:
        return f"Plugin {plugin_id} not configured"

    argv, env, cwd = resolve_td_target("-M", plugin_id)
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error running '<td> -M {plugin_id}': {exc}"

    manifest = proc.stdout.strip()
    if manifest:
        return manifest
    # Configured plugins should normally produce a manifest; fall back to
    # something informative rather than an empty popup if it didn't.
    detail = proc.stderr.strip()
    return f"Plugin {plugin_id}: no manifest output (exit {proc.returncode}){': ' + detail if detail else ''}"


def render_readme_html():
    if not LINKS_MD.exists():
        return "<p><em>links.md not found</em></p>"
    text = LINKS_MD.read_text(encoding="utf-8")
    return markdown.markdown(text, extensions=MD_EXTENSIONS, extension_configs=MD_EXT_CONFIG)


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>myhttpd — Testdrive control panel</title>
<style>
  body {{ max-width: 980px; margin: 40px auto; padding: 0 20px;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1f2328; line-height: 1.5; }}
  h1, h2, h3 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .3em; }}
  hr {{ border: none; border-top: 1px solid #d0d7de; margin: 32px 0; }}
  code {{ background: #f6f8fa; padding: .15em .35em; border-radius: 4px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  pre code {{ display: block; padding: 12px; overflow-x: auto; }}
  table {{ border-collapse: collapse; }}
  table, th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; }}

  ul.plugin-list {{ list-style: none; padding-left: 0; }}
  ul.plugin-list li {{ padding: 4px 0; }}
  ul.plugin-list a {{ text-decoration: none; color: #0969da; cursor: pointer; }}
  ul.plugin-list a:hover {{ text-decoration: underline; }}

  .panel {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 16px; }}
  .row {{ display: flex; align-items: center; gap: 8px; }}
  .row label {{ min-width: 90px; font-weight: 600; font-family: ui-monospace, monospace; }}
  .row input[type=text] {{ flex: 1; font-family: ui-monospace, monospace; padding: 6px 8px;
                            border: 1px solid #d0d7de; border-radius: 6px; }}
  button {{ padding: 6px 14px; border: 1px solid #d0d7de; border-radius: 6px;
            background: #f6f8fa; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #eaeef2; }}
  button:active {{ background: #d0d7de; }}
  button.run {{ background: #2da44e; color: white; border-color: #2da44e; }}
  button.run:hover {{ background: #2c974b; }}
  button.stop {{ background: #d1242f; color: white; border-color: #d1242f; }}
  button.stop:hover {{ background: #b91c28; }}

  #console {{ width: 100%; height: 260px; box-sizing: border-box; resize: vertical;
              font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              font-size: 13px; background: #0d1117; color: #c9d1d9;
              border: 1px solid #30363d; border-radius: 6px; padding: 10px; white-space: pre; }}
  .rc-row label {{ min-width: 90px; font-weight: 600; }}
  .rc-row input[type=text] {{ width: 80px; font-family: ui-monospace, monospace;
                               text-align: center; padding: 4px; }}
  .console-buttons {{ display: flex; gap: 8px; }}
</style>
</head>
<body>

<h1>Testdrive control panel</h1>

<div class="panel">
  <div class="row">
    <label for="td_url">td_url</label>
    <input type="text" id="td_url" value="{td_url}">
  </div>

  <div class="row">
    <label for="td_home">td_home</label>
    <input type="text" id="td_home" value="{td_home}">
  </div>

  <div class="row">
    <label for="td_python">td_python</label>
    <input type="text" id="td_python" value="{td_python}">
  </div>

  <div class="row">
    <label for="td_plugins">td_plugins</label>
    <input type="text" id="td_plugins" value="{td_plugins}">
  </div>

  <div class="row">
    <label for="td_install">td_install</label>
    <input type="text" id="td_install" value="{td_install}">
    <button class="run" onclick="runTemplate('td_install')">Run</button>
  </div>

  <div class="row">
    <label for="td_test">td_test</label>
    <input type="text" id="td_test" value="{td_test}">
    <button class="run" onclick="runTemplate('td_test')">Run</button>
  </div>

  <div class="row">
    <label for="td_config">td_config</label>
    <input type="text" id="td_config" value="{td_config}">
    <button class="run" onclick="runTemplate('td_config')">Run</button>
  </div>

  <div class="row">
    <button class="stop" onclick="stopCurrent()">Stop</button>
  </div>

  <textarea id="console" readonly></textarea>

  <div class="row rc-row">
    <label for="rc">return code</label>
    <input type="text" id="rc" readonly value="">
  </div>

  <div class="console-buttons">
    <button onclick="clearConsole()">Clear console</button>
    <button onclick="copyConsole()">Copy to Clipboard</button>
  </div>
</div>

<hr>

<h1>Plugins</h1>
<ul class="plugin-list">
{plugin_items}
</ul>
<hr>
{rendered_readme}

<script>
const consoleBox = document.getElementById('console');
const rcBox = document.getElementById('rc');
const RC_MARKER = {rc_marker_json};
const STDOUT_MARKER = {stdout_marker_json};

function b64DecodeUtf8(b64) {{
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new TextDecoder('utf-8').decode(bytes);
}}

function showStdoutPopup(text) {{
  const w = window.open('', '_blank', 'width=640,height=520,scrollbars=yes');
  if (!w) {{
    // Popup blocked -- fall back to putting it in the console instead
    // of silently losing it.
    appendConsole('\\n[popup blocked, showing manifest here instead]\\n' + text + '\\n');
    return;
  }}
  w.document.title = 'testdrive -M output';
  w.document.body.style.margin = '0';
  const pre = w.document.createElement('pre');
  pre.style.whiteSpace = 'pre-wrap';
  pre.style.fontFamily = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  pre.style.fontSize = '13px';
  pre.style.padding = '14px';
  pre.textContent = text;
  w.document.body.appendChild(pre);
}}

function appendConsole(text) {{
  consoleBox.value += text;
  consoleBox.scrollTop = consoleBox.scrollHeight;
}}

function clearConsole() {{
  consoleBox.value = '';
  rcBox.value = '';
}}

function copyConsole() {{
  navigator.clipboard.writeText(consoleBox.value).catch(() => {{
    consoleBox.select();
    document.execCommand('copy');
  }});
}}

function stopCurrent() {{
  fetch('/stop', {{method: 'POST'}})
    .then(r => r.text())
    .then(t => appendConsole('\\n[' + t.trim() + ']\\n'))
    .catch(() => appendConsole('\\n[stop request failed]\\n'));
}}

async function runRequest(payload, echoLine) {{
  rcBox.value = '';
  appendConsole('\\n$ ' + echoLine + '\\n');
  const resp = await fetch('/run', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload),
  }});
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {{
    const {{done, value}} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {{stream: true}});
    let idx;
    while ((idx = buf.indexOf('\\n')) !== -1) {{
      const line = buf.slice(0, idx + 1);
      buf = buf.slice(idx + 1);
      if (line.startsWith(RC_MARKER)) {{
        rcBox.value = line.slice(RC_MARKER.length).trim();
      }} else if (line.startsWith(STDOUT_MARKER)) {{
        showStdoutPopup(b64DecodeUtf8(line.slice(STDOUT_MARKER.length).trim()));
      }} else {{
        appendConsole(line);
      }}
    }}
  }}
  if (buf) {{
    if (buf.startsWith(RC_MARKER)) {{
      rcBox.value = buf.slice(RC_MARKER.length).trim();
    }} else if (buf.startsWith(STDOUT_MARKER)) {{
      showStdoutPopup(b64DecodeUtf8(buf.slice(STDOUT_MARKER.length).trim()));
    }} else {{
      appendConsole(buf);
    }}
  }}
}}

// td_install / td_test / td_config: run the (resolved) entrybox content
// exactly as typed -- no '<td>' executable is prepended here anymore.
function runTemplate(fieldId) {{
  const tdHome = document.getElementById('td_home').value;
  const tdPython = document.getElementById('td_python').value;
  const tdUrl = document.getElementById('td_url').value;
  const tdPlugins = document.getElementById('td_plugins').value;
  let cmd = document.getElementById(fieldId).value;
  cmd = cmd.split('<td_home>').join(tdHome);
  cmd = cmd.split('<td_python>').join(tdPython);
  cmd = cmd.split('<td_url>').join(tdUrl);
  cmd = cmd.split('<td_plugins>').join(tdPlugins);
  runRequest({{mode: 'raw', cmd: cmd}}, cmd);
}}

// Plugin links still go through the real testdrive executable ('<td>').
document.querySelectorAll('.plugin-link').forEach(function (el) {{
  el.addEventListener('click', function (ev) {{
    ev.preventDefault();
    const plugin = el.dataset.plugin;
    runRequest({{mode: 'td', plugin: plugin}}, '<td> -M ' + plugin);
  }});
}});

// Pressing Enter in a run-command entrybox runs it, same as its button.
['td_install', 'td_test', 'td_config'].forEach(function (fieldId) {{
  document.getElementById(fieldId).addEventListener('keydown', function (ev) {{
    if (ev.key === 'Enter') {{
      ev.preventDefault();
      runTemplate(fieldId);
    }}
  }});
}});
</script>

</body>
</html>
"""


def _killable_popen_kwargs():
    """Extra Popen kwargs so the WHOLE descendant tree can be killed
    later, not just the immediate child -- see _stop_current_proc().
    A plain proc.terminate() only ever reaches the immediate child; with
    shell=True that's often the shell (or autoconfig.py) itself, and any
    grandchild it spawns (a pip install, testdrive's own -T/-TT call)
    can keep running as an orphan even after our streaming correctly
    reports 'stopped', since Python doesn't propagate a terminate signal
    to a subprocess's own children automatically."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# Tracks whichever subprocess is currently streaming to the console (at
# most one at a time, matching the single shared console UI), so the
# Stop button can terminate it regardless of which run box started it.
_proc_lock = threading.Lock()
_current_proc = None


def _set_current_proc(proc):
    global _current_proc
    with _proc_lock:
        _current_proc = proc


def _clear_current_proc(proc):
    global _current_proc
    with _proc_lock:
        if _current_proc is proc:
            _current_proc = None


def _stop_current_proc():
    """Terminate whatever's currently running, if anything -- the WHOLE
    process tree, not just the one process we Popen'd (see
    _killable_popen_kwargs()). Returns True if something was actually
    running and got a terminate signal."""
    with _proc_lock:
        proc = _current_proc
    if proc is None or proc.poll() is not None:
        return False
    if os.name == "nt":
        # taskkill /T walks the process tree by parent-PID, independent
        # of the CREATE_NEW_PROCESS_GROUP flag; /F forces it.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        # start_new_session=True (see _killable_popen_kwargs()) made
        # proc's own pid its process group's leader, so killing that
        # whole group reaches every descendant, not just proc itself.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass
    return True


class Handler(BaseHTTPRequestHandler):
    server_version = "myhttpd/0.2"

    def _send(self, status, content_type, body_bytes, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_bytes)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body_bytes)

    def _send_chunk(self, data_bytes):
        self.wfile.write(("%x\r\n" % len(data_bytes)).encode("ascii"))
        self.wfile.write(data_bytes)
        self.wfile.write(b"\r\n")

    def _end_chunks(self):
        self.wfile.write(b"0\r\n\r\n")

    def _serve_static(self, rel_path):
        """Serve a file from BASE_DIR if it exists and is safely inside it
        (used for images etc. referenced from links.md)."""
        candidate = (BASE_DIR / rel_path).resolve()
        try:
            candidate.relative_to(BASE_DIR)
        except ValueError:
            self._send(403, "text/plain; charset=utf-8", b"Forbidden")
            return True
        if candidate.is_file():
            ctype = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            self._send(200, ctype, candidate.read_bytes())
            return True
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "/index.html":
            page = PAGE_TEMPLATE.format(
                td_url=html.escape(DEFAULT_TD_URL),
                td_home=html.escape(DEFAULT_TD_HOME),
                td_python=html.escape(DEFAULT_TD_PYTHON),
                td_plugins=html.escape(default_td_plugins_value()),
                td_install=html.escape(DEFAULT_TD_INSTALL),
                td_test=html.escape(DEFAULT_TD_TEST),
                td_config=html.escape(DEFAULT_TD_CONFIG),
                plugin_items=render_plugin_list_html(),
                rendered_readme=render_readme_html(),
                rc_marker_json=json.dumps(RC_MARKER),
                stdout_marker_json=json.dumps(STDOUT_MARKER),
            )
            self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
            return

        # External URL to read a plugin's manifest/status from outside
        # this app entirely (e.g. opened directly, or via window.open()
        # from another site/tool) -- not the same code path as the
        # in-page plugin-link click, which streams live via /run.
        if path.startswith("/plugin/"):
            plugin_id = path[len("/plugin/") :]
            if plugin_id:
                body_text = get_plugin_status_text(plugin_id)
                page = PLUGIN_POPUP_TEMPLATE.format(
                    title=html.escape(f"testdrive -M {plugin_id}"),
                    body=html.escape(body_text),
                )
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return

        # Static files referenced by links.md (e.g. /family4.jpg)
        rel = path.lstrip("/")
        if rel and self._serve_static(rel):
            return

        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _run_merged(self, cmd):
        """raw mode: stdout+stderr merged, streamed live as before."""
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **_killable_popen_kwargs(),
            )
        except OSError as exc:
            self._send_chunk(f"error launching: {exc}\n".encode("utf-8"))
            return -1

        _set_current_proc(proc)
        try:
            for line in proc.stdout:
                self._send_chunk(line.encode("utf-8"))
            rc = proc.wait()
        finally:
            _clear_current_proc(proc)
        return rc

    def _run_td_split(self, argv, env, cwd):
        """td mode ('<td> -M <plugin>'): stderr streams to the console
        live, same as before; stdout (the manifest) is captured whole and
        sent as a single base64-marked line so the client can pop it up
        in its own window instead of dumping it into the console.

        `argv`/`env`/`cwd` come from resolve_td_target(), which bypasses
        cmd.exe/testdrive.bat on Windows when possible (see its
        docstring for why: Wine's cmd.exe reimplementation and piped
        I/O). When bypassing isn't possible, cwd falls back to TD_EXE's
        own directory rather than BASE_DIR -- on Windows, pip's
        console_scripts .exe wrapper runs its entry point via runpy,
        which prepends the process's cwd to sys.path, so if cwd were
        BASE_DIR (which has a "testdrive/" subfolder, since that's where
        TD_EXE lives), Python could resolve `import testdrive` to that
        empty subfolder as a namespace package instead of the real
        installed one, giving exactly "ImportError: cannot import name
        '__version__' from 'testdrive' (unknown location)".
        """
        try:
            proc = subprocess.Popen(
                argv,
                shell=False,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                **_killable_popen_kwargs(),
            )
        except OSError as exc:
            self._send_chunk(f"error launching td: {exc}\n".encode("utf-8"))
            return -1

        _set_current_proc(proc)
        try:
            q = queue.Queue()

            def pump(stream, tag):
                for line in stream:
                    q.put((tag, line))
                stream.close()
                q.put((tag, None))

            t_out = threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True)
            t_err = threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True)
            t_out.start()
            t_err.start()

            stdout_parts = []
            streams_done = 0
            while streams_done < 2:
                tag, line = q.get()
                if line is None:
                    streams_done += 1
                    continue
                if tag == "err":
                    self._send_chunk(line.encode("utf-8"))
                else:
                    stdout_parts.append(line)

            t_out.join()
            t_err.join()
            rc = proc.wait()
        finally:
            _clear_current_proc(proc)

        stdout_text = "".join(stdout_parts)
        b64 = base64.b64encode(stdout_text.encode("utf-8")).decode("ascii")
        self._send_chunk(f"{STDOUT_MARKER}{b64}\n".encode("utf-8"))
        return rc

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/stop":
            stopped = _stop_current_proc()
            msg = b"stopped\n" if stopped else b"nothing running\n"
            self._send(200, "text/plain; charset=utf-8", msg)
            return

        if path != "/run":
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        mode = payload.get("mode", "td")
        if mode == "raw":
            # td_install / td_test / td_config: run the entrybox content
            # exactly as given, no '<td>' executable involved. This one
            # genuinely needs shell=True (it's a whole shell command line,
            # e.g. with "2>&1" redirection baked in), so it stays a string.
            cmd = str(payload.get("cmd", ""))
        else:
            # Plugin links: still routed through the real testdrive
            # executable, e.g. '<td> -M yunet' -- resolve_td_target()
            # bypasses cmd.exe/testdrive.bat on Windows when possible.
            plugin = str(payload.get("plugin", ""))
            argv, env, cwd = resolve_td_target("-M", plugin)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if mode == "raw":
            rc = self._run_merged(cmd)
        else:
            rc = self._run_td_split(argv, env, cwd)

        self._send_chunk(f"{RC_MARKER}{rc}\n".encode("utf-8"))
        self._end_chunks()

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"myhttpd serving on http://localhost:{port}/  (Ctrl+C to stop)")
    print(f"<td> resolves to: {TD_EXE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
