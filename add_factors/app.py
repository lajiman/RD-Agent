import os
import re
import pathlib
import yaml
import html
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import difflib
import subprocess
import sys
import time
from datetime import datetime

import gradio as gr

from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI


# ----------------------------
# Repo-relative targets
# ----------------------------
REL_ENRICH_DAILY = "rdagent/scenarios/qlib/experiment/factor_data_template/enrich_daily_py.py"
REL_EXPERIMENT_PROMPTS = "rdagent/scenarios/qlib/experiment/prompts.yaml"
REL_SCENARIO_PROMPTS = "rdagent/scenarios/qlib/prompts.yaml"

KEY_QLIB_FACTOR_BACKGROUND = "qlib_factor_background"
KEY_FACTOR_HYP_SPEC = "factor_hypothesis_specification"

# prompt pack + assets
PROMPT_PACK_REL = "add_factors/prompt_pack.yaml"


# ----------------------------
# Robust YAML block scalar locator
# ----------------------------
def _find_yaml_block(yaml_text: str, key: str) -> Dict[str, Any]:
    """
    Find a YAML block-scalar for a given key, supporting:
      - any indentation level
      - |, |- , |+
    Returns dict with:
      indent: leading whitespace of the key line
      header_span: (start, end) span of the header line (including newline)
      body_span: (start, end) span of the body (indented content)
      body_indent: indentation string used for body lines (preserved from file)
    Raises ValueError if not found.
    """
    # Match a key line like:
    # <indent>key: |-
    # <indent>key: |
    # <indent>key: |+   (rare but valid)
    # allow trailing spaces/comments after the scalar indicator
    key_re = re.compile(
        rf"^(?P<indent>[ \t]*){re.escape(key)}\s*:\s*\|(?P<chomp>[+-])?\s*(?:#.*)?\n",
        flags=re.MULTILINE,
    )
    m = key_re.search(yaml_text)
    if not m:
        raise ValueError(
            f"Cannot find YAML block for key '{key}:' with a block scalar (|, |-, |+)."
        )

    indent = m.group("indent")
    header_start, header_end = m.span(0)
    header_span = (header_start, header_end)

    # Body continues while lines are more indented than the key line
    body_start = header_end
    pos = body_start
    n = len(yaml_text)

    body_end = body_start
    while pos < n:
        next_nl = yaml_text.find("\n", pos)
        if next_nl == -1:
            line = yaml_text[pos:]
            line_end = n
        else:
            line = yaml_text[pos : next_nl + 1]
            line_end = next_nl + 1

        if line.strip() == "":
            # keep blank lines as part of the block (safe behavior for editing use-case)
            body_end = line_end
            pos = line_end
            continue

        # block lines must be indented more than the key indent
        if re.match(rf"^{re.escape(indent)}[ \t]+", line):
            body_end = line_end
            pos = line_end
            continue

        break

    body_span = (body_start, body_end)

    # preserve original body indentation from first non-empty body line
    body_indent = indent + "  "
    body_text = yaml_text[body_start:body_end]
    for ln in body_text.splitlines():
        if ln.strip():
            leading = re.match(r"^([ \t]+)", ln)
            if leading:
                body_indent = leading.group(1)
            break

    return {
        "indent": indent,
        "header_span": header_span,
        "body_span": body_span,
        "body_indent": body_indent,
    }


def extract_yaml_block(yaml_text: str, key: str) -> str:
    """
    Extract block scalar content for `key`, de-indenting by the detected body indent.
    Returns the block text WITHOUT the YAML key line.
    """
    info = _find_yaml_block(yaml_text, key)
    body_start, body_end = info["body_span"]
    body_indent = info["body_indent"]
    body = yaml_text[body_start:body_end]

    out_lines = []
    for line in body.splitlines(True):
        if line.strip() == "":
            out_lines.append(line)
        else:
            if line.startswith(body_indent):
                out_lines.append(line[len(body_indent) :])
            else:
                # fallback: strip one indent group
                out_lines.append(re.sub(r"^[ \t]+", "", line, count=1))
    return "".join(out_lines)


def replace_yaml_block(yaml_text: str, key: str, new_body_deindented: str) -> str:
    """
    Replace block scalar content for `key`, re-indenting with the original detected body indent.
    """
    info = _find_yaml_block(yaml_text, key)
    body_start, body_end = info["body_span"]
    body_indent = info["body_indent"]

    nb = new_body_deindented
    if nb and not nb.endswith("\n"):
        nb += "\n"

    indented = []
    for line in nb.splitlines(True):
        if line == "\n":
            indented.append(body_indent + "\n")
        else:
            indented.append(body_indent + line)

    return yaml_text[:body_start] + "".join(indented) + yaml_text[body_end:]


# ----------------------------
# File IO
# ----------------------------
def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@dataclass
class RepoFiles:
    repo_root: pathlib.Path

    @property
    def enrich_daily(self) -> pathlib.Path:
        return self.repo_root / REL_ENRICH_DAILY

    @property
    def experiment_prompts(self) -> pathlib.Path:
        return self.repo_root / REL_EXPERIMENT_PROMPTS

    @property
    def scenario_prompts(self) -> pathlib.Path:
        return self.repo_root / REL_SCENARIO_PROMPTS

    @property
    def prompt_pack(self) -> pathlib.Path:
        return self.repo_root / PROMPT_PACK_REL


# ----------------------------
# Prompt pack (your fixed examples + instructions)
# ----------------------------
def load_prompt_pack(repo_root: pathlib.Path) -> Dict[str, Any]:
    """
    Load add_factors/prompt_pack.yaml.
    Returns {} if missing/unreadable.
    """
    p = (repo_root / PROMPT_PACK_REL).resolve()
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_asset(repo_root: pathlib.Path, rel_path: str) -> str:
    p = (repo_root / rel_path).expanduser().resolve()
    return p.read_text(encoding="utf-8")


def build_examples_pack_text(repo_root: pathlib.Path, pack: Dict[str, Any], target_key: str) -> str:
    """
    target_key in { KEY_QLIB_FACTOR_BACKGROUND, KEY_FACTOR_HYP_SPEC }

    Reads:
      add_factors/prompt_assets/example_enrich_daily_py.py
      add_factors/prompt_assets/example_qlib_factor_background.txt
      add_factors/prompt_assets/example_factor_hypothesis_specification.txt
    Paths are configured in prompt_pack.yaml under assets: ...
    """
    assets = pack.get("assets") or {}

    missing = []
    for k in [
        "example_enrich_daily_py",
        "example_qlib_factor_background",
        "example_factor_hypothesis_specification",
    ]:
        if k not in assets:
            missing.append(k)
    if missing:
        raise RuntimeError(
            "prompt_pack.yaml missing assets keys: " + ", ".join(missing)
        )

    ex_enrich = _read_asset(repo_root, assets["example_enrich_daily_py"])
    ex_bg = _read_asset(repo_root, assets["example_qlib_factor_background"])
    ex_hyp = _read_asset(repo_root, assets["example_factor_hypothesis_specification"])

    if target_key == KEY_QLIB_FACTOR_BACKGROUND:
        ex_block = ex_bg
        other_block = ex_hyp
        target_name = KEY_QLIB_FACTOR_BACKGROUND
    else:
        ex_block = ex_hyp
        other_block = ex_bg
        target_name = KEY_FACTOR_HYP_SPEC

    # Now we just need ex_enrich and ex_block
    return (
        "EXAMPLE PACK (fixed reference; follow style/structure; do not copy verbatim):\n"
        "=== Example enrich_daily_py.py ===\n"
        f"{ex_enrich}\n\n"
        f"=== Example {target_name} block ===\n"
        f"{ex_block}\n\n"
        # "=== Example other block (consistency reference) ===\n"
        # f"{other_block}\n"
    )


def get_instruction_and_examples_defaults(pack: Dict[str, Any], key: str) -> Tuple[str, str]:
    """
    Fetch instruction/examples defaults from prompt_pack.yaml.
    """
    gen = pack.get("generation") or {}
    cfg = gen.get(key) or {}
    instruction = cfg.get("instruction") or ""
    # 'examples' here is optional; main examples come from prompt_assets
    examples = cfg.get("examples") or ""
    return instruction, examples


# ----------------------------
# Azure OpenAI helper
# ----------------------------
def _get_azure_client() -> AzureOpenAI:
    if AzureOpenAI is None:
        raise RuntimeError("openai SDK not available. Please `pip install openai` (>=1.x).")

    api_key = os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_API_BASE")
    api_version = os.getenv("AZURE_API_VERSION")

    if not api_key or not endpoint or not api_version:
        raise RuntimeError(
            "Missing Azure env vars. Need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION in your environment/.env"
        )

    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


def azure_suggest_revision(
    target_key: str,
    current_block_text: str,
    instruction: str,
    examples_pack_text: str,
    extra_examples_text: str,
    current_enrich_daily_py: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> str:
    """
    Produce suggested revised version of the YAML-block text.
    Returns ONLY the revised text (no YAML keys, no code fences).
    """
    model = "gpt-5.2-chat"
    client = _get_azure_client()

    system_msg = (
        "You are assisting with prompt/spec editing for a quantitative factor research agent.\n"
        "Return ONLY the revised block text content.\n"
        "Do NOT include YAML keys. Do NOT include code fences.\n"
        "Preserve no-lookahead constraints and avoid global/full-sample statistics.\n"
        "Keep wording concise, enforceable, and aligned with the example pack.\n"
    )

    user_msg = (
        f"TARGET BLOCK: {target_key}\n\n"
        "TASK INSTRUCTION:\n"
        f"{instruction.strip()}\n\n"
        "CURRENT enrich_daily_py.py (authoritative, updated by user):\n"
        "-----\n"
        f"{current_enrich_daily_py}\n"
        "-----\n\n"
        "CURRENT BLOCK TEXT TO REVISE:\n"
        "-----\n"
        f"{current_block_text.strip()}\n"
        "-----\n\n"
        f"{examples_pack_text}\n\n"
        "OPTIONAL EXTRA EXAMPLES / NOTES:\n"
        "-----\n"
        f"{(extra_examples_text or '').strip()}\n"
        "-----\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Output ONLY the revised block text.\n"
        "- Do NOT output YAML keys.\n"
        "- Do NOT output triple backticks.\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    out = resp.choices[0].message.content or ""
    # strip accidental fences if any
    out = re.sub(r"^\s*```.*?\n", "", out, flags=re.DOTALL)
    out = re.sub(r"\n```(\s*)$", r"\1", out)
    out = out.strip()
    if out and not out.endswith("\n"):
        out += "\n"
    return out


# ----------------------------
# Compare texts
# ----------------------------
def unified_diff_text(old: str, new: str, fromfile: str, tofile: str) -> str:
    """
    Git-like unified diff text for display.
    """
    old_lines = (old or "").splitlines(keepends=True)
    new_lines = (new or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def html_diff_table(old: str, new: str, context_lines: int = 3) -> str:
    import difflib

    old_lines = (old or "").splitlines()
    new_lines = (new or "").splitlines()

    hd = difflib.HtmlDiff(tabsize=4, wrapcolumn=140)
    table = hd.make_table(
        fromlines=old_lines,
        tolines=new_lines,
        fromdesc="current",
        todesc="ai_suggested",
        context=True,
        numlines=context_lines,
    )

    css = """
    <style>
      .diff-wrap {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
      }

      table.diff {
        width: 100%;
        border-collapse: collapse;
        /* auto is important: allows column to collapse cleanly when we hide cells */
        table-layout: auto;
      }

      table.diff th, table.diff td {
        border: 1px solid #e5e7eb;
        padding: 4px 6px;
        vertical-align: top;
      }

      table.diff th {
        background: #f9fafb;
        font-weight: 600;
      }

      table.diff td {
        white-space: pre-wrap;
        word-break: break-word;
      }

      /* difflib highlight classes */
      .diff_header { background: #f3f4f6; color: #6b7280; }
      .diff_add    { background: #dcfce7; }  /* green */
      .diff_chg    { background: #fef9c3; }  /* yellow */
      .diff_sub    { background: #fee2e2; }  /* red */

      /* ===== Hide the 'next' column =====
         In your output it's the LAST (5th) column. We hide both by class and by position.
      */
      table.diff td.diff_next,
      table.diff th.diff_next { display: none !important; }

      /* Some versions also define a <col class="diff_next">; make it zero-width as well. */
      table.diff col.diff_next { width: 0 !important; }

      /* Positional fallback: hide the 5th column entirely (works with your screenshot layout) */
      table.diff tr > :nth-child(5) { display: none !important; }

      /* ===== Column width tuning =====
         Layout (your screenshot): 1=ln, 2=code, 3=ln, 4=code, 5=next
         After hiding col 5, set line numbers narrow and code wide.
      */
      table.diff colgroup col:nth-child(1),
      table.diff colgroup col:nth-child(3) { width: 60px; }

      table.diff colgroup col:nth-child(2),
      table.diff colgroup col:nth-child(4) { width: calc((100% - 120px) / 2); }

      /* Ensure line-number cells don't wrap and stay tight */
      table.diff tr > td:nth-child(1),
      table.diff tr > td:nth-child(3) {
        white-space: nowrap;
        color: #6b7280;
      }
    </style>
    """

    return f'{css}<div class="diff-wrap" style="max-height: 520px; overflow:auto;">{table}</div>'



# ----------------------------
# UI actions
# ----------------------------
def load_all(repo_root_str: str) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    Returns:
      enrich_text, bg_block_text, hyp_block_text,
      bg_instruction, bg_extra_examples, hyp_instruction, hyp_extra_examples,
      status
    """
    repo_root = pathlib.Path(repo_root_str).expanduser().resolve()
    files = RepoFiles(repo_root)

    status_msgs = []
    # Load enrich
    try:
        enrich = read_text(files.enrich_daily)
    except Exception as e:
        enrich = ""
        status_msgs.append(f"[ERROR] Failed to read enrich_daily_py.py: {repr(e)}")

    # Load YAML blocks
    try:
        exp_yaml = read_text(files.experiment_prompts)
    except Exception as e:
        exp_yaml = ""
        status_msgs.append(f"[ERROR] Failed to read experiment/prompts.yaml: {repr(e)}")

    try:
        scen_yaml = read_text(files.scenario_prompts)
    except Exception as e:
        scen_yaml = ""
        status_msgs.append(f"[ERROR] Failed to read scenarios/qlib/prompts.yaml: {repr(e)}")

    try:
        bg = extract_yaml_block(exp_yaml, KEY_QLIB_FACTOR_BACKGROUND) if exp_yaml else ""
    except Exception as e:
        bg = ""
        status_msgs.append(
            f"[WARN] Could not locate block '{KEY_QLIB_FACTOR_BACKGROUND}' in experiment/prompts.yaml: {repr(e)}"
        )

    try:
        hyp = extract_yaml_block(scen_yaml, KEY_FACTOR_HYP_SPEC) if scen_yaml else ""
    except Exception as e:
        hyp = ""
        status_msgs.append(
            f"[WARN] Could not locate block '{KEY_FACTOR_HYP_SPEC}' in scenarios/qlib/prompts.yaml: {repr(e)}"
        )

    # Load prompt pack defaults
    pack = load_prompt_pack(repo_root)
    if not pack:
        status_msgs.append(
            f"[WARN] prompt_pack.yaml not found or unreadable at: {files.prompt_pack}"
        )

    bg_i, bg_extra = get_instruction_and_examples_defaults(pack, KEY_QLIB_FACTOR_BACKGROUND) if pack else ("", "")
    hyp_i, hyp_extra = get_instruction_and_examples_defaults(pack, KEY_FACTOR_HYP_SPEC) if pack else ("", "")

    if not status_msgs:
        status_msgs.append("Loaded successfully.")
    status = "\n".join(status_msgs)

    return enrich, bg, hyp, bg_i, bg_extra, hyp_i, hyp_extra, status


def save_enrich(repo_root_str: str, enrich_text: str) -> str:
    files = RepoFiles(pathlib.Path(repo_root_str).expanduser().resolve())
    write_text(files.enrich_daily, enrich_text)
    return f"Saved: {files.enrich_daily}"


def save_bg(repo_root_str: str, bg_text: str) -> str:
    files = RepoFiles(pathlib.Path(repo_root_str).expanduser().resolve())
    yaml_text = read_text(files.experiment_prompts)
    new_yaml = replace_yaml_block(yaml_text, KEY_QLIB_FACTOR_BACKGROUND, bg_text)
    write_text(files.experiment_prompts, new_yaml)
    return f"Saved: {files.experiment_prompts}  (key={KEY_QLIB_FACTOR_BACKGROUND})"


def save_hyp(repo_root_str: str, hyp_text: str) -> str:
    files = RepoFiles(pathlib.Path(repo_root_str).expanduser().resolve())
    yaml_text = read_text(files.scenario_prompts)
    new_yaml = replace_yaml_block(yaml_text, KEY_FACTOR_HYP_SPEC, hyp_text)
    write_text(files.scenario_prompts, new_yaml)
    return f"Saved: {files.scenario_prompts}  (key={KEY_FACTOR_HYP_SPEC})"


def ai_suggest_bg(
    repo_root_str: str,
    current_bg_block: str,
    instruction: str,
    extra_examples: str,
) -> Tuple[str, str, str]:
    repo_root = pathlib.Path(repo_root_str).expanduser().resolve()
    files = RepoFiles(repo_root)

    try:
        current_enrich = read_text(files.enrich_daily)
    except Exception as e:
        return "", "", f"AI suggestion failed: cannot read enrich_daily_py.py: {repr(e)}"

    pack = load_prompt_pack(repo_root)
    if not pack:
        return "", "", f"AI suggestion failed: prompt_pack.yaml missing/unreadable at {files.prompt_pack}"

    try:
        examples_pack = build_examples_pack_text(repo_root, pack, KEY_QLIB_FACTOR_BACKGROUND)
    except Exception as e:
        return "", "", f"AI suggestion failed: cannot load prompt_assets example pack: {repr(e)}"

    try:
        suggested = azure_suggest_revision(
            target_key=KEY_QLIB_FACTOR_BACKGROUND,
            current_block_text=current_bg_block,
            instruction=instruction,
            examples_pack_text=examples_pack,
            extra_examples_text=extra_examples,
            current_enrich_daily_py=current_enrich,
        )
        diff_html = html_diff_table(current_bg_block, suggested, context_lines=3)
        return suggested, diff_html, "AI suggestion generated (original preserved)."
    except Exception as e:
        return "", "", f"AI suggestion failed: {repr(e)}"


def ai_suggest_hyp(
    repo_root_str: str,
    current_hyp_block: str,
    instruction: str,
    extra_examples: str,
) -> Tuple[str, str, str]:
    repo_root = pathlib.Path(repo_root_str).expanduser().resolve()
    files = RepoFiles(repo_root)

    try:
        current_enrich = read_text(files.enrich_daily)
    except Exception as e:
        return "", "", f"AI suggestion failed: cannot read enrich_daily_py.py: {repr(e)}"

    pack = load_prompt_pack(repo_root)
    if not pack:
        return "", "", f"AI suggestion failed: prompt_pack.yaml missing/unreadable at {files.prompt_pack}"

    try:
        examples_pack = build_examples_pack_text(repo_root, pack, KEY_FACTOR_HYP_SPEC)
    except Exception as e:
        return "", "", f"AI suggestion failed: cannot load prompt_assets example pack: {repr(e)}"

    try:
        suggested = azure_suggest_revision(
            target_key=KEY_FACTOR_HYP_SPEC,
            current_block_text=current_hyp_block,
            instruction=instruction,
            examples_pack_text=examples_pack,
            extra_examples_text=extra_examples,
            current_enrich_daily_py=current_enrich,
        )
        diff_html = html_diff_table(current_hyp_block, suggested, context_lines=3)
        return suggested, diff_html, "AI suggestion generated (original preserved)."
    except Exception as e:
        return "", "", f"AI suggestion failed: {repr(e)}"


# ----------------------------
# UI actions - generate h5
# ----------------------------
def _syntax_check_python_file(py_path: pathlib.Path) -> Tuple[bool, str]:
    """
    Check Python syntax via compile(). Does not execute code.
    """
    try:
        src = py_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"[ERROR] Cannot read file: {py_path}\n{repr(e)}"

    try:
        compile(src, str(py_path), "exec")
        return True, f"[OK] Syntax check passed: {py_path}"
    except SyntaxError as e:
        # Show near lines for debugging
        lines = src.splitlines()
        start = max((e.lineno or 1) - 3, 1)
        end = min((e.lineno or 1) + 2, len(lines))
        snippet = "\n".join(f"{i:>4}: {lines[i-1]}" for i in range(start, end + 1))
        return False, (
            f"[ERROR] SyntaxError in {py_path}\n"
            f"  line={e.lineno}, offset={e.offset}\n"
            f"  msg={e.msg}\n\n"
            f"Context:\n{snippet}\n"
        )
    except Exception as e:
        return False, f"[ERROR] Unexpected error during syntax check: {repr(e)}"


def _find_recent_h5(dir_path: pathlib.Path, since_ts: float) -> str:
    """
    Find .h5 files modified after since_ts; return formatted summary.
    """
    if not dir_path.exists():
        return f"[WARN] Directory does not exist: {dir_path}"

    h5_files = []
    for p in dir_path.rglob("*.h5"):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            continue
        if mtime >= since_ts:
            h5_files.append((mtime, p))

    if not h5_files:
        return "[INFO] No newly modified .h5 files detected after generation."

    h5_files.sort(key=lambda x: x[0], reverse=True)
    lines = ["[OK] Newly generated/updated .h5 files:"]
    for mtime, p in h5_files[:20]:
        lines.append(f"  - {p}  (mtime={datetime.fromtimestamp(mtime).isoformat(timespec='seconds')})")
    if len(h5_files) > 20:
        lines.append(f"  ... and {len(h5_files)-20} more")
    return "\n".join(lines)


def run_generate_dataset(repo_root_str: str) -> str:
    """
    1) Syntax-check enrich_daily_py.py (the user's manual edits).
    2) Run generate.py under factor_data_template directory.
    3) Report logs + newly generated h5 files.
    """
    repo_root = pathlib.Path(repo_root_str).expanduser().resolve()
    files = RepoFiles(repo_root)

    factor_dir = files.enrich_daily.parent  # .../factor_data_template
    generate_py = factor_dir / "generate.py"

    log_lines = []
    log_lines.append(f"[INFO] Repo root: {repo_root}")
    log_lines.append(f"[INFO] Factor template dir: {factor_dir}")
    log_lines.append(f"[INFO] Target enrich file: {files.enrich_daily}")
    log_lines.append(f"[INFO] generate.py: {generate_py}")

    # 0) Basic existence checks
    if not files.enrich_daily.exists():
        return "\n".join(log_lines + [f"[ERROR] enrich_daily_py.py not found: {files.enrich_daily}"])
    if not generate_py.exists():
        return "\n".join(log_lines + [f"[ERROR] generate.py not found: {generate_py}"])

    # 1) Syntax check
    ok, msg = _syntax_check_python_file(files.enrich_daily)
    log_lines.append(msg)
    if not ok:
        log_lines.append("[ABORT] Fix syntax errors, then retry.")
        return "\n".join(log_lines)

    # 2) Run generate.py
    start_ts = time.time()
    log_lines.append("[INFO] Running: python generate.py")
    try:
        proc = subprocess.run(
            [sys.executable, "generate.py"],
            cwd=str(factor_dir),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        log_lines.append(f"[INFO] Return code: {proc.returncode}")
        if proc.stdout:
            log_lines.append("----- STDOUT -----")
            log_lines.append(proc.stdout.strip())
        if proc.stderr:
            log_lines.append("----- STDERR -----")
            log_lines.append(proc.stderr.strip())

        if proc.returncode != 0:
            log_lines.append("[ERROR] generate.py failed. See STDERR above.")
            return "\n".join(log_lines)

    except Exception as e:
        log_lines.append(f"[ERROR] Failed to run generate.py: {repr(e)}")
        return "\n".join(log_lines)

    # 3) Locate new/updated h5 files (best-effort)
    # Usually output is within factor_data_template or its subdirs; scan that dir.
    log_lines.append(_find_recent_h5(factor_dir, since_ts=start_ts))

    return "\n".join(log_lines)

# ----------------------------
# Build Gradio app
# ----------------------------
def build_app():
    with gr.Blocks(title="RD-Agent Factor Editor (add_factors)") as demo:
        gr.Markdown(
            "# RD-Agent Factor Editor (add_factors)\n"
            "需要修改这些文件:\n"
            f"- `{REL_ENRICH_DAILY}` (manual)\n"
            f"- `{REL_EXPERIMENT_PROMPTS}` → `{KEY_QLIB_FACTOR_BACKGROUND}: |...`\n"
            f"- `{REL_SCENARIO_PROMPTS}` → `{KEY_FACTOR_HYP_SPEC}: |...`\n\n"
            "1. 修改 enrich_daily_py.py **并保存** 后，就可以通过修改后的代码询问 AI，如何相应的修改prompt \n"
            "2. AI 的建议不会直接覆盖prompts，除非点击“Apply AI suggestion → current editor” button \n"
            "3. 对于任何修改，记得点击对应的 **Save** button 保存到磁盘 \n"
        )

        repo_root = gr.Textbox(
            label="Repo root directory",
            value=str(pathlib.Path(".").resolve()),
        )

        with gr.Row():
            btn_load = gr.Button("Load from disk", variant="primary")
            status = gr.Textbox(label="Status", value="", interactive=False)

        with gr.Tabs():
            # ----------------------------
            # Tab 1: enrich_daily_py.py
            # ----------------------------
            with gr.Tab("enrich_daily_py.py (manual)"):
                enrich_editor = gr.Code(
                    label="enrich_daily_py.py",
                    language="python",
                    lines=28,
                )
                with gr.Row():
                    btn_save_enrich = gr.Button(
                        "Save enrich_daily_py.py", variant="primary"
                    )
                    btn_run_generate = gr.Button("Run generate.py (syntax-check + build h5)", variant="secondary")
                save_msg_enrich = gr.Textbox(label="Save result", interactive=False)
                gen_log = gr.Textbox(label="Generate log",lines=18,interactive=False)

            # ----------------------------
            # Tab 2: qlib_factor_background
            # ----------------------------
            with gr.Tab("qlib_factor_background (YAML block)"):
                bg_editor = gr.Textbox(
                    label="qlib_factor_background content (current)",
                    lines=20,
                )

                gr.Markdown(
                    "## AI assist (Azure OpenAI)\n"
                    "- 点击 AI suggest，可以获得修改 prompt 的建议\n"
                    "- AI suggestions 不会直接覆盖当前编辑器\n"
                    "- 查看 AI 建议前后的差异。如果满意，可直接点击 “Apply AI suggestion → current editor” button。如不满意，也可以参考并手动编辑。\n"
                )

                with gr.Accordion("Show AI instruction / extra examples", open=False):
                    bg_instruction = gr.Textbox(
                        label="Instruction to AI (loaded from prompt_pack.yaml; you can override here)",
                        lines=5,
                    )
                    bg_extra_examples = gr.Textbox(
                        label="Extra examples / notes (optional; appended after the fixed example pack)",
                        lines=6,
                        value="",                    
                    )

                with gr.Row():
                    btn_ai_bg = gr.Button("AI suggest revision (background)")
                    btn_apply_bg = gr.Button("Apply AI suggestion → current editor")
                    btn_save_bg = gr.Button(
                        "Save experiment/prompts.yaml", variant="primary"
                    )

                bg_status = gr.Textbox(label="Result", interactive=False)

                with gr.Row():
                    bg_suggested = gr.Textbox(
                        label="AI suggested background (not applied)",
                        lines=20,
                        interactive=True,
                    )
                with gr.Row():
                    bg_diff = gr.HTML(label="Diff (current ↔ suggested) [HTML]")

            # ----------------------------
            # Tab 3: factor_hypothesis_specification
            # ----------------------------
            with gr.Tab("factor_hypothesis_specification (YAML block)"):
                hyp_editor = gr.Textbox(
                    label="factor_hypothesis_specification content (current)",
                    lines=20,
                )

                gr.Markdown(
                    "## AI assist (Azure OpenAI)\n"
                    "- 点击 **AI suggest** 生成候选修订版本。\n"
                    "- AI suggestions 不会直接覆盖当前编辑器。\n"
                    "- 查看 AI 建议前后的差异。如果满意，可直接点击 “Apply AI suggestion → current editor” button。如不满意，也可以参考并手动编辑。\n"
                )

                with gr.Accordion("Show AI instruction / extra examples", open=False):
                    hyp_instruction = gr.Textbox(
                        label="Instruction to AI (loaded from prompt_pack.yaml; you can override here)",
                        lines=5,
                    )
                    hyp_extra_examples = gr.Textbox(
                        label="Extra examples / notes (optional; appended after the fixed example pack)",
                        lines=6,
                        value="",
                    )

                with gr.Row():
                    btn_ai_hyp = gr.Button("AI suggest revision (spec)")
                    btn_apply_hyp = gr.Button("Apply AI suggestion → current editor")
                    btn_save_hyp = gr.Button(
                        "Save scenarios/qlib/prompts.yaml", variant="primary"
                    )

                hyp_status = gr.Textbox(label="Result", interactive=False)

                with gr.Row():
                    hyp_suggested = gr.Textbox(
                        label="AI suggested spec (not applied)",
                        lines=20,
                        interactive=True,
                    )
                with gr.Row():
                    hyp_diff = gr.HTML(label="Diff (current ↔ suggested) [HTML]")

        # ----------------------------
        # Wiring
        # ----------------------------
        btn_load.click(
            fn=load_all,
            inputs=[repo_root],
            outputs=[
                enrich_editor,
                bg_editor,
                hyp_editor,
                bg_instruction,
                bg_extra_examples,
                hyp_instruction,
                hyp_extra_examples,
                status,
            ],
        )

        btn_save_enrich.click(
            fn=save_enrich,
            inputs=[repo_root, enrich_editor],
            outputs=[save_msg_enrich],
        )
        btn_run_generate.click(
            fn=run_generate_dataset,
            inputs=[repo_root],
            outputs=[gen_log],
        )

        # Background: AI suggest → suggested + diff (no overwrite)
        btn_ai_bg.click(
            fn=ai_suggest_bg,
            inputs=[repo_root, bg_editor, bg_instruction, bg_extra_examples],
            outputs=[bg_suggested, bg_diff, bg_status],
        )

        # Apply background suggestion to current editor
        btn_apply_bg.click(
            fn=lambda s: s or "",
            inputs=[bg_suggested],
            outputs=[bg_editor],
        )

        # Save background editor to YAML
        btn_save_bg.click(
            fn=save_bg,
            inputs=[repo_root, bg_editor],
            outputs=[bg_status],
        )

        # Hypothesis: AI suggest → suggested + diff (no overwrite)
        btn_ai_hyp.click(
            fn=ai_suggest_hyp,
            inputs=[repo_root, hyp_editor, hyp_instruction, hyp_extra_examples],
            outputs=[hyp_suggested, hyp_diff, hyp_status],
        )

        # Apply hypothesis suggestion to current editor
        btn_apply_hyp.click(
            fn=lambda s: s or "",
            inputs=[hyp_suggested],
            outputs=[hyp_editor],
        )

        # Save hypothesis editor to YAML
        btn_save_hyp.click(
            fn=save_hyp,
            inputs=[repo_root, hyp_editor],
            outputs=[hyp_status],
        )

    return demo




if __name__ == "__main__":
    app = build_app()
    # Avoid version-specific args like show_api; keep launch minimal for compatibility.
    app.launch(server_name="127.0.0.1", server_port=7860)
