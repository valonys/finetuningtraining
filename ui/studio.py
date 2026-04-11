"""
ValonyLabs Studio v3.0 — Gradio UI.

Talks to the FastAPI backend (default: http://localhost:8000) via HTTP so the
UI is completely decoupled from the training/inference engines. You can run
the UI on your laptop and point VALONY_API_URL at a remote Brev instance.

The UI is fully domain-agnostic — it never hardcodes a domain name. You create
domains in the "🏷️ Domains" tab (or via CLI / API), and every other tab picks
them up from a refreshable dropdown.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

API_URL = os.environ.get("VALONY_API_URL", "http://localhost:8000")


def _call(path: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    url = f"{API_URL}{path}"
    r = requests.request(method, url, timeout=600, **kwargs)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code} {r.reason}: {detail}")
    return r.json() if r.content else {}


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────
def check_health() -> str:
    try:
        h = _call("/healthz")
    except Exception as e:
        return f"❌ API unreachable: {e}"
    return (
        f"**Status:** {h['status']} (v{h['version']})\n\n"
        f"**Hardware tier:** `{h['hardware']['tier']}` — {h['hardware']['device_name']}\n\n"
        f"**Effective memory:** {h['hardware']['effective_memory_gb']} GB\n\n"
        f"**Training backend:** `{h['profile']['training_backend']}`\n\n"
        f"**Inference backend:** `{h['inference_backend']}` "
        f"(`{h['profile']['inference_backend']}` preferred)\n\n"
        f"**Trained adapters:** {', '.join(h['registered_domains']) or '(none yet)'}\n\n"
        f"**OCR engines:** {', '.join(h['available_ocr'])}\n\n"
        f"**Templates:** {', '.join(h['available_templates'])}\n\n"
        f"**Latency stats:** `{h['latency_stats']}`"
    )


# ──────────────────────────────────────────────────────────────
# Domains — list, create, show
# ──────────────────────────────────────────────────────────────
def list_domains() -> Tuple[List[str], List[str]]:
    try:
        resp = _call("/v1/domains/configs")
        return resp.get("configs", []), resp.get("examples", [])
    except Exception:
        return [], []


def domains_refresh_markdown() -> str:
    configs, examples = list_domains()
    lines = ["### 📂 Your domains (`configs/domains/`)", ""]
    if configs:
        for c in configs:
            lines.append(f"- **{c}** — `configs/domains/{c}.yaml`")
    else:
        lines.append("_No domain configs yet — create one below._")
    lines += ["", "### 🌱 Seed examples (`configs/domains/examples/`)", ""]
    if examples:
        for e in examples:
            lines.append(f"- `{e}` _(copy with the form below — set **Copy from** to `{e}`)_")
    else:
        lines.append("_No examples shipped._")
    return "\n".join(lines)


def domains_create(
    name: str,
    system_prompt: str,
    constitution_text: str,
    copy_from: str,
    overwrite: bool,
):
    name = (name or "").strip()
    copy_from = (copy_from or "").strip() or None
    if not name:
        return "⚠️  Please provide a domain name.", gr.update(), gr.update()

    constitution = [
        line.strip() for line in (constitution_text or "").splitlines() if line.strip()
    ]
    body: Dict[str, Any] = {
        "name": name,
        "system_prompt": system_prompt or None,
        "constitution": constitution or None,
        "overwrite": overwrite,
    }
    if copy_from:
        body["copy_from"] = copy_from

    try:
        resp = _call("/v1/domains/configs", method="POST", json=body)
    except Exception as e:
        return f"❌ {e}", gr.update(), gr.update()

    cfg_preview = json.dumps(resp.get("config", {}), indent=2, ensure_ascii=False)
    md = (
        f"✅ Created domain **{resp['name']}** at `{resp['path']}`\n\n"
        f"```yaml\n{cfg_preview}\n```"
    )
    # Refresh the domains dropdown on the Train tab + the list on the Domains tab
    configs, _ = list_domains()
    return md, gr.update(choices=configs, value=name), gr.update(value=domains_refresh_markdown())


def domain_show(name: str) -> str:
    if not name:
        return "_Select a domain to preview._"
    try:
        resp = _call(f"/v1/domains/configs/{name}")
    except Exception as e:
        return f"❌ {e}"
    return f"```yaml\n{json.dumps(resp.get('config', {}), indent=2, ensure_ascii=False)}\n```"


# ──────────────────────────────────────────────────────────────
# Data Forge
# ──────────────────────────────────────────────────────────────
def forge_build(files: List[str], task: str, base_model: str, system_prompt: str,
                synth_qa: bool, target_size: Optional[float]):
    if not files:
        return "⚠️  Please upload at least one file."
    try:
        resp = _call(
            "/v1/forge/build_dataset",
            method="POST",
            json={
                "paths": files,
                "task": task,
                "base_model": base_model,
                "system_prompt": system_prompt,
                "synth_qa": synth_qa,
                "target_size": int(target_size) if target_size else None,
            },
        )
    except Exception as e:
        return f"❌ Build failed: {e}"
    return (
        f"✅ **Built `{resp['task']}` dataset** with template `{resp['template']}`\n\n"
        f"**Examples:** {resp['num_examples']}\n\n"
        f"**Saved to:** `{resp['output_path']}`\n\n"
        f"**Sources:** {len(resp['sources'])} file(s)"
    )


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def start_training(domain_name: str, base_model: str, method: str, dataset_path: str):
    if not domain_name:
        return "⚠️  Pick a domain (create one first in the 🏷️ Domains tab)."
    if not dataset_path:
        return "⚠️  Please provide a dataset path."
    try:
        resp = _call(
            "/v1/jobs/create",
            method="POST",
            json={
                "domain_config_name": domain_name,
                "base_model": base_model,
                "training_method": method,
                "dataset_path": dataset_path,
            },
        )
    except Exception as e:
        return f"❌ Failed to queue job: {e}"
    return (
        f"✅ Job queued: `{resp['job_id']}` (status: {resp['status']})\n\n"
        f"**Domain:** {domain_name}\n\n"
        f"**Method:** {method}\n\n"
        f"**Base model:** {base_model}"
    )


def poll_job(job_id: str):
    if not job_id:
        return "Enter a job id."
    try:
        j = _call(f"/v1/jobs/{job_id}")
    except Exception as e:
        return f"❌ {e}"
    lines = [
        f"**Job** `{j['job_id']}`  ({j['status']})",
        f"**Progress**: {j['progress']:.0%}",
        f"**Method**: {j.get('method')}",
        f"**Backend**: {j.get('backend')}",
        f"**Template**: {j.get('template')}",
        f"**Hardware**: {j.get('hardware')}",
    ]
    if j.get("final_loss") is not None:
        lines.append(f"**Final loss**: {j['final_loss']:.4f}")
    if j.get("error_message"):
        lines.append(f"❌ **Error**: {j['error_message']}")
    if j.get("adapter_path"):
        lines.append(f"**Adapter**: `{j['adapter_path']}`")
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────────────────────
def chat(message: str, domain: str, temperature: float, max_new_tokens: int):
    try:
        resp = _call(
            "/v1/chat",
            method="POST",
            json={
                "message": message,
                "domain_config_name": domain or "base",
                "temperature": temperature,
                "max_new_tokens": int(max_new_tokens),
            },
        )
    except Exception as e:
        return f"❌ {e}", ""
    telemetry = (
        f"backend=`{resp['backend']}` | "
        f"ttft={resp['ttft_ms']:.1f} ms | "
        f"latency={resp['latency_ms']:.1f} ms | "
        f"tokens={resp['tokens_generated']} ({resp['tokens_per_second']:.1f} tok/s)"
    )
    return resp["response"], telemetry


def refresh_dropdowns():
    configs, examples = list_domains()
    return (
        gr.update(choices=configs, value=configs[0] if configs else None),
        gr.update(choices=["(none — start blank)"] + examples, value="(none — start blank)"),
        gr.update(choices=["base"] + configs, value="base"),
    )


# ──────────────────────────────────────────────────────────────
# Build the UI
# ──────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ValonyLabs Studio v3.0", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# 🛠️ ValonyLabs Studio v3.0\n"
            "*Agnostic post-training + inference for Mac / RTX / Colab / Brev. "
            "Pick any domain you want — **`asset_integrity`**, **`customer_grasps`**, "
            "**`ai_llm`**, **`legal_nda_review`**, whatever fits your engagement.*"
        )

        # ── Health ────────────────────────────────────────────
        with gr.Tab("🩺 Health"):
            health_out = gr.Markdown()
            gr.Button("Refresh").click(fn=check_health, outputs=health_out)
            ui.load(fn=check_health, outputs=health_out)

        # ── Domains ───────────────────────────────────────────
        with gr.Tab("🏷️ Domains"):
            gr.Markdown(
                "Create one domain per engagement. Each becomes its own YAML "
                "under `configs/domains/<name>.yaml` and produces its own "
                "adapter at `outputs/<name>/` after training."
            )
            domains_list = gr.Markdown(value=domains_refresh_markdown())
            gr.Button("Refresh list").click(
                fn=domains_refresh_markdown, outputs=domains_list
            )

            gr.Markdown("### ➕ Create a new domain")
            with gr.Row():
                d_name = gr.Textbox(
                    label="Domain name",
                    placeholder="e.g., asset_integrity, customer_grasps, ai_llm, legal_nda_review",
                )
                d_copy = gr.Dropdown(
                    label="Copy from example (optional)",
                    choices=["(none — start blank)"] + list_domains()[1],
                    value="(none — start blank)",
                    allow_custom_value=True,
                )
            d_system = gr.Textbox(
                label="System prompt (the persona the model should adopt)",
                lines=4,
                placeholder="You are a senior <role> specialising in <area>...",
            )
            d_rules = gr.Textbox(
                label="Constitution — one rule per line (optional)",
                lines=5,
                placeholder="Always cite the relevant standard.\nNever speculate without noting uncertainty.",
            )
            d_overwrite = gr.Checkbox(value=False, label="Overwrite if the domain already exists")
            d_submit = gr.Button("Create domain", variant="primary")
            d_result = gr.Markdown()

            gr.Markdown("### 🔎 Preview an existing domain")
            d_select = gr.Dropdown(
                label="Existing domains",
                choices=list_domains()[0],
                allow_custom_value=False,
            )
            d_preview = gr.Markdown()
            d_select.change(fn=domain_show, inputs=d_select, outputs=d_preview)

        # ── Data Forge ────────────────────────────────────────
        with gr.Tab("🧱 Data Forge"):
            files = gr.File(file_count="multiple", type="filepath",
                            label="Upload PDFs, DOCX, XLSX, PPTX, images, HTML, TXT, CSV")
            task = gr.Radio(["sft", "dpo", "orpo", "kto", "grpo"], value="sft", label="Task")
            base_model_f = gr.Textbox(value="Qwen/Qwen2.5-7B-Instruct", label="Base model ID")
            system_prompt_f = gr.Textbox(
                value="You are a helpful assistant.",
                label="System prompt (leave default or use the one from your domain)",
                lines=2,
            )
            synth_qa = gr.Checkbox(value=True, label="Synthesize Q/A pairs")
            target_size = gr.Number(value=500, label="Target size (rows)", precision=0)
            btn = gr.Button("Build dataset", variant="primary")
            forge_out = gr.Markdown()
            btn.click(
                fn=forge_build,
                inputs=[files, task, base_model_f, system_prompt_f, synth_qa, target_size],
                outputs=forge_out,
            )

        # ── Train ─────────────────────────────────────────────
        with gr.Tab("🏋️ Train"):
            gr.Markdown(
                "Pick a **domain** (create it in the 🏷️ Domains tab first), "
                "a base model, a training method, and a dataset path. "
                "Template + hardware backend + LoRA rank are auto-resolved."
            )
            t_domain = gr.Dropdown(
                label="Domain",
                choices=list_domains()[0],
                allow_custom_value=False,
            )
            t_model = gr.Textbox(value="Qwen/Qwen2.5-7B-Instruct", label="Base model")
            method = gr.Radio(
                ["sft", "dpo", "orpo", "kto", "grpo"], value="sft", label="Method"
            )
            ds_path = gr.Textbox(label="Dataset path (local folder or .json)")
            btn_train = gr.Button("Queue training job", variant="primary")
            train_out = gr.Markdown()
            btn_train.click(
                fn=start_training,
                inputs=[t_domain, t_model, method, ds_path],
                outputs=train_out,
            )

            gr.Markdown("---")
            gr.Markdown("### Job status")
            job_id = gr.Textbox(label="Job ID")
            btn_poll = gr.Button("Poll status")
            job_out = gr.Markdown()
            btn_poll.click(fn=poll_job, inputs=job_id, outputs=job_out)

        # ── Chat ──────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            msg = gr.Textbox(label="Message", lines=4)
            dom = gr.Dropdown(
                label="Domain (or 'base' for the raw base model)",
                choices=["base"] + list_domains()[0],
                value="base",
                allow_custom_value=True,
            )
            temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            max_new = gr.Slider(16, 4096, value=512, step=16, label="Max new tokens")
            btn_chat = gr.Button("Send", variant="primary")
            chat_out = gr.Markdown()
            telemetry = gr.Markdown()
            btn_chat.click(
                fn=chat,
                inputs=[msg, dom, temp, max_new],
                outputs=[chat_out, telemetry],
            )

        # ── Cross-tab plumbing ────────────────────────────────
        # When a new domain is created, refresh dropdowns on Train + Chat tabs.
        d_submit.click(
            fn=domains_create,
            inputs=[d_name, d_system, d_rules, d_copy, d_overwrite],
            outputs=[d_result, t_domain, domains_list],
        )
        # After creating, also refresh the chat dropdown + preview dropdown.
        d_submit.click(
            fn=refresh_dropdowns,
            inputs=None,
            outputs=[d_select, d_copy, dom],
        )

    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=False)
