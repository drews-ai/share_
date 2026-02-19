import marimo

__generated_with = "0.0.0"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import json
    from pathlib import Path

    import marimo as mo

    project_root = Path(__file__).resolve().parent
    artifacts_root = project_root / "artifacts"
    thesis_report = project_root / "THESIS_CLAIMS_END_TO_END_EVAL_2026-02-19.md"

    return artifacts_root, csv, json, mo, project_root, thesis_report


@app.cell
def _(mo, project_root, thesis_report):
    title = [
        "# ShareWith Citation Workbench",
        "",
        "Use this as a source-of-truth view for claim-level evidence, code, and run artifacts.",
        "",
        f"- Project root: `{project_root}`",
        f"- Latest thesis claim audit: `{thesis_report}`",
    ]
    mo.md("\n".join(title))
    return


@app.cell
def _(project_root):
    citations = [
        {
            "claim": "Weighted trust/patience edge scoring",
            "verdict": "Supported",
            "evidence": f"{project_root / 'server.py'}:864",
        },
        {
            "claim": "Hungarian optimal assignment",
            "verdict": "Supported",
            "evidence": f"{project_root / 'server.py'}:982",
        },
        {
            "claim": "Cycle detection + atomic commit/rollback",
            "verdict": "Supported",
            "evidence": f"{project_root / 'server.py'}:1138, {project_root / 'server.py'}:1271, {project_root / 'server.py'}:1336",
        },
        {
            "claim": "Patience-wave expansion across tessellations",
            "verdict": "Supported",
            "evidence": f"{project_root / 'wave_tessellation_fulltilt.py'}:325, {project_root / 'wave_tessellation_fulltilt.py'}:331, {project_root / 'wave_tessellation_fulltilt.py'}:367",
        },
        {
            "claim": "Partition-first + bridge matching",
            "verdict": "Supported",
            "evidence": f"{project_root / 'wave_tessellation_fulltilt.py'}:578, {project_root / 'wave_tessellation_fulltilt.py'}:629, {project_root / 'wave_tessellation_fulltilt.py'}:710",
        },
        {
            "claim": "40M tractability proved empirically",
            "verdict": "Not proven yet",
            "evidence": f"{project_root / 'THESIS_CLAIMS_END_TO_END_EVAL_2026-02-19.md'}:102",
        },
        {
            "claim": "Guild trust floor in active runtime",
            "verdict": "Partial",
            "evidence": f"{project_root / 'THESIS_CLAIMS_END_TO_END_EVAL_2026-02-19.md'}:125",
        },
    ]
    return (citations,)


@app.cell
def _(citations, mo):
    claim_matrix_lines = [
        "## Claim Matrix (Citation-Level)",
        "",
        "| Claim | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for item in citations:
        claim_matrix_lines.append(
            f"| {item['claim']} | **{item['verdict']}** | `{item['evidence']}` |"
        )
    mo.md("\n".join(claim_matrix_lines))
    return


@app.cell
def _(project_root):
    core_files = [
        project_root / "server.py",
        project_root / "wave_tessellation_fulltilt.py",
        project_root / "test_sharewith_engine.py",
        project_root / "test_wave_thesis_tdd.py",
        project_root / "THESIS_CLAIMS_END_TO_END_EVAL_2026-02-19.md",
        project_root / "SHARING_ALGORITHM.md",
    ]
    available = [path for path in core_files if path.exists()]
    options = {str(path.relative_to(project_root)): str(path) for path in available}
    return options, project_root


@app.cell
def _(mo, options):
    file_picker = mo.ui.dropdown(
        options=options,
        value=next(iter(options.values())) if options else None,
        label="Source file",
    )
    start_line = mo.ui.number(value=1, start=1, step=1, label="Start line")
    line_count = mo.ui.number(value=180, start=20, step=20, label="Line count")

    mo.md("## Code Browser")
    mo.vstack([file_picker, start_line, line_count])
    return file_picker, line_count, start_line


@app.cell
def _(file_picker, line_count, mo, start_line):
    from pathlib import Path as _Path

    selected_path = _Path(file_picker.value) if file_picker.value else None
    if selected_path is None or not selected_path.exists():
        selected_path = None

    if selected_path is None:
        code_browser_md = "No file selected."
    else:
        text = selected_path.read_text(encoding="utf-8", errors="replace")
        file_lines = text.splitlines()
        start = max(1, int(start_line.value))
        count = max(1, int(line_count.value))
        end = min(len(file_lines), start + count - 1)

        suffix = selected_path.suffix.lower()
        if suffix == ".py":
            lang = "python"
        elif suffix in {".md", ".markdown"}:
            lang = "markdown"
        elif suffix == ".json":
            lang = "json"
        elif suffix == ".csv":
            lang = "csv"
        else:
            lang = ""

        snippet = "\n".join(
            f"{idx + 1:5d}: {file_lines[idx]}" for idx in range(start - 1, end)
        )
        code_browser_md = (
            f"### `{selected_path}`\n"
            f"Showing lines `{start}` to `{end}` of `{len(file_lines)}`.\n\n"
            f"```{lang}\n{snippet}\n```"
        )

    mo.md(code_browser_md)
    return


@app.cell
def _(artifacts_root, json, mo):
    summary_paths = [
        artifacts_root / "thesis_claims_eval_2026_02_19/reval_wave_global_27k_w6/summary.json",
        artifacts_root / "thesis_claims_eval_2026_02_19/reval_wave_pb_27k_w6/summary.json",
        artifacts_root / "wave_fulltilt_fixed/summary.json",
        artifacts_root / "kontur_partition_bridge/comparison_snapshot_2026-02-19.json",
    ]

    summary_md_lines = ["## Key Run Summaries", ""]
    for summary_path in summary_paths:
        if not summary_path.exists():
            summary_md_lines.append(f"- Missing: `{summary_path}`")
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_md_lines.append(f"### `{summary_path.name}`")
        if "mean_week_runtime_s" in payload:
            summary_md_lines.append(
                f"- mean_week_runtime_s: `{payload['mean_week_runtime_s']}`"
            )
        if "mean_completed_per_1000_active" in payload:
            summary_md_lines.append(
                f"- mean_completed_per_1000_active: `{payload['mean_completed_per_1000_active']}`"
            )
        if "mean_unmet_ratio" in payload:
            summary_md_lines.append(
                f"- mean_unmet_ratio: `{payload['mean_unmet_ratio']}`"
            )
        if "total_agents" in payload:
            summary_md_lines.append(f"- total_agents: `{payload['total_agents']}`")
        if "loaded" in payload:
            summary_md_lines.append("- contains multi-run snapshot under `loaded`")
        summary_md_lines.append("")

    mo.md("\n".join(summary_md_lines))
    return


@app.cell
def _(mo, project_root):
    mo.md(
        "## Run Commands\n"
        "```bash\n"
        f"cd {project_root}\n"
        ".venv/bin/python -m marimo run marimo_sharewith_citation_workbench.py --host 127.0.0.1 --port 2718\n"
        "```\n"
    )
    return


if __name__ == "__main__":
    app.run()
