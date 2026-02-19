import marimo

app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    root = Path(__file__).resolve().parent
    algo_files = {
        "server.py (core matching + cycles + trust)": root / "server.py",
        "wave_tessellation_fulltilt.py (wave/tessellation runner)": root
        / "wave_tessellation_fulltilt.py",
        "kontur_partition_bridge_benchmark.py (Kontur population runner)": root
        / "kontur_partition_bridge_benchmark.py",
    }
    return algo_files, mo


@app.cell
def _(algo_files, mo):
    available_paths = [str(path) for path in algo_files.values()]
    picker = mo.ui.dropdown(
        options=available_paths,
        value=available_paths[0] if available_paths else None,
        label="Algorithm file",
    )
    start_line = mo.ui.number(value=1, start=1, step=1, label="Start line")
    line_count = mo.ui.number(value=220, start=20, step=20, label="Line count")
    show_full = mo.ui.checkbox(value=False, label="Show full file")

    mo.md("# ShareWith Algorithm Source Viewer")
    mo.vstack([picker, show_full, start_line, line_count])
    return line_count, picker, show_full, start_line


@app.cell
def _(line_count, mo, picker, show_full, start_line):
    from pathlib import Path as _Path

    selected_path = _Path(picker.value) if picker.value else None
    if selected_path is None or not selected_path.exists():
        output_md = "No file selected."
    else:
        raw = selected_path.read_text(encoding="utf-8", errors="replace")
        file_lines = raw.splitlines()

        if show_full.value:
            start = 1
            end = len(file_lines)
        else:
            start = max(1, int(start_line.value))
            count = max(1, int(line_count.value))
            end = min(len(file_lines), start + count - 1)

        suffix = selected_path.suffix.lower()
        if suffix == ".py":
            lang = "python"
        elif suffix == ".md":
            lang = "markdown"
        else:
            lang = ""

        code = "\n".join(
            f"{idx + 1:5d}: {file_lines[idx]}" for idx in range(start - 1, end)
        )
        output_md = (
            f"## `{selected_path}`\n"
            f"Showing lines `{start}` to `{end}` of `{len(file_lines)}`.\n\n"
            f"```{lang}\n{code}\n```"
        )

    mo.md(output_md)
    return


if __name__ == "__main__":
    app.run()
