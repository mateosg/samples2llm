from __future__ import annotations

from samples2llm.config.schema import SampleRecord


def generate_markdown(records: list[SampleRecord], directory_tree: str) -> str:
    lines = ["# Sample Library", "", "## Directory Structure", "", "```", directory_tree, "```", ""]
    lines += ["## Samples", ""]
    lines += ["| Path | Summary | Tags | Flags |",
              "|---|---|---|---|"]

    for r in records:
        tags = ", ".join(r.derived_tags) or "-"
        flags = ", ".join(r.confidence_flags) or "-"
        lines.append(f"| {r.path} | {r.summary or '-'} | {tags} | {flags} |")

    return "\n".join(lines)
