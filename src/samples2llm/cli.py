from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from samples2llm.config.schema import OutputStyle, Samples2LlmConfig
from samples2llm.core.packager import pack

app = typer.Typer(add_completion=False)


@app.command()
def main(
    directory: Path = typer.Argument(..., help="Sample folder to process"),
    style: OutputStyle = typer.Option(OutputStyle.JSON, "--style", help="json or markdown"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file path"),
    no_tonal: bool = typer.Option(False, "--no-tonal", help="Disable pitch/tonal detection (faster)"),
    no_rhythmic: bool = typer.Option(False, "--no-rhythmic", help="Disable tempo/rhythmic detection"),
) -> None:
    config = Samples2LlmConfig()
    config.output.style = style
    config.extraction.enable_tonal = not no_tonal
    config.extraction.enable_rhythmic = not no_rhythmic

    result = pack(directory, config)

    if output:
        output.write_text(result, encoding="utf-8")
        typer.echo(f"Written to {output}")
    else:
        typer.echo(result)


if __name__ == "__main__":
    app()
