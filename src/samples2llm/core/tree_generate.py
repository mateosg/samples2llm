from __future__ import annotations

from pathlib import Path


def generate_directory_tree(paths: list[Path], root_dir: Path) -> str:
    """Version simplificada de fileTreeGenerate.ts: construye un arbol de
    texto indentado a partir de las rutas relativas de los samples
    encontrados. No pretende ser un algoritmo de arbol optimizado, solo
    legible."""
    tree: dict = {}
    for p in paths:
        parts = p.relative_to(root_dir).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    def render(node: dict, prefix: str = "") -> list[str]:
        lines = []
        for name in sorted(node.keys()):
            is_dir = bool(node[name])
            lines.append(f"{prefix}{name}{'/' if is_dir else ''}")
            if is_dir:
                lines.extend(render(node[name], prefix + "  "))
        return lines

    return "\n".join(render(tree))
