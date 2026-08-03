from __future__ import annotations

from pathlib import Path

from samples2llm.core.tree_generate import generate_directory_tree


def test_generate_directory_tree_includes_nested_structure(tmp_path: Path) -> None:
    p1 = tmp_path / "Drums" / "Kicks" / "kick.wav"
    p2 = tmp_path / "Bass" / "Sub" / "sub.wav"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_bytes(b"x")
    p2.write_bytes(b"x")

    tree = generate_directory_tree([p1, p2], tmp_path)

    assert "Drums/" in tree
    assert "Kicks/" in tree
    assert "kick.wav" in tree
    assert "Bass/" in tree
    assert "Sub/" in tree
    assert "sub.wav" in tree


def test_generate_directory_tree_empty_input() -> None:
    tree = generate_directory_tree([], Path("."))
    assert tree == ""
