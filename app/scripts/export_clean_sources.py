import json
import os
import re
import unicodedata
from pathlib import Path

from app.config.settings import PROCESSED_DIR


def _natural_key(name: str) -> tuple[int, str]:
    m = re.search(r"(\d+)", name)
    if not m:
        return (10**9, name)
    return (int(m.group(1)), name)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _iter_chapter_files(chapters_dir: Path) -> list[Path]:
    files = [p for p in chapters_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    files.sort(key=lambda p: _natural_key(p.name))
    return files


def export_clean_sources(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for novel_dir in sorted(PROCESSED_DIR.iterdir()):
        if not novel_dir.is_dir():
            continue

        chapters_dir = novel_dir / "chapters"
        if not chapters_dir.exists():
            continue

        chapter_files = _iter_chapter_files(chapters_dir)
        if not chapter_files:
            continue

        blocks: list[str] = []
        title = ""
        author = ""

        for fpath in chapter_files:
            with open(fpath, "r", encoding="utf-8") as f:
                row = json.load(f)
            title = row.get("title", title)
            author = row.get("author", author)
            chapter = row.get("chapter_label", "")
            content = _normalize_text(row.get("page_content", ""))
            blocks.append(f"{chapter}\n\n{content}")

        full_text = "\n\n".join(blocks).strip() + "\n"

        out_file = output_dir / f"{novel_dir.name}.clean.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        manifest.append(
            {
                "novel": novel_dir.name,
                "title": title,
                "author": author,
                "chapters": len(chapter_files),
                "output_file": str(out_file).replace("\\", "/"),
                "bytes": out_file.stat().st_size,
            }
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "assets" / "clean_sources"
    manifest = export_clean_sources(out_dir)
    print(f"exported {len(manifest)} clean source files to: {out_dir}")
    for row in manifest:
        print(f"- {row['novel']}: {row['chapters']} chapters -> {row['output_file']}")


if __name__ == "__main__":
    main()
