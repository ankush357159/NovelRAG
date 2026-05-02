import os
import re
import json
import hashlib
import shutil
from pathlib import Path

from app.config.settings import ASSETS_DIR, PROCESSED_DIR
from app.config.novels import NOVELS


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_file_hash(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def strip_gutenberg_markers(text: str) -> str:
    """
    Remove the Project Gutenberg header (everything before the
    '*** START OF ***' line) and footer (everything after '*** END OF ***').
    """
    start_match = re.search(
        r"\*\*\* START OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL
    )
    if start_match:
        text = text[start_match.end():]

    end_match = re.search(
        r"\*\*\* END OF.*?\*\*\*", text, re.IGNORECASE | re.DOTALL
    )
    if end_match:
        text = text[: end_match.start()]

    return text


def normalize_illustrations(text: str) -> str:
    """
    Handle Gutenberg illustration and annotation blocks:

    1. Convert illustration blocks that embed a chapter heading
       (e.g. ``[Illustration: ... Chapter I.]``) into a proper standalone
       ``CHAPTER I.`` heading so the chapter splitter can detect it.
    2. Remove all remaining ``[Illustration ...]`` blocks (including those
       with nested ``[Copyright ...]`` inner brackets that end with ``]]``).
    3. Remove ``[Transcriber's Note ...]`` and similar annotation blocks.
    """
    # Step 1 – Illustration blocks that contain a chapter reference.
    # These match [Illustration: <text with no '['>  Chapter X.]
    text = re.sub(
        r"\[Illustration:[^\[]*?Chapter[ \t]+([IVXLCDMivxlcdm]+|\d+)\.?[ \t]*\]",
        lambda m: f"\nCHAPTER {m.group(1).upper()}.\n",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Step 2 – Illustration blocks that contain one level of nested brackets
    # (e.g. ``[Illustration: "text" [_Copyright 1894..._]]``)
    text = re.sub(
        r"\[Illustration:(?:[^\[\]]|\[[^\[\]]*\])*\]",
        "",
        text,
        flags=re.DOTALL,
    )

    # Step 3 – Bare ``[Illustration]`` tags and any remaining simple blocks
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.DOTALL)

    # Step 4 – Transcriber / editor annotation blocks
    text = re.sub(
        r"\[(?:Transcriber|Editor|Note)[^\]]*\]", "", text, flags=re.DOTALL | re.IGNORECASE
    )

    return text


def strip_preamble(text: str) -> str:
    """
    Drop front matter (title page, publisher info, preface, table of
    contents) that precedes the actual novel text.

    Strategy
    --------
    * If the text contains a standalone ``INTRODUCTION`` heading (on its own
      line), start there — this preserves the author's introduction that
      appears in many Gutenberg editions (e.g. Frankenstein).
    * Otherwise start at the first standalone ``CHAPTER X.`` heading.

    Using ``^...$`` line anchors (MULTILINE) prevents false positives from
    the word "introduction" appearing inside ordinary prose sentences.
    """
    intro_match = re.search(
        r"(?m)^[ \t]*INTRODUCTION\.?[ \t]*$", text, re.IGNORECASE
    )
    if intro_match:
        return text[intro_match.start():]

    chapter_match = re.search(
        r"(?m)^[ \t]*CHAPTER[ \t]+(?:[IVXLCDM]+|\d+)\.?[ \t]*$",
        text,
        re.IGNORECASE,
    )
    if chapter_match:
        return text[chapter_match.start():]

    return text


def normalize_text(text: str) -> str:
    """General whitespace normalization."""
    text = re.sub(r"\r\n", "\n", text)
    # Collapse three or more consecutive blank lines to one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove spurious leading space that Gutenberg sometimes adds after newlines
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chapter splitting
# ---------------------------------------------------------------------------

def split_and_save_chapters(text: str, chapters_dir: str, meta: dict) -> list[str]:
    """
    Split *text* into chapters and write each as a JSON file under
    *chapters_dir*.  Returns the list of written file paths.

    The split regex anchors to line boundaries (MULTILINE) so that the word
    "chapter" inside ordinary sentences is never treated as a heading.
    """
    os.makedirs(chapters_dir, exist_ok=True)

    # Split at lines that consist solely of a CHAPTER heading.
    # (?im) = case-insensitive + multiline so ^ / $ match line boundaries.
    chapter_regex = (
        r"(?im)(?=^[ \t]*chapter[ \t]+(?:[ivxlcdm]+|\d+)\.?[ \t]*$)"
    )
    segments = re.split(chapter_regex, text)

    chapters = [
        s.strip()
        for s in segments
        if s.strip() and re.match(r"(?i)^chapter", s.strip())
    ]

    chapter_files: list[str] = []

    for i, content in enumerate(chapters, start=1):
        chapter_path = os.path.join(chapters_dir, f"chapter_{i}.json")

        first_line = content.split("\n")[0].strip()

        chapter_data = {
            "title": meta["title"],
            "author": meta["author"],
            "chapter_index": i,
            "chapter_label": first_line,
            "page_content": content,
        }

        with open(chapter_path, "w", encoding="utf-8") as f:
            json.dump(chapter_data, f, indent=2, ensure_ascii=False)

        chapter_files.append(chapter_path)

    return chapter_files


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_all_novels() -> list[dict]:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    metadata_list: list[dict] = []

    for filename, meta in NOVELS.items():
        input_path = os.path.join(ASSETS_DIR, filename)

        if not os.path.exists(input_path):
            print(f"Source not found: {input_path}")
            continue

        file_stem = Path(filename).stem
        novel_output_dir = os.path.join(PROCESSED_DIR, file_stem)
        chapters_dir = os.path.join(novel_output_dir, "chapters")
        hash_file = os.path.join(novel_output_dir, ".source_hash")

        # ---- Change detection ------------------------------------------------
        current_hash = compute_file_hash(input_path)
        stored_hash: str | None = None
        if os.path.exists(hash_file):
            with open(hash_file) as fh:
                stored_hash = fh.read().strip()

        if stored_hash == current_hash and os.path.exists(chapters_dir):
            print(f"Skipping (source unchanged): {filename}")
            existing = sorted(
                os.path.join(chapters_dir, f)
                for f in os.listdir(chapters_dir)
                if f.endswith(".json")
            )
            metadata_list.append(
                {
                    "filename": filename,
                    "title": meta["title"],
                    "author": meta["author"],
                    "processed_directory": str(novel_output_dir),
                    "chapter_files": existing,
                }
            )
            continue

        # ---- (Re-)process ----------------------------------------------------
        print(f"Processing: {filename}")

        # Wipe stale chapter files before re-generating
        if os.path.exists(chapters_dir):
            shutil.rmtree(chapters_dir)

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text: str = f.read()

        text = strip_gutenberg_markers(raw_text)
        text = normalize_illustrations(text)
        text = normalize_text(text)
        text = strip_preamble(text)

        created_chapters = split_and_save_chapters(text, chapters_dir, meta)

        # Persist hash so subsequent unchanged runs are skipped
        os.makedirs(novel_output_dir, exist_ok=True)
        with open(hash_file, "w") as fh:
            fh.write(current_hash)

        metadata_list.append(
            {
                "filename": filename,
                "title": meta["title"],
                "author": meta["author"],
                "processed_directory": str(novel_output_dir),
                "chapter_files": created_chapters,
            }
        )

    metadata_file = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\nSuccess! Processed {len(metadata_list)} novels.")
    return metadata_list


if __name__ == "__main__":
    process_all_novels()