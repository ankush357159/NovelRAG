import os
import requests

from app.config.novels import NOVELS
from app.config.settings import ASSETS_DIR


def download_file(url: str, filepath: str):
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)


def download_novels():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    for filename, meta in NOVELS.items():
        url = meta["url"]
        filepath = os.path.join(ASSETS_DIR, filename)

        if os.path.exists(filepath):
            print(f"Skipping (already exists): {filename}")
            continue

        print(f"Downloading: {meta['title']} by {meta['author']}")
        download_file(url, filepath)

    print("All novels downloaded successfully.")


if __name__ == "__main__":
    download_novels()