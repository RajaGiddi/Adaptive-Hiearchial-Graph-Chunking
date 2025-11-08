import os
from pathlib import Path

def normalize_txt_filenames(directory: str):
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {directory}")
        return

    for file in path.glob("*.txt"):
        new_name = file.name.lower().replace(" ", "_")
        if new_name != file.name:
            new_path = file.with_name(new_name)
            print(f"Renaming: {file.name} -> {new_name}")
            file.rename(new_path)

if __name__ == "__main__":
    folder = "data/arxiv_texts"  # change to your folder
    normalize_txt_filenames(folder)
