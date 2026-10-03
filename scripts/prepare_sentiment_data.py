"""Descarga el ZIP de Sentiment Labelled Sentences (UCI), extrae los 3
archivos de texto y los combina en data/raw/dataset.csv. Se ejecuta una
sola vez; data/raw/ no se edita manualmente después de esto."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
import pandas as pd
import requests

from inf8239_u02.config import ROOT

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
FILES = {
    "imdb_labelled.txt": "imdb",
    "amazon_cells_labelled.txt": "amazon",
    "yelp_labelled.txt": "yelp",
}

def main() -> int:
    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    frames = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for name in zf.namelist():
            base = name.split("/")[-1]
            if base in FILES:
                with zf.open(name) as f:
                    df = pd.read_csv(f, sep="\t", header=None, names=["text", "label"], quoting=csv.QUOTE_NONE)
                    df["source"] = FILES[base]
                    frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output = ROOT / "data" / "raw" / "dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"Guardado: {output} ({len(combined)} filas)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())