from pathlib import Path
import pandas as pd
import csv

RAW_PATH = Path("data/raw/UCI_Credit_Card.csv")
OUT_PATH = Path("data/training/credit_default_training.csv")

TARGET_CANON = "default.payment.next.month"
ID_COL = "ID"

def normalize_cols(cols):
    clean = []
    for c in cols:
        c = str(c).strip()
        # quita comillas externas y comillas dobles
        c = c.replace('""', '"').strip('"').strip()
        clean.append(c)
    return clean

def load_csv(path: Path) -> pd.DataFrame:
    # detecta si el header vino todo en una sola celda
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header_row = next(csv.reader(f))

    if len(header_row) == 1 and "," in header_row[0]:
        cols = [x.strip().strip('"') for x in header_row[0].split(",")]
        df = pd.read_csv(path, skiprows=1, header=None, names=cols)
    else:
        df = pd.read_csv(path)

    # normaliza nombres de columnas (tu caso)
    df.columns = normalize_cols(df.columns)
    return df

def main():
    df = load_csv(RAW_PATH)

    df = df.drop_duplicates()

    # elimina ID
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # valida target (a veces viene con comillas)
    if TARGET_CANON not in df.columns:
        raise ValueError(f"No encuentro el target '{TARGET_CANON}'. Columnas actuales: {df.columns.tolist()[:10]}...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"OK -> Saved training dataset: {OUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df[TARGET_CANON].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()