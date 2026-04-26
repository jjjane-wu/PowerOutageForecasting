import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "C:/Users/User/Desktop/ML/final/dataset/county_sheets_with_log_out.xlsx"   # ← change to your actual file path
OUTPUT_FILE = "cleaned_output.xlsx"
THRESHOLD   = 0.9                # drop if |Spearman correlation| > this value

# Columns that are identifiers / targets – never drop these
NON_FEATURE_COLS = {"location", "timestamp", "tracked", "out", "log_out",
                    "state_x", "state_y"}
# ──────────────────────────────────────────────────────────────────────────────


def get_columns_to_drop(df: pd.DataFrame, threshold: float) -> list[str]:
    """
    Return a list of columns to drop based on Spearman correlation.
    For each correlated pair (|r| > threshold) the *second* column is dropped
    (upper-triangle traversal, so the first-encountered column is kept).
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in NON_FEATURE_COLS
    ]

    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop


def process_file(input_path: str, output_path: str, threshold: float) -> None:
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(input_path, sheet_name=None)

    summary = {}
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            original_cols = df.shape[1]
            dropped = get_columns_to_drop(df, threshold)

            df_clean = df.drop(columns=dropped)
            df_clean.to_excel(writer, sheet_name=sheet_name, index=False)

            summary[sheet_name] = {
                "original_columns": original_cols,
                "dropped_columns":  len(dropped),
                "remaining_columns": df_clean.shape[1],
                "dropped_names": dropped,
            }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Spearman correlation filter  (threshold = {threshold})")
    print(f"{'='*60}")
    for sheet, info in summary.items():
        print(f"\nSheet : {sheet}")
        print(f"  Original columns  : {info['original_columns']}")
        print(f"  Dropped columns   : {info['dropped_columns']}")
        print(f"  Remaining columns : {info['remaining_columns']}")
        if info["dropped_names"]:
            print(f"  Dropped           : {', '.join(info['dropped_names'])}")
    print(f"\n✓ Saved cleaned file → {output_path}\n")


if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE, THRESHOLD)
