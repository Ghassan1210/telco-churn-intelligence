"""Load the local Telco churn workbook into a validated SQLite staging table."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path


EXPECTED_ROWS = 7_043
REQUIRED_PACKAGES = ("pandas", "openpyxl")


def check_dependencies() -> None:
    """Fail with the exact installation command when a required package is absent."""
    missing = [package for package in REQUIRED_PACKAGES if importlib.util.find_spec(package) is None]
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            f"Missing dependency: {', '.join(missing)}. Install it with: pip install {packages}"
        )


def discover_workbook(project_root: Path) -> Path:
    """Find the only supported workbook location, rejecting ambiguous inputs."""
    candidates = [
        path
        for directory in (project_root, project_root / "data")
        for path in sorted(directory.glob("*.xlsx"))
        if not path.name.startswith("~$")
    ]
    candidates = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if not candidates:
        raise FileNotFoundError("No .xlsx file found in the project root or data/ subfolder.")
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Multiple .xlsx files found; keep one source workbook: {paths}")
    return candidates[0]


def clean_data(workbook_path: Path):
    """Read the preferred sheet and enforce the source data contract."""
    import pandas as pd

    workbook = pd.ExcelFile(workbook_path, engine="openpyxl")
    sheet_name = "Data_Master" if "Data_Master" in workbook.sheet_names else workbook.sheet_names[0]
    data = pd.read_excel(workbook, sheet_name=sheet_name, engine="openpyxl")
    data.columns = data.columns.astype(str).str.strip()

    required_columns = {"TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen", "Churn"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Required columns are missing: {', '.join(missing_columns)}")

    for column in ("TotalCharges", "MonthlyCharges"):
        charges = data[column].astype("string").str.strip().replace("", "0")
        data[column] = pd.to_numeric(charges, errors="coerce").fillna(0.0)
        data[column] = data[column].astype(float)

    for column in ("tenure", "SeniorCitizen"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{column} contains values that cannot be represented as integers.")
        data[column] = numeric.astype(int)

    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} data rows, found {len(data):,}.")
    if data[["TotalCharges", "MonthlyCharges"]].isna().any().any():
        raise ValueError("Charge columns still contain null values after cleaning.")

    return data, sheet_name


def write_database(data, database_path: Path) -> None:
    """Replace the staging table with the cleaned workbook contents."""
    with sqlite3.connect(database_path) as connection:
        data.to_sql("stg_telco", connection, if_exists="replace", index=False)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    try:
        check_dependencies()
        workbook_path = discover_workbook(project_root)
        data, sheet_name = clean_data(workbook_path)
        database_path = project_root / "telco.db"
        write_database(data, database_path)

        churn_counts = data["Churn"].value_counts().to_dict()
        print(f"Source Excel file: {workbook_path}")
        print(f"Sheet used: {sheet_name}")
        print(f"Target database: {database_path}")
        print(f"Total rows loaded: {len(data):,}")
        print(f"Total columns loaded: {len(data.columns):,}")
        print(f"Baseline churned records (Yes): {int(churn_counts.get('Yes', 0)):,}")
        print(f"Baseline active records (No): {int(churn_counts.get('No', 0)):,}")
        print("Zero null values in charges: confirmed")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, OSError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())