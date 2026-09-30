from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_FILE_NAME = "telco_churn_cleaned.xlsx"
TARGET_COLUMN = "Churn"


def find_dataset_path() -> Path:
    """Find the workbook from the current directory or this script's project root."""
    search_directories = tuple(
        dict.fromkeys(
            (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent)
        )
    )
    candidate_names = (DATA_FILE_NAME, f"{DATA_FILE_NAME}.xlsx")

    for candidate_name in candidate_names:
        for directory in search_directories:
            candidate = directory / candidate_name
            if candidate.is_file():
                return candidate.resolve()

    searched = ", ".join(str(directory) for directory in search_directories)
    raise FileNotFoundError(
        f"Could not find {DATA_FILE_NAME} in: {searched}. "
        "The doubled .xlsx.xlsx extension is also accepted for the current project file."
    )


def main() -> None:
    dataset_path = find_dataset_path()
    try:
        df = pd.read_excel(dataset_path)
    except ImportError as exc:
        raise ImportError(
            "Excel support is unavailable. Install openpyxl in this Python environment."
        ) from exc

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET_COLUMN!r}.")

    labels = df[TARGET_COLUMN]
    unexpected_labels = set(labels.dropna().astype(str).unique()) - {"Yes", "No"}
    if unexpected_labels:
        raise ValueError(f"Unexpected Churn labels: {sorted(unexpected_labels)}")

    target = labels.map({"No": 0, "Yes": 1})
    labeled_rows = target.notna()
    features = df.loc[labeled_rows].drop(
        columns=[TARGET_COLUMN, "customerID"], errors="ignore"
    )
    target = target.loc[labeled_rows].astype("int8")

    if target.nunique() != 2:
        raise ValueError("Training requires both 'Yes' and 'No' Churn labels.")

    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"], errors="coerce"
        )

    numeric_columns = features.select_dtypes(include="number").columns
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    if len(categorical_columns):
        features[categorical_columns] = features[categorical_columns].fillna("Unknown")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    numeric_columns = X_train_raw.select_dtypes(include="number").columns
    train_medians = X_train_raw[numeric_columns].median()
    X_train_raw[numeric_columns] = X_train_raw[numeric_columns].fillna(
        train_medians
    ).fillna(0)
    X_test_raw[numeric_columns] = X_test_raw[numeric_columns].fillna(
        train_medians
    ).fillna(0)

    X_train = pd.get_dummies(X_train_raw, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test_raw, drop_first=True, dtype=int).reindex(
        columns=X_train.columns, fill_value=0
    )
    if X_train.empty or X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Prepared training or test features are empty or contain missing values.")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    churn_class_index = list(model.classes_).index(1)
    churn_probabilities = model.predict_proba(X_test)[:, churn_class_index]

    print(f"Dataset: {dataset_path}")
    print(f"Rows used: {len(features):,} | Encoded model features: {X_train.shape[1]:,}")
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            digits=3,
            zero_division=0,
        )
    )
    print(f"ROC-AUC: {roc_auc_score(y_test, churn_probabilities):.3f}")

    top_importances = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(5)
    )
    print("\nTop 5 churn feature importances:")
    print(top_importances.to_string(float_format=lambda value: f"{value:.4f}"))

    output_directory = Path(__file__).resolve().parent
    model_path = output_directory / "churn_model.pkl"
    feature_path = output_directory / "model_features.pkl"
    joblib.dump(model, model_path)
    joblib.dump(list(X_train.columns), feature_path)
    print(f"\nSaved model: {model_path}")
    print(f"Saved feature columns: {feature_path}")


if __name__ == "__main__":
    main()
