from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error


# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
DATA_PATH = ROOT_DIR / "data" / "raw" / "later.csv"

# Features used by v1
FEATURE_COLUMNS = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]

TARGET = "resale_thousand_rs"
GROUP_COLUMN = "city_zone"


def evaluate_group_mae():
    # Load v1 model and later dataset
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    # Prepare features and actual values
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    # Generate predictions using v1
    predictions = model.predict(X)

    # Add predictions to dataframe
    df["prediction"] = predictions

    # Calculate MAE separately for each group
    results = {}

    for group, group_data in df.groupby(GROUP_COLUMN):
        mae = mean_absolute_error(
            group_data[TARGET],
            group_data["prediction"],
        )
        results[group] = mae

    # Display results
    print("\nMAE by Group")
    print("-" * 30)

    for group, mae in results.items():
        print(f"{group}: {mae:.2f}")

    # Identify worst-served group
    worst_group = max(results, key=results.get)

    print("-" * 30)
    print(f"Worst-served group: {worst_group}")
    print(f"Highest MAE: {results[worst_group]:.2f}")


if __name__ == "__main__":
    evaluate_group_mae()