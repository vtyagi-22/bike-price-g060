import json
import numpy as np
import pandas as pd

NUMERIC = ["age_years", "km_driven", "engine_cc", "owners",
           "service_records", "accidents"]


def psi(expected, actual, bins=10, eps=1e-4):
    """PSI with bin edges taken from the reference (train) quantiles."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(expected, edges)[0] / len(expected)
    pa = np.histogram(actual, edges)[0] / len(actual)
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_report(train_clean: pd.DataFrame, later: pd.DataFrame) -> dict:
    return {c: round(psi(train_clean[c], later[c]), 4) for c in NUMERIC}


def most_drifted(report: dict) -> str:
    return max(report, key=report.get)


if __name__ == "__main__":
    import yaml
    from bikeprice.clean import clean_train   # use your team's actual module name

    params = yaml.safe_load(open("params.yaml"))
    train = clean_train(pd.read_csv(params["data"]["train"]))
    later = pd.read_csv(params["data"]["later"])
    report = psi_report(train, later)
    out = {"psi": report, "most_drifted_feature": most_drifted(report)}
    print(json.dumps(out, indent=2))
    json.dump(out, open("reports/psi.json", "w"), indent=2)