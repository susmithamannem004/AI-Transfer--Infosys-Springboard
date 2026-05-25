from pathlib import Path

from src.data_sources import generate_raw_datasets
from src.features import build_feature_table
from src.models import train_and_evaluate
from src.reporting import write_dashboard, write_methodology_report


ROOT = Path(__file__).resolve().parent


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    model_dir = ROOT / "models"
    report_dir = ROOT / "reports"

    for directory in (raw_dir, processed_dir, model_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = generate_raw_datasets(raw_dir)
    features = build_feature_table(raw, processed_dir)
    results = train_and_evaluate(features, model_dir, report_dir)
    write_dashboard(features, results, report_dir)
    write_methodology_report(results, report_dir)

    print("TransferIQ pipeline complete.")
    print(f"Metrics: {report_dir / 'metrics.csv'}")
    print(f"Predictions: {report_dir / 'predictions.csv'}")
    print(f"Dashboard: {report_dir / 'dashboard.html'}")
    print(f"Report: {report_dir / 'methodology_report.md'}")


if __name__ == "__main__":
    main()

