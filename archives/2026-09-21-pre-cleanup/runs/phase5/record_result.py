"""Append a completed measurement to the Phase5 table, preserving full precision."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--down-format", required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--qat-data", default="none")
    parser.add_argument("--parent", default="")
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    result = json.loads(evidence.read_text())
    root = Path(__file__).parent
    table = root / "summary.csv"
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if row["evidence_path"] != str(evidence)]
    row = dict(model=args.model, seed=args.seed, stage=args.stage, down_format=args.down_format,
        qat_data=args.qat_data, parent_package=args.parent, evaluation_package=args.package,
        dataset=result.get("dataset", "Salesforce/wikitext"), split=result.get("split", "validation"),
        subset=result.get("subset", "wikitext-2-raw-v1"),
        subset_identity=result.get("input_token_path", "full cached WikiText-2 validation"),
        metric_protocol=result["evaluation_precision"], ppl=result["ppl"], nll=result["nll"],
        targets=result["predicted_tokens"], input_tokens=result["token_count"],
        unscored_tail_tokens=result["unscored_tail_tokens"], reuse_status="new_phase5_measurement",
        completion="COMPLETED", evidence_path=str(evidence))
    rows.append(row)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
