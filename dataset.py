import argparse
import csv
import hashlib
import io
import math
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/850/data.csv"
DEFAULT_OUTPUT = Path("data/clean-v1.csv")
FEATURE_COLUMNS = {
    "Area": "area_px",
    "Perimeter": "perimeter_px",
    "MajorAxisLength": "major_axis_px",
    "MinorAxisLength": "minor_axis_px",
}


def fetch_csv(url: str) -> bytes:
    context = ssl.create_default_context()
    macos_bundle = Path("/etc/ssl/cert.pem")
    if sys.platform == "darwin" and not context.get_ca_certs() and macos_bundle.is_file():
        context.load_verify_locations(cafile=macos_bundle)
    with urlopen(url, timeout=30, context=context) as response:
        return response.read()


def prepare_csv(content: bytes) -> tuple[bytes, int]:
    source = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    missing = (set(FEATURE_COLUMNS) | {"Class"}) - set(source.fieldnames or [])
    if missing:
        raise ValueError(f"Missing source columns: {', '.join(sorted(missing))}")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([*FEATURE_COLUMNS.values(), "variety"])
    variety_counts = {"Kecimen": 0, "Besni": 0}
    for row in source:
        variety = (row["Class"] or "").strip()
        if variety not in variety_counts:
            raise ValueError(f"Line {source.line_num}: unexpected variety {variety!r}")

        measurements = [(row[column] or "").strip() for column in FEATURE_COLUMNS]
        try:
            values = [float(measurement) for measurement in measurements]
        except ValueError as error:
            raise ValueError(f"Line {source.line_num}: invalid measurement ({error})") from None
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"Line {source.line_num}: measurements must be finite and positive")

        writer.writerow([*measurements, variety])
        variety_counts[variety] += 1

    if not all(variety_counts.values()):
        raise ValueError("The source must contain both Kecimen and Besni raisins")
    return output.getvalue().encode("utf-8"), sum(variety_counts.values())


def download_clean_csv(url: str, expected_sha256: str) -> bytes:
    content = fetch_csv(url)
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or download the Raisin dataset.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Clean the official UCI CSV")
    download = commands.add_parser("download", help="Fetch and verify a published clean CSV")
    download.add_argument("--url", required=True, help="Public dataset URL")
    download.add_argument("--sha256", required=True, help="Expected SHA-256 digest")
    for command in (prepare, download):
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            content, rows = prepare_csv(fetch_csv(UCI_CSV_URL))
            summary = f"Prepared {rows} rows"
        else:
            content = download_clean_csv(args.url, args.sha256)
            summary = "Downloaded verified CSV"

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            temporary_file = Path(directory) / args.output.name
            temporary_file.write_bytes(content)
            temporary_file.replace(args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"{summary} in {args.output}")
    print(f"SHA-256 {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
