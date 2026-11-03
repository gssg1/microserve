import argparse
import json
from pathlib import Path

from microserve.utils.environment import server_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="server_report.json")
    args = parser.parse_args()
    report = server_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
