import argparse
import json
from pathlib import Path

from benchmark_project import summarize_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    merged = {}
    for name in args.inputs:
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key == "kv_cache_storage_bytes":
                merged.setdefault(key, {}).update(value)
            elif key == "environment":
                merged.setdefault(key, value)
            else:
                merged[key] = value
    merged["summary"] = summarize_suite(merged)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged["summary"], indent=2))
    print("saved", output)


if __name__ == "__main__":
    main()
