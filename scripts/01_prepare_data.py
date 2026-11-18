from __future__ import annotations

import argparse
import json

from chat_moderation.config import load_config
from chat_moderation.data import prepare_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the OffensEval-TR dataset."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the source archive again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config()

    prepared_paths, statistics = prepare_dataset(
        config=config,
        force_download=args.force_download,
    )

    print("Dataset preparation completed.")
    print(f"Train: {prepared_paths.train}")
    print(f"Validation: {prepared_paths.validation}")
    print(f"Test: {prepared_paths.test}")
    print(f"Statistics: {prepared_paths.statistics}")
    print()
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()