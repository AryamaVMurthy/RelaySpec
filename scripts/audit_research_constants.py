from __future__ import annotations

import argparse
from pathlib import Path

from relayspec.research_protocol import audit_config, load_protocol, validate_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/relayspec_protocol.yaml")
    )
    parser.add_argument("configs", nargs="*", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(args.protocol)
    validate_protocol(protocol, repository_root=root)
    failures: list[str] = []
    for config in args.configs:
        failures.extend(
            f"{config}: {error}" for error in audit_config(config, protocol)
        )
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"protocol {protocol['protocol_version']} valid; "
        f"audited {len(args.configs)} explicit config(s)"
    )


if __name__ == "__main__":
    main()
