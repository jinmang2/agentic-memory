from __future__ import annotations

from .task import is_on_or_before_cutoff


def main() -> int:
    checks = {
        "TZ_CUTOFF_INCLUDES_UTC_DAY": is_on_or_before_cutoff(
            "2026-09-08T23:59:59+00:00", "2026-09-08"
        ),
        "TZ_CUTOFF_EXCLUDES_NEXT_DAY": not is_on_or_before_cutoff(
            "2026-09-09T00:00:00+00:00", "2026-09-08"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(failed[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
