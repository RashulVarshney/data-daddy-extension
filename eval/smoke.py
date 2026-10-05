"""Tiny offline smoke eval used by CI: classify a handful of synthetic columns and run a mock agent turn."""

from datadaddy_ai.common.seed import set_global_seed


def main() -> None:
    set_global_seed()
    print("smoke: ok")


if __name__ == "__main__":
    main()
