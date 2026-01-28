import argparse
import faulthandler
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ui_main_window import MainWindow, calibrate_extents


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Run correctness self-test and exit.")
    parser.add_argument(
        "--selftest-fast",
        action="store_true",
        help="Run a smaller subset of the self-test harness.",
    )
    args, _ = parser.parse_known_args()
    if args.selftest or args.selftest_fast:
        from ui_main_window import run_selftest

        code = run_selftest(fast=args.selftest_fast)
        raise SystemExit(code)
    if os.getenv("CALIBRATE_EXTENTS") == "1":
        table = calibrate_extents()
        print("EXTENT_TABLE = {")
        for key, value in sorted(table.items()):
            print(f"    {key!r}: {value},")
        print("}")
        return
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
