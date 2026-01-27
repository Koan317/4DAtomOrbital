import argparse
import faulthandler
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ui_main_window import MainWindow, calibrate_extents, run_selftest


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Run correctness self-test and exit.")
    args, _ = parser.parse_known_args()
    if args.selftest:
        code = run_selftest()
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
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
