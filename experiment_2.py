import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

from features_2 import FEATURES_2_PATH, build_rolling_features_2
from modeling_2 import run_walkforward_validation_2
from preprocessing_2 import clean_and_save_data_2
from reporting_2 import run_reporting_2


def run_experiment_2():
    print("\n[experiment_2] 1/5 Gelismis preprocessing")
    clean_and_save_data_2(force=False)

    print("\n[experiment_2] 2/5 Rolling feature uretimi")
    build_rolling_features_2()

    print("\n[experiment_2] 3/5 Walk-forward validation")
    run_walkforward_validation_2(features_path=FEATURES_2_PATH, rebuild_features=False)

    print("\n[experiment_2] 4/5 Baseline ve icgoru raporlari")
    run_reporting_2(features_path=FEATURES_2_PATH)

    print("\n[experiment_2] Tamamlandi.")


if __name__ == "__main__":
    run_experiment_2()
