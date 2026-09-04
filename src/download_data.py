"""Download Home Credit Default Risk data from Kaggle into data/raw/.

Requires a valid ~/.kaggle/kaggle.json API token and that the user has
joined the competition (accepted its rules) on kaggle.com. This script
does NOT fall back to synthetic data if the download fails -- it stops
and reports the error so the user can fix credentials/access.
"""
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
COMPETITION = "home-credit-default-risk"


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print(f"[ERROR] kaggle 패키지를 불러올 수 없습니다: {e}")
        print("pip install kaggle 로 설치했는지 확인하세요.")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[ERROR] Kaggle 인증 실패: {e}")
        print("~/.kaggle/kaggle.json 이 올바른 위치에 있는지 확인하세요.")
        return 1

    try:
        print(f"Downloading competition files for '{COMPETITION}' into {RAW_DIR} ...")
        api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)
    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        print(
            "대회 페이지(https://www.kaggle.com/competitions/home-credit-default-risk)에서 "
            "'Join Competition'으로 규칙에 동의했는지 확인하세요."
        )
        return 1

    # Kaggle competition downloads arrive as a single zip
    import zipfile

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    expected = RAW_DIR / "application_train.csv"
    if not expected.exists():
        print(f"[ERROR] 예상 파일이 없습니다: {expected}")
        return 1

    print(f"[OK] 데이터 다운로드 완료: {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
