from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Used when center-bias template is not provided by user assets.
DEFAULT_CENTERBIAS_SHAPE = (1024, 1024)
