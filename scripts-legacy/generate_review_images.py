from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ---------- PATH STRUCTURE ----------
HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting" / "predicted_order.csv"
OUT = WORK / "review_images"
OUT.mkdir(exist_ok=True)

# External SSD dataset root
DATA = Path("D:/Cohort1_5HT")
CHANNEL_ROOT = DATA / "02_channels"

# ---------- TEST MODE ----------
TEST_MODE = False     # False to run everything
TEST_N = 2           # slices per rat

# ---------- IMAGE SETTINGS ----------
MAX_SIZE = 2048
JPEG_QUALITY = 90

# map our output names -> folder/file target names on disk
TARGETS = {
    "dapi": "DAPI",
    "5ht": "5HT",
    "mCherry": "mCherry",
    "neun": "NeuN",
}

# ---------- helpers ----------
def apply_window_u16(img, vmin, vmax):
    """Linear window of a 16-bit-ish image to uint8."""
    img = img.astype(np.float32)
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (img * 255).astype(np.uint8)

def apply_gamma_u8(img_u8, gamma):
    """Gamma <1 brightens midtones."""
    x = img_u8.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    return (x * 255).astype(np.uint8)

def subtract_background_u8(img_u8, radius_px):
    """Approx rolling-ball via morphological opening."""
    k = int(max(3, radius_px // 2 * 2 + 1))  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(img_u8, bg)

def safe_percentile_norm(img, low=1, high=99):

    img = img.astype(np.float32)

    p_low, p_high = np.percentile(img, (low, high))

    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip((img - p_low) / (p_high - p_low), 0, 1)

    return (img * 255).astype(np.uint8)

def resize_max(img):
    h, w = img.shape[:2]
    scale = min(MAX_SIZE / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def percentile_normalize_u8(img_u16_or_u8):
    img = img_u16_or_u8.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 <= p1:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = np.clip((img - p1) / (p99 - p1), 0, 1)
    return (img * 255).astype(np.uint8)

def enhance_dapi(img):
    img = percentile_normalize_u8(img)
    return cv2.GaussianBlur(img, (3, 3), 0)

def enhance_5ht(img):
    # img is float32 from the loader; treat as 16-bit dynamic range
    img_u8 = apply_window_u16(img, vmin=3000, vmax=36000)
    img_u8 = apply_gamma_u8(img_u8, gamma=0.75)
    return img_u8

def enhance_mcherry(img):
    img_u8 = apply_window_u16(img, vmin=0, vmax=60000)
    img_u8 = subtract_background_u8(img_u8, radius_px=40)
    # optional small gamma lift (comment out if you don't want it)
    img_u8 = apply_gamma_u8(img_u8, gamma=0.85)
    return img_u8

def enhance_neun(img):
    img = safe_percentile_norm(img, 2, 98)
    img = (img * 0.8).astype(np.uint8)

    return img

def process_channel(img, key):
    if key == "5ht":
        return enhance_5ht(img)
    if key == "mCherry":
        return enhance_mcherry(img)
    if key == "dapi":
        return enhance_dapi(img)
    if key == "neun":
        return enhance_neun(img)
    return percentile_normalize_u8(img)  # fallback

def parse_section_number(image_name: str) -> int:
    # expects something like "761CB_sec10" -> 10
    # robust to leading zeros
    if "_sec" not in image_name:
        raise ValueError(f"Can't parse section from image='{image_name}'")
    sec_part = image_name.split("_sec", 1)[1]
    digits = "".join([c for c in sec_part if c.isdigit()])
    return int(digits)

def raw_tif_path(rat: str, section: int, target: str) -> Path:
    # D:/Cohort1_5HT/02_channels/{rat}/{target}/{rat}_sec##_{target}.tif
    return CHANNEL_ROOT / rat / target / f"{rat}_sec{section:02d}_{target}.tif"



# ---------- main ----------
def main():
    order = pd.read_csv(ORDER)

    # expected cols: rat, image, predicted_order
    if "rat" not in order.columns:
        raise ValueError(f"predicted_order.csv missing 'rat' column. Columns: {list(order.columns)}")
    if "image" not in order.columns:
        raise ValueError(f"predicted_order.csv missing 'image' column. Columns: {list(order.columns)}")
    if "predicted_order" not in order.columns:
        raise ValueError(f"predicted_order.csv missing 'predicted_order' column. Columns: {list(order.columns)}")

    for rat in order["rat"].unique():
        print(f"\nProcessing {rat}", flush=True)

        out_dir = OUT / rat
        out_dir.mkdir(parents=True, exist_ok=True)

        subset = order[order["rat"] == rat].sort_values("predicted_order")

        if TEST_MODE:
            print(f"TEST MODE: first {TEST_N} slices", flush=True)
            subset = subset.head(TEST_N)

        for _, row in subset.iterrows():
            image_name = row["image"]          # e.g. 761CB_sec10
            ord_idx = int(row["predicted_order"])
            sec = parse_section_number(image_name)

            for key, target in TARGETS.items():
                src = raw_tif_path(rat, sec, target)

                if not src.exists():
                    print("missing:", src)
                    continue

                img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)

                if img is None:
                    print("unreadable:", src)
                    continue

                # if RGB image, choose channel with strongest signal
                if img.ndim == 3:

                    means = [img[:,:,i].mean() for i in range(img.shape[2])]
                    best_channel = int(np.argmax(means))

                    img = img[:,:,best_channel]

                img = img.astype(np.float32)

                # ---- process channel ----
                proc = process_channel(img, key)

                # resize for review dataset
                proc = resize_max(proc)

                out_path = out_dir / f"{ord_idx:03d}_{key}.png"
                cv2.imwrite(str(out_path), proc, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        print(f"done {rat}", flush=True)

    print("\nReview images complete.", flush=True)

if __name__ == "__main__":
    main()