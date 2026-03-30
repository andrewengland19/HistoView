from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# -----------------------------
# PATHS (standard)
# -----------------------------
HOME = Path.home()
WORK = HOME / "SectionSorter"

ORDER = WORK / "sorting" / "predicted_order.csv"
OUT = WORK / "diagnostics" / "qc_grids"
OUT.mkdir(parents=True, exist_ok=True)

# dataset root on SSD
DATA = Path("D:/Cohort1_5HT")
CHANNEL_ROOT = DATA / "02_channels"

# rotation/mirror manifest (must include rat + section_label or image, rotation_deg, mirror_lr)
MANIFEST = DATA / "05_metadata" / "overlay_manifest.csv"  # change if yours is elsewhere

# -----------------------------
# SETTINGS
# -----------------------------
TEST_MODE = False
TEST_N = 6                 # columns per rat when testing

THUMB_H = 120               # thumbnail height (width scales)
MAX_COLS_PER_BLOCK = 20     # wrap after this many columns
GAP = 6                     # px gap between thumbs
ROW_GAP = 10                # px gap between stain rows
BLOCK_GAP = 18              # px gap between wrapped blocks
LABEL_W = 80                # left label width

# Fiji-calibrated visualization settings
FIVEHT_MIN, FIVEHT_MAX, FIVEHT_GAMMA = 3000, 36000, 0.75
MCH_MIN, MCH_MAX, MCH_BG_RADIUS = 0, 60000, 40

TARGETS = {
    "dapi": "DAPI",
    "5ht": "5HT",
    "mCherry": "mCherry",
    "neun": "NeuN",
}

ROWS = ["dapi", "5ht", "mCherry", "neun"]  # reorder or drop rows if you want

# -----------------------------
# helpers
# -----------------------------
def parse_section_number(image_name: str) -> int:
    # expects "761CB_sec10" (or sec010 etc)
    if "_sec" not in image_name:
        raise ValueError(f"Can't parse section from image='{image_name}'")
    sec_part = image_name.split("_sec", 1)[1]
    digits = "".join([c for c in sec_part if c.isdigit()])
    return int(digits)

def raw_tif_path(rat: str, section: int, target: str) -> Path:
    # D:/Cohort1_5HT/02_channels/{rat}/{target}/{rat}_sec##_{target}.tif
    return CHANNEL_ROOT / rat / target / f"{rat}_sec{section:02d}_{target}.tif"

def pick_signal_plane(img):
    # if RGB, pick plane with most mean signal
    if img.ndim == 3:
        means = [img[:, :, i].mean() for i in range(img.shape[2])]
        return img[:, :, int(np.argmax(means))]
    return img

def apply_window_u16(img, vmin, vmax):
    img = img.astype(np.float32)
    if vmax <= vmin:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return (img * 255).astype(np.uint8)

def apply_gamma_u8(img_u8, gamma):
    x = img_u8.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    return (x * 255).astype(np.uint8)

def subtract_background_u8(img_u8, radius_px):
    k = int(max(3, radius_px // 2 * 2 + 1))  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(img_u8, bg)

def enhance(img, key):
    # img is float32-ish grayscale
    if key == "dapi":
        # gentle auto window
        p1, p99 = np.percentile(img, (1, 99))
        return apply_window_u16(img, p1, p99)

    if key == "neun":
        # keep from blowing out
        p2, p98 = np.percentile(img, (2, 98))
        out = apply_window_u16(img, p2, p98)
        out = (out.astype(np.float32) * 0.8).astype(np.uint8)
        return out

    if key == "5ht":
        out = apply_window_u16(img, FIVEHT_MIN, FIVEHT_MAX)
        out = apply_gamma_u8(out, FIVEHT_GAMMA)
        return out

    if key == "mCherry":
        out = apply_window_u16(img, MCH_MIN, MCH_MAX)
        out = subtract_background_u8(out, MCH_BG_RADIUS)
        return out

    return apply_window_u16(img, np.percentile(img, 1), np.percentile(img, 99))

def apply_transform(img_u8, rot_deg=0, mirror_lr=0):
    # rot_deg in {0,90,180,270}, mirror_lr {0,1}
    rot_deg = int(rot_deg) % 360
    if rot_deg == 90:
        img_u8 = cv2.rotate(img_u8, cv2.ROTATE_90_CLOCKWISE)
    elif rot_deg == 180:
        img_u8 = cv2.rotate(img_u8, cv2.ROTATE_180)
    elif rot_deg == 270:
        img_u8 = cv2.rotate(img_u8, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if int(mirror_lr) == 1:
        img_u8 = cv2.flip(img_u8, 1)
    return img_u8

def to_thumb(img_u8):
    h, w = img_u8.shape[:2]
    scale = THUMB_H / max(h, 1)
    tw = max(1, int(w * scale))
    thumb = cv2.resize(img_u8, (tw, THUMB_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

def label_cell(text, height):
    canvas = np.zeros((height, LABEL_W, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (6, int(height*0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return canvas

def hstack_with_gaps(tiles):
    if not tiles:
        return np.zeros((THUMB_H, 1, 3), dtype=np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        gap = np.zeros((out.shape[0], GAP, 3), dtype=np.uint8)
        out = np.hstack([out, gap, t])
    return out

# -----------------------------
# load manifest transforms
# -----------------------------
def load_transform_map():
    if not MANIFEST.exists():
        print(f"[warn] no manifest found at {MANIFEST} — no rotation/mirror applied")
        return {}

    df = pd.read_csv(MANIFEST)

    # accept either section_label OR image column
    key_col = None
    for c in ["section_label", "image"]:
        if c in df.columns:
            key_col = c
            break

    if key_col is None or "rat" not in df.columns:
        print("[warn] manifest missing 'rat' + ('section_label' or 'image') — no rotation/mirror applied")
        return {}

    rot_col = "rotation_deg" if "rotation_deg" in df.columns else None
    mir_col = "mirror_lr" if "mirror_lr" in df.columns else None

    if rot_col is None and mir_col is None:
        print("[warn] manifest missing rotation_deg/mirror_lr — no rotation/mirror applied")
        return {}

    # map (rat, section_label/image) -> (rot, mirror)
    m = {}
    for _, r in df.iterrows():
        rat = str(r["rat"])
        sec_key = str(r[key_col])
        rot = int(r[rot_col]) if rot_col else 0
        mir = int(r[mir_col]) if mir_col else 0
        m[(rat, sec_key)] = (rot, mir)
    return m, key_col

# -----------------------------
# main
# -----------------------------
def main():
    order = pd.read_csv(ORDER)

    # expected: rat, image, predicted_order
    for col in ["rat", "image", "predicted_order"]:
        if col not in order.columns:
            raise ValueError(f"{ORDER} missing column '{col}'. Columns: {list(order.columns)}")

    transform_map, manifest_key_col = load_transform_map()

    for rat in order["rat"].unique():
        rat_df = order[order["rat"] == rat].sort_values("predicted_order")
        if TEST_MODE:
            rat_df = rat_df.head(TEST_N)

        # wrap into blocks so output isn't absurdly wide
        blocks = []
        rows_rendered = []

        images = rat_df["image"].tolist()  # e.g., 761CB_sec10
        chunks = [images[i:i+MAX_COLS_PER_BLOCK] for i in range(0, len(images), MAX_COLS_PER_BLOCK)]

        print(f"QC grid: {rat} ({len(images)} sections, {len(chunks)} block(s))", flush=True)

        for chunk_idx, chunk in enumerate(chunks):
            row_imgs = []

            for row_key in ROWS:
                tiles = []
                for img_name in chunk:
                    sec = parse_section_number(img_name)
                    target = TARGETS[row_key]
                    src = raw_tif_path(rat, sec, target)

                    if not src.exists():
                        # blank tile placeholder
                        tiles.append(np.zeros((THUMB_H, THUMB_H, 3), dtype=np.uint8))
                        continue

                    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                    if raw is None:
                        tiles.append(np.zeros((THUMB_H, THUMB_H, 3), dtype=np.uint8))
                        continue

                    raw = pick_signal_plane(raw).astype(np.float32)
                    img_u8 = enhance(raw, row_key)

                    # lookup transform by (rat, section_label/image)
                    rot, mir = (0, 0)
                    # manifest likely uses section_label not "image"; but we support either
                    # try exact match first
                    if (rat, img_name) in transform_map:
                        rot, mir = transform_map[(rat, img_name)]
                    else:
                        # also try "section_label" format if yours differs (e.g., sec10 vs sec010)
                        # common case: section_label is "sec10"
                        sec_label_guess = f"sec{sec}"
                        if (rat, sec_label_guess) in transform_map:
                            rot, mir = transform_map[(rat, sec_label_guess)]

                    img_u8 = apply_transform(img_u8, rot, mir)
                    tiles.append(to_thumb(img_u8))

                strip = hstack_with_gaps(tiles)
                strip = np.hstack([label_cell(row_key, strip.shape[0]), strip])
                row_imgs.append(strip)

            # stack rows for this block
            block = row_imgs[0]
            for rimg in row_imgs[1:]:
                gap = np.zeros((ROW_GAP, block.shape[1], 3), dtype=np.uint8)
                # pad row width if needed
                if rimg.shape[1] < block.shape[1]:
                    pad = np.zeros((rimg.shape[0], block.shape[1] - rimg.shape[1], 3), dtype=np.uint8)
                    rimg = np.hstack([rimg, pad])
                elif rimg.shape[1] > block.shape[1]:
                    pad = np.zeros((block.shape[0], rimg.shape[1] - block.shape[1], 3), dtype=np.uint8)
                    block = np.hstack([block, pad])
                block = np.vstack([block, gap, rimg])

            blocks.append(block)

        # stack blocks vertically with spacing
        final = blocks[0]
        for b in blocks[1:]:
            gap = np.zeros((BLOCK_GAP, max(final.shape[1], b.shape[1]), 3), dtype=np.uint8)
            # pad widths
            if final.shape[1] < gap.shape[1]:
                pad = np.zeros((final.shape[0], gap.shape[1] - final.shape[1], 3), dtype=np.uint8)
                final = np.hstack([final, pad])
            if b.shape[1] < gap.shape[1]:
                pad = np.zeros((b.shape[0], gap.shape[1] - b.shape[1], 3), dtype=np.uint8)
                b = np.hstack([b, pad])
            final = np.vstack([final, gap, b])

        out_path = OUT / f"qc_grid_{rat}.png"
        cv2.imwrite(str(out_path), final)
        print(f"  wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()