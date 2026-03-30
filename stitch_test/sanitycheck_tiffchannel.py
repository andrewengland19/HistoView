import tifffile
import numpy as np

img = tifffile.imread("stitched_XY01_CH1.tif")

print("shape:", img.shape)
print("dtype:", img.dtype)

if img.ndim == 3:
    for i in range(img.shape[2]):
        ch = img[:,:,i]
        print(f"channel {i}: min={ch.min()} max={ch.max()} mean={ch.mean()}")

else:
    print("min:", img.min(), "max:", img.max(), "mean:", img.mean())