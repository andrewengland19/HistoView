1. python extract_features.py (from conda microscopy)
    define INPUT, PNG_OUTPUT, and FEATURE_OUTPUT
2. python run_pca_sort.py  (from conda research)
3. python qc_order_strips.py
    produces DAPI strip for quick visualization
4. downsample_tiffs.py
    downsamples remaining samples to small PNG files for QC
5. qc_grid_all.py
    produces the nice-looking QC grids for each animal