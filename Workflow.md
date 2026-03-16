# On Windows PC at work
## Pre-processing
1. Scan dataset structure
2. Validate tile completeness
3. Generate a dataset manifest
4. Build an XY --> rat lookup table.

STEP 1: Scan the dataset structure
CGPT generate script that produces:
* dataset_inventory.csv --> list of all XY tiles and SETS
* tile_map.csv --> automatically pull grid arrangement from .gci files
* validation_report.txt
* dataset_summary.txt

