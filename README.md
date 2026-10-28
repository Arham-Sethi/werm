# WERM - complete first-party source snapshot

Prepared for repository review, not published. This package preserves source variants rather than silently replacing historical experiments.

## Contents and version boundary

- `desktop/`: all 75 files from the previously reviewed WERM staging group: 52 Python sources (50 under WERM/src plus two root spatial scripts), two configuration files, and 21 small result/input-table CSVs. Original relative layout retained.
- `drive-archive-001/`: six selected first-party/experiment-record files from the user-supplied `WERM-20261028T084321Z-1-001.zip`: two Python sources plus one fusion CSV and three run metadata/comparison JSONs.
- `source-manifest.json`: 81 input files with origin, source-relative path, SHA-256, size, syntax check and portability notes. Identical bytes do not erase path provenance.
- `archive-exclusions.json`: every omitted archive entry with its reason. Images, checkpoints, cached bytecode, bundled pip/venv code and external dataset code are excluded.
- `VERSION-MAP.md`: explicit experiment boundaries and identical-content mapping.

The 35 Python entries in the Drive ZIP comprise TWO first-party source files, TWO third-party Modified-Unet dataset utility files, and 31 environment/vendor files. Both first-party Drive sources exactly match desktop files at different paths. They are retained separately as provenance, not mislabeled as 33 new WERM implementations. Only the supplied -001 archive part was inspected here; other multi-part downloads are not assumed complete.

## How to inspect or reproduce

Start from `desktop/WERM/src/fusion_engine.py` for reusable arithmetic, `desktop/WERM/config/fusion_config.yaml` and `desktop/WERM/src/experiments/` for experimental settings, and the paired spatial scripts for grid analyses. `fusion_final.py` is a separate integrated script. No single entry point is designated authoritative for all paper figures.

Python sources parse successfully, but training, image inference, weather requests and experiment regeneration were NOT run. Many scripts retain Colab/Drive or local paths. Adapt these in a new run configuration; preserve original snapshots and record changed values. The alpha-sweep reference is assigned after some comparisons in the historical script; retain historical results while fixing a new version.

No original environment lockfile was found in this selected WERM source. `IMPORT-INVENTORY.md` records imported module names; common libraries include NumPy, pandas, torch/torchvision, scikit-learn, OpenCV, matplotlib and image/statistical utilities, but this is not a tested installation recipe. Do not invent matching package pins.

## Model/data acquisition and release limits

Acquire the exact moisture regression/multitask checkpoint, residue checkpoint and training/export notebooks; identify dataset source/licenses, frozen label/split manifests, image hashes and weather snapshots. The local three-class moisture trainer does not by itself establish the regression checkpoint used by fusion. Downloaded third-party dataset code requires its own license/attribution review before redistribution. Large model weights and source photos belong in separately versioned model/data artifacts, not this source repository.

Included CSV/JSONs document saved outputs, not an independent rerun. Full reproduction additionally needs the exact source version, checkpoints, frozen inputs, calibration parameters, seeds, acquisition dates and figure-to-run mapping. No blanket reproducibility, validation or licensing claim is made.

## Source audit

Credential-shaped literals/private-key headers were scanned without exposing values; no selected file was withheld by this scan. Hardcoded local/Colab paths and environment use are listed per file in the manifest. This is a bounded static check, not a security certification. Original files remain unchanged. No private .env or Git history is included. Licensing/authorship for a public release still needs an explicit decision by the project owners.


## Browse source

[Every published file](FILE-INDEX.md) · [Snapshot hashes](publication-manifest.json)
