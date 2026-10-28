# WERM version and provenance map

| Area | Preserved implementation | Boundary |
|---|---|---|
| Historical additive prototype | desktop/WERM/src/fuse_risk.py | Includes placeholder inputs and earlier class convention; not canonical final fusion |
| Product inference variants | fusion_infer.py and fusion_infer1.py under desktop/WERM/src | Different model-head and texture assumptions; do not merge by filename similarity |
| Integrated geometric analysis | desktop/WERM/src/fusion_final.py | Uses robust-scale factor and tanh-width distinct from reusable engine |
| Reusable numerical core | desktop/WERM/src/fusion_engine.py | Numeric scale differs from integrated script even when rank ordering agrees |
| Temperature/texture versions | temperature_gdd.py, tillage_module.py, tillage_texture.py | Frozen weather/date and texture policy are run identity |
| Reusable experiment support | experiment_runner.py, config and experiments | Numbered scripts also retain hardcoded paths; config alone does not define every historical run |
| Desktop model trainers | soil_moisture/ and residue_cover/ | Classification vs regression checkpoint lineage must be resolved |
| Spatial work | desktop/run_spatial_werm.py, spatial_analysis.py and small CSVs | 61-image grid distinct from 333-image factor analysis |
| Drive flattened trainer | drive-archive-001/WERM/src/train.py | SHA-identical to desktop/WERM/src/soil_moisture/train.py |
| Drive flattened residue loader | drive-archive-001/WERM/src/dataset_residue.py | SHA-identical to desktop/WERM/src/residue_cover/dataset_residue.py |
| Drive saved experiment records | drive-archive-001/WERM/outputs/ | Three run JSONs and fusion CSV; not silently substituted for desktop results |

All hashes and paths are in source-manifest.json. Duplicate migration CSV groups are retained because they correspond to named species/scaling conditions. Matching bytes do not establish that conditions are redundant. Original A-high and later A-low/E-high conventions must be labeled with the producing version.
