# OR_tools

This repository is now organized by function rather than by ad hoc experiment folders.

## Layout

```text
OR_tools/
├── analysis/
│   └── data_prep/
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── main_optimizer/
│   ├── evaluation/
│   │   ├── whole_order/
│   │   └── partial_order/
│   └── literature/
│       ├── hybrid_v2/
│       └── benchmark_vs_hybrid_v2/
├── src/
│   ├── main_optimizer/
│   ├── evaluation/
│   └── literature/
├── 对比文献2/
├── 整单结果/
└── 部分履约结果/
```

## Source Entry Points

- Main optimizer: `src/main_optimizer/dark_store_optimizer_with_plans.py`
- Whole-order basket replay: `src/evaluation/basket_simulation_whole.py`
- Partial-order basket replay: `src/evaluation/basket_simulation_partial.py`
- Literature baseline: `src/literature/li_transchel_hybrid_baseline_v2.py`
- Benchmark vs literature baseline: `src/literature/benchmark_vs_hybrid_v2.py`
- Data prep and EDA: `analysis/data_prep/dark_store_data_prep.py`

## Data

- Raw scanner data: `data/raw/scanner_data.csv`
- Processed optimizer inputs: `data/processed/sku_params.csv`, `data/processed/daily_demand.csv`

## Results

- Main optimizer outputs: `results/main_optimizer/`
- Basket replay outputs: `results/evaluation/`
- Literature baseline and benchmark outputs: `results/literature/`

## Compatibility

Legacy folders are now reduced to lightweight pointers only.

- Generated outputs are kept only under `results/`
- Source files are kept only under `src/`
- Legacy folders such as `对比文献2`, `整单结果`, and `部分履约结果` now contain notes or old entry points, not duplicated result files
