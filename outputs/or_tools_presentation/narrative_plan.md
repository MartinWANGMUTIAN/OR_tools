Audience: instructor, advisor, or thesis defense panel evaluating a research project on dark-store assortment and inventory optimization.

Objective: explain the project from business motivation through literature review, modeling, solution approach, data processing, empirical results, managerial implications, and limitations in a way that supports a 15-20 minute oral presentation.

Narrative arc:
1. Explain why front warehouses face a structural tradeoff between assortment breadth and inventory depth.
2. Position the project against existing literature and clarify the research gap.
3. Present the joint optimization logic, demand estimation approach, and solution methods.
4. Show the data pipeline and descriptive findings from scanner data.
5. Report the main optimization, replay, and benchmark results using project outputs.
6. Close with discussion, managerial implications, limitations, and future extensions.

Visual system:
- 16:9 editable academic-business presentation
- light paper background, dark ink text, green/gold/coral accents
- serif titles and sans-serif body copy
- structured card-based layouts with high editability
- no rasterized text; all visible text lives in native PowerPoint objects

Slide list:
1. Cover
2. Research background
3. Core research question
4. Research significance
5. Literature review overview
6. Gap and contribution
7. Research framework
8. Business setting and modeling logic
9. Optimization model
10. Demand modeling and parameter estimation
11. Solution methods
12. Rolling-window experiment design
13. Data source and cleaning
14. Descriptive data highlights
15. Main optimization results
16. Order replay evaluation
17. Literature benchmark comparison
18. Discussion and interpretation
19. Managerial implications
20. Conclusion, limitations, and future work

Source plan:
- project README for repository structure and workflow
- src/main_optimizer/dark_store_optimizer_with_plans.py for model, scenario, and algorithm descriptions
- analysis/data_prep/dark_store_data_prep.py for data construction logic
- results/main_optimizer/summary.csv for primary optimization results
- results/evaluation/whole_order/basket_summary.csv and partial_order/basket_summary.csv for replay findings
- results/literature/benchmark_vs_hybrid_v2/benchmark_whole_summary.csv and benchmark_partial_summary.csv for baseline comparison
- data/raw/scanner_data.csv and data/processed/sku_params.csv for descriptive sample counts

Editability plan:
- use native title, subtitle, card, and metric shapes only
- attach presenter guidance and source references in speaker notes on each slide
- export a single editable PPTX artifact to the output directory
