# Forecasting & replenishment ML standards

These standards apply to every model that feeds a stocking decision in the domain. They are short on
purpose: each rule exists because breaking it has cost money somewhere, and each one is enforced
by code in this repository, not by a review comment.

## 1. One contract for every forecaster

A model implements `fit(fb, origin)` and `predict(fb, origin, horizon) -> Forecast`
(`src/dre/models/base.py`). A `Forecast` is a **distribution**, not a point: mean and negative-binomial
dispersion per series and day. Ordering decisions need quantiles. A point forecast cannot
set a safety stock.

*Why:* the optimiser, the backtest, the gate and the API consume the same object, so a squad can
swap models without touching anything downstream.

## 2. Features only see the past

Every feature row is built from data up to the forecast origin, plus what is known in advance about
the target day: calendar, events and the planned price (`src/dre/features.py`).
`tests/test_data_features.py::test_features_use_no_future_data` overwrites every day after the
origin and checks the features do not change. The test is required for every new feature.

Data-derived labels obey the same rule. A stock-out flag depends on how long a zero run lasts, so it
is recomputed per backtest origin from data up to that origin (`pipeline.feature_builder`).

## 3. Sales are not demand

Recorded sales are censored by stock-outs. Zero runs that are improbable given the item's recent
selling rate are flagged (`src/dre/data/stockouts.py`). Long gaps (over 14 days) are treated as
delisting, not lost demand. Whether to mask flagged days from training targets is a modelling
choice that must be justified with evidence, not assumed. Two pieces of evidence count:
- **Planted stock-outs** (`evaluation/censoring.py`). With known truth, masking removed the bias that
  stock-outs put into the forecast.
- **The release gate on real history.** On M5, masking short stock-outs made no measurable
  difference, so it was rejected.

During development, masking every flagged run, months-long gaps included, over-forecast by about 7%
on one fold, which is why runs over 14 days are no longer flagged.

## 4. One evaluation procedure

- Rolling-origin backtest over at least three folds, retraining at every origin
  (`evaluation/backtest.py`). Each fold's uncertainty is calibrated on the previous fold's
  out-of-sample errors, never on the period being scored.
- Report WRMSSE (12 hierarchy levels), WSPL, interval coverage, bias and WAPE (days 1–28 and 1–7),
  plus fit time. The WRMSSE implementation is pinned to the published M5 seasonal-naive score
  (0.847).
- Always include the baselines (seasonal naive, moving average, TSB). A model that does not beat a
  moving average is not discussed further.

## 5. The release gate decides, not the author

`evaluation/gate.py`, with thresholds in `config/dre.yaml`. A challenger replaces the champion only
if **all** of these hold:

1. Its WRMSSE is lower, averaged over the folds.
2. The dollar-weighted per-series improvement is significant (the 95% paired-bootstrap interval
   lies below zero).
3. WSPL is not worse by more than 1%.
4. Absolute bias is not worse by more than 1 point.
5. It wins at least 2 of the 3 folds.

The decision and its evidence are stored with the model in the registry.

## 6. Value is measured in the decision, not the metric

A forecast improvement is reported in euros. The replenishment simulation (`replenishment/`)
replays real demand day by day and reports, for each policy:

- fill rate, waste and capital tied up;
- margin after waste.

It also separates the value of a better forecast from the value of a better policy:

- **Better forecast:** the same newsvendor policy run on the moving-average forecast vs the champion.
- **Better policy:** cover-days vs newsvendor vs perishable-aware vs capital-constrained, all on the
  champion forecast.

## 7. Reproducible and auditable

Every registered model carries:
- its artifact;
- the config hash, the code version and its backtest metrics;
- the gate decision that promoted it;
- the forecast it served (`src/dre/registry.py`).

Economic assumptions such as cost, shelf life, packs and delivery days live in config and are the
same for every policy.

## 8. Monitored from day one

Daily data checks (missing or negative sales, unpriced listings, stock-out flag share). Accuracy
and bias by store and department, with alerts relative to the backtest. PSI drift on key inputs
(`monitoring/report.py`). A store-level bias alert is investigated before any model change: it is
usually a supply problem or a feed problem.

## 9. LLMs explain decisions; they do not make them

Order quantities come from the optimiser. An LLM may rephrase the drivers of an order for a store
manager. Every number in its text must match a driver, or a deterministic template is used
instead (`explain/llm.py`). The service works without any LLM.

## Review checklist (for every model PR)

- [ ] Implements the `Forecaster` contract and returns a distribution
- [ ] No-future-data test passes for every new feature
- [ ] Backtest on all folds, with baselines in the same table
- [ ] Gate result attached; any failed check is explained
- [ ] Replenishment simulation run: € impact by temperature zone
- [ ] Model card regenerated; limitations updated
- [ ] Monitoring: new inputs added to the drift list
- [ ] Runtime and memory fit the nightly batch window
