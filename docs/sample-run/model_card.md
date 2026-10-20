# Model card — torch_nb

Generated 2026-10-20T13:08:25+00:00 · data 14,370 store × item series, 2011-01-29 to 2016-06-19 (M5, FOODS) · registry `torch_nb-20261020T130824`

## Intended use
Daily demand forecasts (mean and negative-binomial distribution, 1–28 days) per store × item, consumed by the replenishment optimiser to set order quantities. Not intended for items without sales history (new listings need a cold-start rule) or for price setting.

## Backtest (rolling origin, mean over folds d_1885, d_1913, d_1941)
| Model | WRMSSE | WSPL | Bias | WAPE | WAPE days 1–7 | 50% cover | 95% cover |
|---|---|---|---|---|---|---|---|
| torch_nb | 0.568 | 0.2712 | 0.0078 | 0.6729 | 0.6503 | 0.7573 | 0.9676 |
| torch_nb_demand | 0.5855 | 0.2712 | 0.0215 | 0.6753 | 0.6521 | 0.7577 | 0.9682 |
| lightgbm_demand | 0.5863 | 0.2734 | -0.0141 | 0.6688 | 0.6478 | 0.7509 | 0.9657 |
| moving_average | 0.7521 | 0.2764 | -0.0145 | 0.6827 | 0.6742 | 0.7425 | 0.9551 |
| seasonal_naive | 0.9312 | 0.3949 | -0.0406 | 0.82 | 0.8137 | 0.6578 | 0.7942 |
| tsb | 1.0698 | 0.2894 | 0.0003 | 0.6923 | 0.6863 | 0.737 | 0.9599 |

WRMSSE as defined in M5, on the FOODS hierarchy (the implementation reproduces the published M5 seasonal-naive score of 0.847 on the full dataset).

## Release gate
- torch_nb vs moving_average: **promoted** — WRMSSE 0.7521 → 0.568, weighted RMSSE difference -0.02473 (95% CI [-0.02942, -0.02052]), folds won 3/3, checks all passed
- torch_nb_demand vs torch_nb: **rejected** — WRMSSE 0.568 → 0.5855, weighted RMSSE difference 0.00021 (95% CI [-0.00129, 0.00163]), folds won 1/3, checks lower_wrmsse, significant, bias_ok, consistent failed
- lightgbm_demand vs torch_nb: **rejected** — WRMSSE 0.568 → 0.5863, weighted RMSSE difference 0.00239 (95% CI [-0.00221, 0.0073]), folds won 1/3, checks lower_wrmsse, significant, consistent failed
- seasonal_naive vs torch_nb: **rejected** — WRMSSE 0.568 → 0.9312, weighted RMSSE difference 0.2727 (95% CI [0.26439, 0.28063]), folds won 0/3, checks lower_wrmsse, significant, wspl_ok, bias_ok, consistent failed
- tsb vs torch_nb: **rejected** — WRMSSE 0.568 → 1.0698, weighted RMSSE difference 0.03738 (95% CI [0.03279, 0.04162]), folds won 0/3, checks lower_wrmsse, significant, wspl_ok, consistent failed

## Replenishment value (simulated on the real daily demand of 2016-05-23 + 28 days)
| Policy | Fill rate | Waste / COGS | Avg inventory capital | Margin after waste | Stock-out item-days |
|---|---|---|---|---|---|
| cover_days | 90.7% | 4.2% | 190,772 | 527,559 | 5.5% |
| newsvendor[moving_average] | 92.7% | 6.3% | 237,583 | 512,463 | 3.5% |
| newsvendor[torch_nb] | 93.0% | 6.7% | 237,909 | 508,667 | 3.3% |
| perishable_newsvendor[moving_average] | 90.3% | 2.0% | 212,366 | 553,087 | 6.2% |
| perishable_newsvendor[torch_nb] | 91.0% | 2.1% | 213,122 | 556,824 | 5.5% |
| perishable_newsvendor_penalty_0.6[torch_nb] | 91.6% | 2.6% | 218,232 | 554,486 | 4.8% |
| capital_cap_3d_proportional[torch_nb] | 65.4% | 2.4% | 89,088 | 414,397 | 16.1% |
| capital_cap_3d_milp[torch_nb] | 74.9% | 2.1% | 96,043 | 459,862 | 14.6% |
| capital_cap_4d_proportional[torch_nb] | 79.2% | 2.1% | 126,640 | 493,228 | 10.2% |
| capital_cap_4d_milp[torch_nb] | 84.7% | 2.0% | 136,123 | 521,446 | 8.8% |
| oracle | 94.1% | 0.3% | 63,872 | 605,567 | 2.7% |

By temperature zone:

| Policy | Zone | Fill rate | Waste / COGS | Margin after waste |
|---|---|---|---|---|
| cover_days | fresh | 95.2% | 25.0% | 32,490 |
| cover_days | chilled | 93.5% | 3.5% | 140,476 |
| cover_days | ambient | 89.1% | 0.0% | 354,593 |
| newsvendor[moving_average] | fresh | 96.5% | 32.6% | 18,994 |
| newsvendor[moving_average] | chilled | 96.6% | 7.4% | 130,906 |
| newsvendor[moving_average] | ambient | 91.0% | 0.0% | 362,563 |
| newsvendor[torch_nb] | fresh | 97.1% | 35.3% | 14,016 |
| newsvendor[torch_nb] | chilled | 97.2% | 7.4% | 131,798 |
| newsvendor[torch_nb] | ambient | 91.1% | 0.0% | 362,852 |
| perishable_newsvendor[moving_average] | fresh | 82.8% | 9.7% | 51,276 |
| perishable_newsvendor[moving_average] | chilled | 92.7% | 3.3% | 139,351 |
| perishable_newsvendor[moving_average] | ambient | 91.0% | 0.0% | 362,460 |
| perishable_newsvendor[torch_nb] | fresh | 84.9% | 10.1% | 52,053 |
| perishable_newsvendor[torch_nb] | chilled | 94.8% | 3.6% | 141,930 |
| perishable_newsvendor[torch_nb] | ambient | 91.0% | 0.0% | 362,841 |
| perishable_newsvendor_penalty_0.6[torch_nb] | fresh | 88.6% | 12.7% | 50,436 |
| perishable_newsvendor_penalty_0.6[torch_nb] | chilled | 95.7% | 4.2% | 141,201 |
| perishable_newsvendor_penalty_0.6[torch_nb] | ambient | 91.1% | 0.0% | 362,849 |

## Censored demand
Planted stock-outs in CA_3 (909 items selling ≥ 1/day): the detector flags 54% of planted days and 49% of episodes.

| Training history | Bias | WAPE |
|---|---|---|
| clean history | -0.0593 | 0.6501 |
| damaged, censored days kept | -0.1314 | 0.6542 |
| damaged, censored days masked | -0.0599 | 0.6539 |

## Monitoring at go-live
Data checks: {'missing_sales_values': 0, 'negative_sales_values': 0, 'items_listed_without_price': 0, 'stockout_flag_share_last_28d': 0.01, 'stockout_flag_share_prior_year': 0.0114}

Drift (PSI): mean_28 0.0408 (green), zero_share_28 0.0338 (green), stockout_share_28 0.0 (green), price 0.0002 (green), price_vs_origin 0.0 (green), days_since_sale 0.0411 (green)

Alerts: 5
- CA_3 bias 0.0621 (threshold 0.05)
- CA_4 wape 0.844 (threshold 0.8075)
- WI_2 bias 0.0668 (threshold 0.05)
- FOODS_1 bias 0.0522 (threshold 0.05)
- FOODS_2 bias 0.0762 (threshold 0.05)

## Limitations
- M5 is Walmart supermarket data, not dark-store data: assortment, basket and delivery promise differ. The method transfers; the numbers do not.
- Recorded sales are censored demand. The replenishment simulation replays recorded sales, so policies that stock more cannot sell more than history did: fill-rate gains are lower bounds, waste is not.
- Costs, shelf lives, pack sizes, shelf capacity and delivery days are documented assumptions (config), identical for every policy.
- Departments stand in for temperature zones; M5 does not say which products are fresh.
