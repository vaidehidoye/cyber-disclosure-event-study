|                                          |   n | mean_car   | median_car   | pct_negative   |     t |    p_t |   bmp_t |   p_bmp |   p_wilcoxon |
|:-----------------------------------------|----:|:-----------|:-------------|:---------------|------:|-------:|--------:|--------:|-------------:|
| Baseline: market model                   |  42 | -2.74%     | -1.98%       | 80.95%         | -4.76 | 0      |   -3.98 |  0.0003 |       0      |
| Including unreliable price series        |  45 | -2.90%     | -2.03%       | 80.00%         | -3.63 | 0.0007 |   -3.98 |  0.0003 |       0      |
| Fama-French 3-factor model               |  41 | -2.97%     | -2.06%       | 82.93%         | -5.04 | 0      |   -4.28 |  0.0001 |       0      |
| Window [-1, +1]                          |  42 | -2.84%     | -1.70%       | 76.19%         | -4.54 | 0      |   -3.97 |  0.0003 |       0.0001 |
| Window [+0, +2]                          |  42 | -2.91%     | -2.32%       | 80.95%         | -4.49 | 0.0001 |   -3.9  |  0.0004 |       0      |
| Winsorized at 5% / 95%                   |  42 | -2.64%     | -1.98%       | 80.95%         | -5.06 | 0      |   -3.98 |  0.0003 |       0      |
| Excluding smallest third (dollar volume) |  28 | -3.31%     | -1.98%       | 82.14%         | -4.29 | 0.0002 |   -3.47 |  0.0018 |       0      |
| Excluding earnings-confounded filing     |  41 | -2.87%     | -2.03%       | 82.93%         | -4.97 | 0      |   -4.07 |  0.0002 |       0      |
| Comparison: already disclosed earlier    |   7 | -0.35%     | -0.88%       | 57.14%         | -0.24 | 0.8193 |   -0.68 |  0.5218 |       0.6875 |

Placebo: replacing each firm's disclosure date with a random non-event date (200 draws) gives a mean CAR[0,+1] of 0.01% (5th-95th percentile: -1.07% to 1.23%). 0 of 200 draws are as negative as the actual -2.74%.
