# Ablation Analysis of Exam Scores

## 1) Raw Metric Comparison
| Dataset | Formula | F1 | Rho | Tau | Videos |
|---|---|---:|---:|---:|---:|
| summe | s_mul_fs_add_fv | 0.3626 | 0.0166 | 0.0136 | 17 |
| summe | s_mul_fs | 0.3772 | 0.0138 | 0.0108 | 17 |
| summe | s_only | 0.3645 | 0.0072 | 0.0060 | 17 |
| summe | s_add_fs | 0.3659 | 0.0162 | 0.0134 | 17 |
| tvsum | s_mul_fs_add_fv | 0.5630 | 0.0072 | 0.0059 | 50 |
| tvsum | s_mul_fs | 0.5276 | 0.0059 | 0.0049 | 50 |
| tvsum | s_only | 0.5360 | 0.0115 | 0.0102 | 50 |
| tvsum | s_add_fs | 0.5679 | 0.0076 | 0.0062 | 50 |

## 2) Ranking by F1 (Primary Criterion)
| Dataset | Rank | Formula | F1 | Rho | Tau | Videos |
|---|---:|---|---:|---:|---:|---:|
| summe | 1 | s_mul_fs | 0.3772 | 0.0138 | 0.0108 | 17 |
| summe | 2 | s_add_fs | 0.3659 | 0.0162 | 0.0134 | 17 |
| summe | 3 | s_only | 0.3645 | 0.0072 | 0.0060 | 17 |
| summe | 4 | s_mul_fs_add_fv | 0.3626 | 0.0166 | 0.0136 | 17 |
| tvsum | 1 | s_add_fs | 0.5679 | 0.0076 | 0.0062 | 50 |
| tvsum | 2 | s_mul_fs_add_fv | 0.5630 | 0.0072 | 0.0059 | 50 |
| tvsum | 3 | s_only | 0.5360 | 0.0115 | 0.0102 | 50 |
| tvsum | 4 | s_mul_fs | 0.5276 | 0.0059 | 0.0049 | 50 |

## 3) Ranking by Correlation (rho + tau)
| Dataset | Rank | Formula | CorrScore(rho+tau) | Rho | Tau | F1 |
|---|---:|---|---:|---:|---:|---:|
| summe | 1 | s_mul_fs_add_fv | 0.0302 | 0.0166 | 0.0136 | 0.3626 |
| summe | 2 | s_add_fs | 0.0296 | 0.0162 | 0.0134 | 0.3659 |
| summe | 3 | s_mul_fs | 0.0246 | 0.0138 | 0.0108 | 0.3772 |
| summe | 4 | s_only | 0.0132 | 0.0072 | 0.0060 | 0.3645 |
| tvsum | 1 | s_only | 0.0217 | 0.0115 | 0.0102 | 0.5360 |
| tvsum | 2 | s_add_fs | 0.0139 | 0.0076 | 0.0062 | 0.5679 |
| tvsum | 3 | s_mul_fs_add_fv | 0.0131 | 0.0072 | 0.0059 | 0.5630 |
| tvsum | 4 | s_mul_fs | 0.0108 | 0.0059 | 0.0049 | 0.5276 |

## 4) Multi-objective Ranking (F1 + normalized correlation)
| Dataset | Rank | Formula | MultiScore | F1 | NormRho | NormTau |
|---|---:|---|---:|---:|---:|---:|
| summe | 1 | s_mul_fs_add_fv | 0.6813 | 0.3626 | 1.0000 | 1.0000 |
| summe | 2 | s_add_fs | 0.6638 | 0.3659 | 0.9616 | 0.9617 |
| summe | 3 | s_mul_fs | 0.5211 | 0.3772 | 0.7075 | 0.6225 |
| summe | 4 | s_only | 0.1823 | 0.3645 | 0.0000 | 0.0000 |
| tvsum | 1 | s_only | 0.7680 | 0.5360 | 1.0000 | 1.0000 |
| tvsum | 2 | s_add_fs | 0.4227 | 0.5679 | 0.3062 | 0.2488 |
| tvsum | 3 | s_mul_fs_add_fv | 0.3853 | 0.5630 | 0.2311 | 0.1842 |
| tvsum | 4 | s_mul_fs | 0.2638 | 0.5276 | 0.0000 | 0.0000 |

## 5) Coverage & Data Completeness
| Dataset | Formula | TotalVideos | Evaluated(ok) | MissingInputs | Error/Other | Coverage |
|---|---|---:|---:|---:|---:|---:|
| summe | s_mul_fs_add_fv | 25 | 17 | 8 | 0 | 0.6800 |
| summe | s_mul_fs | 25 | 17 | 8 | 0 | 0.6800 |
| summe | s_only | 25 | 17 | 8 | 0 | 0.6800 |
| summe | s_add_fs | 25 | 17 | 8 | 0 | 0.6800 |
| tvsum | s_mul_fs_add_fv | 50 | 50 | 0 | 0 | 1.0000 |
| tvsum | s_mul_fs | 50 | 50 | 0 | 0 | 1.0000 |
| tvsum | s_only | 50 | 50 | 0 | 0 | 1.0000 |
| tvsum | s_add_fs | 50 | 50 | 0 | 0 | 1.0000 |

## Best Results Summary

- Best by F1 (summe): **s_mul_fs**, F1=0.3772
- Best by F1 (tvsum): **s_add_fs**, F1=0.5679
- Best by Corr (summe): **s_mul_fs_add_fv**, rho+tau=0.0302
- Best by Corr (tvsum): **s_only**, rho+tau=0.0217
- Best by Multi-score (summe): **s_mul_fs_add_fv**, multi=0.6813
- Best by Multi-score (tvsum): **s_only**, multi=0.7680