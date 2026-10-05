Macro-F1 per split (classes present in each split)

| model | in-dist | unseen header | header-less | unseen locale |
|---|---|---|---|---|
| selected:lightgbm | 0.998 | 1.000 | 0.998 | 0.831 |
| ablation:lightgbm_no_header_dropout | 0.998 | 1.000 | 0.998 | 0.830 |
| other:logreg | 0.992 | 0.994 | 0.992 | 0.805 |
| baseline:regex_only | 0.746 | 0.727 | 0.746 | 0.579 |
| baseline:header_keyword_only | 0.526 | 0.223 | 0.083 | 0.572 |
