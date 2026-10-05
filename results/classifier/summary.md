Macro-F1 per split (classes present in each split)

| model | in-dist | unseen header | header-less | unseen locale |
|---|---|---|---|---|
| selected:logreg | 0.998 | 0.996 | 0.996 | 0.806 |
| ablation:logreg_no_header_dropout | 0.998 | 0.996 | 0.996 | 0.808 |
| other:lightgbm | 0.996 | 0.996 | 0.996 | 0.867 |
| baseline:regex_only | 0.686 | 0.697 | 0.686 | 0.565 |
| baseline:header_keyword_only | 0.538 | 0.191 | 0.046 | 0.544 |
| baseline:presidio | 0.610 | 0.574 | 0.610 | 0.553 |
