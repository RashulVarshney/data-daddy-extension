Provider: mock/naive-mock-v1 (MOCK harness validation)

| category | n | OFF success | ON success |
|---|---|---|---|
| direct_injection | 10 | 10/10 | 0/10 |
| indirect_injection | 14 | 13/14 | 4/14 |
| tool_misuse | 10 | 10/10 | 0/10 |
| raw_row_exfiltration | 10 | 10/10 | 0/10 |
| policy_bypass | 10 | 7/10 | 0/10 |
| misleading_claims | 8 | 8/8 | 0/8 |
| **total** | 62 | 58/62 | 4/62 |

| mode | benign success | benign blocked | p50 wall s | p95 wall s | tokens/case |
|---|---|---|---|---|---|
| off | 25/30 | 0/30 | 0.02 | 0.03 | 466 |
| on | 29/30 | 0/30 | 0.02 | 0.03 | 466 |
