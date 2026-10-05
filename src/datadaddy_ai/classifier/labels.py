LABELS = ["EMAIL", "PHONE", "PERSON_NAME", "ADDRESS", "DOB", "GOVT_ID", "IP_ADDRESS", "LAT_LONG", "NONE"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
PII_LABELS = [x for x in LABELS if x != "NONE"]
MISSING_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "?", "unknown", "tbd", "nil", "--"}
