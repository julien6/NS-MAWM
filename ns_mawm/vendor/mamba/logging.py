"""Local metric sink; no external service or account is required."""
records = []
def init(**kwargs):
    records.clear()
def log(values):
    records.append({k: float(v) for k, v in values.items()})
    del records[:-100]
