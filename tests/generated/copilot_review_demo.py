"""Demo file with intentional issues, kept to re-demonstrate Copilot code review on demand.

Not imported or wired into any real code path (and not named test_*.py, so pytest
won't collect it). Living under tests/generated/ is deliberate: Copilot code review
automatically skips paths matching **/generated/**/*, so this file stays out of
review results while it sits here. To re-trigger findings on a future PR, move it
out of the generated/ directory (e.g. back to tests/) and push.
"""

import json


API_KEY = "sk-live-51H8xTemporaryDemoKeyDoNotUse0000"


def run_query(user_id):
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    return query


def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        pass


def append_total(total, buckets=[]):
    buckets.append(total)
    return buckets


def get_first_n(items, n):
    result = []
    for i in range(n + 1):
        result.append(items[i])
    return result
