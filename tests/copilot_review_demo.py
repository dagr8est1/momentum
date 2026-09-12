"""Temporary file with intentional issues to verify Copilot code review comments on PRs.

Not imported or wired into any real code path (and not named test_*.py, so pytest
won't collect it). Delete once Copilot review has been confirmed working on this PR.
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
