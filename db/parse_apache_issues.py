import time
import requests
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(BASE_DIR, 'chroma_db', 'apache_issues.csv')
BASE_URL = 'https://issues.apache.org/jira'
SEARCH_URL = f'{BASE_URL}/rest/api/2/search'

JQL = 'project = HADOOP AND status = Closed'

FIELDS = [
    'summary',
    'issuetype',
    'priority',
    'created',
    'description',
    'resolutiondate',
]

LIMIT = 5000
PAGE_SIZE = 1000

rows = []
start_at = 0
jira_error = None

while len(rows) < LIMIT:
    params = {
        'jql': JQL,
        'startAt': start_at,
        'maxResults': PAGE_SIZE,
        'fields': ','.join(FIELDS),
    }

    try:
        response = requests.get(
            SEARCH_URL,
            params=params,
            headers={'Accept': 'application/json'},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        jira_error = exc
        print(f'Warning: failed to load Jira issues: {exc}')
        break

    issues = data.get('issues', [])

    if not issues:
        break

    for issue in issues:
        f = issue['fields']

        rows.append({
            'url': f"{BASE_URL}/browse/{issue['key']}",
            'desc': f.get('description'),
            'name': f.get('summary'),
            'issue_type': (
                f.get('issuetype', {}).get('name')
                if f.get('issuetype') else None
            ),
            'priority': (
                f.get('priority', {}).get('name')
                if f.get('priority') else None
            ),
            'created': f.get('created'),
            'resolved': f.get('resolutiondate'),
        })

        if len(rows) >= LIMIT:
            break

    print(f'loaded {len(rows)} issues')

    start_at += len(issues)

    if start_at >= data.get('total', 0):
        break

    time.sleep(0.2)

df = pd.DataFrame(
    rows,
    columns=['url', 'desc', 'name', 'issue_type',
             'priority', 'created', 'resolved'],
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

if jira_error is not None:
    print(f'Warning: wrote {len(rows)} Jira seed rows after error')
