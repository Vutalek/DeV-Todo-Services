import time
from threading import RLock

import requests
from fastapi import HTTPException

from app import config

TRELLO_CACHE_TTL_SECONDS = 300

trello_cache_lock = RLock()
trello_cache: dict[str, object] = {
    "expires_at": 0.0,
    "data": None,
}
trello_cache_refresh_lock = RLock()


def get_trello_data():
    now = time.monotonic()
    with trello_cache_lock:
        cached_data = trello_cache["data"]
        if cached_data is not None and now < trello_cache["expires_at"]:
            return cached_data

    with trello_cache_refresh_lock:
        now = time.monotonic()
        with trello_cache_lock:
            cached_data = trello_cache["data"]
            if cached_data is not None and now < trello_cache["expires_at"]:
                return cached_data

        lists_url = f"https://api.trello.com/1/boards/{config.BOARD_ID}/lists"
        labels_url = f"https://api.trello.com/1/boards/{config.BOARD_ID}/labels"
        query = {'key': config.TRELLO_KEY, 'token': config.TRELLO_TOKEN}

        try:
            lists_response = requests.get(lists_url, params=query, timeout=15)
            labels_response = requests.get(
                labels_url, params=query, timeout=15)
            lists_response.raise_for_status()
            labels_response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Trello data: {exc}",
            ) from exc

        lists_data = lists_response.json()
        labels_data = labels_response.json()

        col_map = {l['name']: l['id'] for l in lists_data}
        lab_map = {lb['name']: lb['id'] for lb in labels_data if lb['name']}

        data = (col_map, lab_map)
        with trello_cache_lock:
            trello_cache["data"] = data
            trello_cache["expires_at"] = time.monotonic() + \
                TRELLO_CACHE_TTL_SECONDS

        return data
