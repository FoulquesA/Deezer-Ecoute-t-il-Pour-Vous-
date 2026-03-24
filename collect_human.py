import sqlite3
import requests
import time

BASE_URL = "https://api.deezer.com"

SEARCH_KEYWORDS = [
    "mix 2026", "vibes", "chill playlist", "workout",
    "road trip", "late night", "summer 2025", "indie mix",
    "hip hop mix", "pop hits", "rock mix", "morning playlist",
    "feel good", "throwback", "dance mix", "acoustic",
    "evening vibes", "top songs", "top 2025", "party playlist", "focus"
]

# Comptes éditoriaux à exclure
BLACKLISTED_OWNER_NAMES = {
    "deezer", "deezer editors", "deezer france", "deezer music",
    "digster", "filtr", "topsify"
}

MIN_FANS = 1000
MIN_TRACKS = 50
TARGET_COUNT = 120


def get_playlist_meta(playlist_id):
    r = requests.get(f"{BASE_URL}/playlist/{playlist_id}")
    if r.status_code != 200:
        return None
    data = r.json()
    if "error" in data:
        return None
    return data


def search_playlists(keyword, limit=50):
    r = requests.get(f"{BASE_URL}/search/playlist", params={
        "q": keyword,
        "limit": limit
    })
    if r.status_code != 200:
        return []
    data = r.json()
    if "error" in data:
        return []
    return data.get("data", [])


def get_playlist_tracks(playlist_id):
    tracks = []
    url = f"{BASE_URL}/playlist/{playlist_id}/tracks?limit=100"
    while url:
        r = requests.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        if "error" in data:
            break
        tracks.extend(data.get("data", []))
        url = data.get("next")
        time.sleep(0.1)
    return tracks


def insert_playlist(conn, p, playlist_type):
    conn.execute("""
        INSERT OR IGNORE INTO playlists
        (playlist_id, name, owner_id, owner_name, fans, tracks_total, type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        p["id"], p["title"],
        str(p["creator"]["id"]), p["creator"]["name"],
        p.get("fans", 0), p.get("nb_tracks", 0),
        playlist_type
    ))


def insert_track(conn, t):
    conn.execute("""
        INSERT OR IGNORE INTO tracks
        (track_id, title, artist_id, artist_name, duration, rank, explicit, preview_url, deezer_bpm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        t["id"], t.get("title", ""),
        t["artist"]["id"], t["artist"]["name"],
        t.get("duration", 0),
        t.get("rank", 0),
        int(t.get("explicit_lyrics", False)),
        t.get("preview", ""),
        t.get("bpm", 0.0)
    ))


def insert_playlist_track(conn, playlist_id, track_id, position):
    conn.execute("""
        INSERT OR IGNORE INTO playlist_tracks (playlist_id, track_id, position)
        VALUES (?, ?, ?)
    """, (playlist_id, track_id, position))


def search_human_playlists():
    seen_ids = set()
    candidates = []

    for keyword in SEARCH_KEYWORDS:
        if len(candidates) >= TARGET_COUNT:
            break

        print(f"Recherche : '{keyword}'...")
        results = search_playlists(keyword)

        for p in results:
            if not p or not p.get("id"):
                continue

            pid = p["id"]
            if pid in seen_ids:
                continue

            owner_name = p.get("creator", {}).get("name", "").lower()
            if any(b in owner_name for b in BLACKLISTED_OWNER_NAMES):
                continue

            nb_tracks = p.get("nb_tracks", 0)
            if nb_tracks < MIN_TRACKS:
                continue

            # Récupère les métadonnées complètes pour avoir les fans
            try:
                meta = get_playlist_meta(pid)
                if not meta:
                    continue

                fans = meta.get("fans", 0)
                if fans < MIN_FANS:
                    continue

                # Double-check que c'est pas éditorial
                owner_name_full = meta["creator"]["name"].lower()
                if any(b in owner_name_full for b in BLACKLISTED_OWNER_NAMES):
                    continue

                seen_ids.add(pid)
                candidates.append(meta)
                print(f"  + {meta['title']} ({fans:,} fans)")
                time.sleep(0.15)

            except Exception as e:
                print(f"  -> Erreur {pid} : {e}")
                continue

        time.sleep(0.3)

    return candidates


def collect_human():
    candidates = search_human_playlists()
    print(f"\n{len(candidates)} playlists humaines trouvées. Collecte des tracks...\n")

    with sqlite3.connect("deezer_study.db") as conn:
        for meta in candidates:
            pid = meta["id"]
            try:
                print(f"Collecte : {meta['title']}...")
                insert_playlist(conn, meta, "human")

                tracks = get_playlist_tracks(pid)
                valid_tracks = [t for t in tracks if t.get("id") and t.get("preview")]

                for i, t in enumerate(valid_tracks):
                    insert_track(conn, t)
                    insert_playlist_track(conn, pid, t["id"], i)

                conn.commit()
                print(f"  -> OK : {len(valid_tracks)} tracks avec preview")
                time.sleep(0.2)

            except Exception as e:
                print(f"  -> Erreur sur {pid} : {e}")
                continue

    print("\nCollecte humaine terminée.")


collect_human()
