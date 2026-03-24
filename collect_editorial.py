import sqlite3
import requests
import time

BASE_URL = "https://api.deezer.com"

GENRE_IDS = [
    0, 116, 132, 152, 113, 165, 106, 466, 144, 129, 98, 85, 169
]

MIN_TRACKS = 50


def get_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "error" in data:
            return None
        return data
    except Exception as e:
        print(f"  -> Requête échouée : {e}")
        return None


def fetch_editorial_playlists():
    playlists = {}

    print("Récupération chart global...")
    data = get_json(f"{BASE_URL}/chart/0/playlists", params={"limit": 50})
    if data:
        for p in data.get("data", []):
            playlists[p["id"]] = p
        print(f"  -> {len(data.get('data', []))} playlists")
    time.sleep(0.3)

    for genre_id in GENRE_IDS:
        print(f"Récupération genre {genre_id}...")
        data = get_json(f"{BASE_URL}/genre/{genre_id}/charts", params={"limit": 50})
        if data:
            genre_playlists = data.get("playlists", {}).get("data", [])
            for p in genre_playlists:
                playlists[p["id"]] = p
            print(f"  -> {len(genre_playlists)} playlists")
        time.sleep(0.3)

    print(f"\nTotal brut : {len(playlists)} playlists uniques.\n")
    return list(playlists.values())


def get_playlist_full(playlist_id):
    return get_json(f"{BASE_URL}/playlist/{playlist_id}")


def get_playlist_tracks(playlist_id):
    tracks = []
    url = f"{BASE_URL}/playlist/{playlist_id}/tracks?limit=100"
    while url:
        data = get_json(url)
        if not data:
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


def collect_editorial():
    raw_playlists = fetch_editorial_playlists()
    collected = 0
    skipped = 0

    with sqlite3.connect("deezer_study.db") as conn:
        for p in raw_playlists:
            pid = p["id"]
            meta = get_playlist_full(pid)
            if not meta:
                skipped += 1
                continue

            nb_tracks = meta.get("nb_tracks", 0)
            if nb_tracks < MIN_TRACKS:
                skipped += 1
                continue

            print(f"Collecte : {meta['title']} (owner: {meta['creator']['name']})...")
            insert_playlist(conn, meta, "editorial")
            tracks = get_playlist_tracks(pid)
            valid_tracks = [t for t in tracks if t.get("id") and t.get("preview")]

            for i, t in enumerate(valid_tracks):
                insert_track(conn, t)
                insert_playlist_track(conn, pid, t["id"], i)

            conn.commit()
            print(f"  -> OK : {len(valid_tracks)} tracks avec preview")
            collected += 1
            time.sleep(0.2)

    print(f"\nCollecte éditoriale terminée : {collected} playlists, {skipped} ignorées.")


collect_editorial()