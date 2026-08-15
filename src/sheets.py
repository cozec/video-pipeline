"""Contact-sheet helpers used to visually verify shot boundaries and identity clustering."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(tile: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Draw `text` on a dark strip along the bottom of a tile."""
    h, w = tile.shape[:2]
    strip = max(18, h // 8)
    cv2.rectangle(tile, (0, h - strip), (w, h), (0, 0, 0), -1)
    scale = strip / 30.0
    cv2.putText(tile, text, (4, h - 6), FONT, scale, color, 1, cv2.LINE_AA)
    return tile


def grid(tiles: list[np.ndarray], cols: int = 6, tile_w: int = 240) -> np.ndarray:
    """Assemble equally-sized tiles into a contact sheet."""
    if not tiles:
        return np.zeros((64, 64, 3), np.uint8)
    ar = tiles[0].shape[0] / tiles[0].shape[1]
    tile_h = int(round(tile_w * ar))
    resized = [cv2.resize(t, (tile_w, tile_h)) for t in tiles]
    rows = int(np.ceil(len(resized) / cols))
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
    for i, t in enumerate(resized):
        r, c = divmod(i, cols)
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
    return sheet


def read_frames(video_path, frame_idxs: list[int]) -> dict[int, np.ndarray]:
    """Read specific frame indices from a video in one sequential pass."""
    wanted = sorted(set(int(i) for i in frame_idxs if i >= 0))
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(video_path))
    idx, pos = 0, 0
    last = wanted[-1]
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        while pos < len(wanted) and wanted[pos] == idx:
            out[idx] = frame.copy()
            pos += 1
        idx += 1
    cap.release()
    return out


def boundary_sheet(video_path, shots: list[dict], cols: int = 6) -> np.ndarray:
    """One tile per shot: the first frame of the shot, captioned with shot id and time."""
    idxs = [s["start_frame"] for s in shots]
    frames = read_frames(video_path, idxs)
    tiles = []
    for s in shots:
        f = frames.get(s["start_frame"])
        if f is None:
            continue
        tiles.append(label(f.copy(), f"shot {s['shot_id']}  {s['start_s']:.1f}-{s['end_s']:.1f}s"))
    return grid(tiles, cols=cols)
