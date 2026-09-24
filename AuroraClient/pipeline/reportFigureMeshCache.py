"""
Per-figure mesh + heatmap source cache for the scientific report editor.

Stored under ``extracted/figures_cache/{figure_id}.npz`` with a small
``.meta.json`` sidecar describing scan/edit/metric/quality so stale entries
are ignored. This is separate from ``.figures.json`` PNG/SVG document cache.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

import numpy as np


def _figures_cache_dir(directory: str) -> str:
    return os.path.join(directory, 'extracted', 'figures_cache')


def sanitize_figure_id(figure_id: str) -> str:
    if not isinstance(figure_id, str) or not figure_id.strip():
        return ''
    safe = os.path.basename(figure_id.strip()).replace('\\', '_').replace('/', '_')
    safe = safe.replace('\0', '')
    safe = ''.join(ch for ch in safe if ch.isalnum() or ch in (' ', '-', '_', '.')).strip()
    return safe.replace(' ', '_')


def figure_mesh_cache_paths(directory: str, figure_id: str) -> tuple[str, str]:
    stem = sanitize_figure_id(figure_id)
    if not stem:
        raise ValueError('invalid figure_id')
    base = os.path.join(_figures_cache_dir(directory), stem)
    return f'{base}.npz', f'{base}.meta.json'


def figure_mesh_glb_path(directory: str, figure_id: str) -> str:
    stem = sanitize_figure_id(figure_id)
    if not stem:
        raise ValueError('invalid figure_id')
    return os.path.join(_figures_cache_dir(directory), f'{stem}.glb')


def _params_match(cached: dict, expected: dict) -> bool:
    keys = (
        'scan',
        'edit',
        'metric',
        'quality',
        'blur',
        'blur_sigma',
        'full_resolution',
    )
    for key in keys:
        if key not in expected:
            continue
        if str(cached.get(key, '')) != str(expected.get(key, '')):
            return False
    return True


def load_figure_mesh_cache(
    directory: str,
    figure_id: str,
    expected_params: Optional[dict] = None,
) -> Optional[dict[str, Any]]:
    npz_path, meta_path = figure_mesh_cache_paths(directory, figure_id)
    if not os.path.isfile(npz_path) or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as mf:
            meta = json.load(mf)
        if expected_params and not _params_match(meta, expected_params):
            return None
        data = np.load(npz_path)
        vertices = np.asarray(data['vertices'], dtype=np.float64)
        faces = np.asarray(data['faces'], dtype=np.uint32)
        distances = np.asarray(data['distances'], dtype=np.float32)
        n_vertices = int(meta.get('n_vertices') or len(vertices))
        if len(distances) != n_vertices:
            return None
        payload: dict[str, Any] = {
            'meta': meta,
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'distances': distances.tolist(),
        }
        glb_path = figure_mesh_glb_path(directory, figure_id)
        if os.path.isfile(glb_path):
            try:
                with open(glb_path, 'rb') as gf:
                    payload['gltf'] = base64.b64encode(gf.read()).decode('utf-8')
            except OSError:
                pass
        return payload
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save_figure_mesh_cache(
    directory: str,
    figure_id: str,
    *,
    vertices,
    faces,
    distances,
    meta: dict,
    gltf: Optional[str] = None,
) -> str:
    npz_path, meta_path = figure_mesh_cache_paths(directory, figure_id)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    verts = np.asarray(vertices, dtype=np.float64)
    fac = np.asarray(faces, dtype=np.uint32)
    dist = np.asarray(distances, dtype=np.float32)
    if len(dist) != len(verts):
        raise ValueError('distances length must match vertex count')
    meta = dict(meta or {})
    meta['figure_id'] = sanitize_figure_id(figure_id)
    meta['n_vertices'] = int(len(verts))
    np.savez(npz_path, vertices=verts, faces=fac, distances=dist)
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)
    if isinstance(gltf, str) and gltf.strip():
        glb_path = figure_mesh_glb_path(directory, figure_id)
        try:
            with open(glb_path, 'wb') as gf:
                gf.write(base64.b64decode(gltf))
        except (OSError, ValueError):
            pass
    return npz_path


def delete_figure_mesh_cache(directory: str, figure_id: str) -> bool:
    npz_path, meta_path = figure_mesh_cache_paths(directory, figure_id)
    glb_path = figure_mesh_glb_path(directory, figure_id)
    deleted = False
    for path in (npz_path, meta_path, glb_path):
        if os.path.isfile(path):
            try:
                os.remove(path)
                deleted = True
            except OSError:
                pass
    return deleted
