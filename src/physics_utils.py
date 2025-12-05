"""Utilities to inspect and set friction attributes in USD files.

Functions try to use the USD Python bindings (`pxr`) when available.
If `pxr` is not installed, the code will try to use the `usdcat` tool
from OpenUSD to convert binary USD to ASCII for inspection.

Usage examples:
  from physics_utils import find_usd_files, list_physics_materials
  for p in find_usd_files('.'):
      print(p, list_physics_materials(p))

Note: setting attributes requires the USD Python API (`pxr`). Fallbacks
for editing non-ascii USD are limited; prefer installing the `pxr`/USD
Python package available with `openusd` in conda.
"""
from __future__ import annotations

import os
import re
import math
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


def find_usd_files(root: str = '.') -> List[str]:
    """Return a list of .usd files under `root` (recurses)."""
    usd_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.usd'):
                usd_files.append(os.path.join(dirpath, fn))
    return sorted(usd_files)


def _has_pxr() -> bool:
    try:
        import pxr  # type: ignore
        return True
    except Exception:
        return False


def list_physics_materials(usd_path: str) -> List[Dict[str, object]]:
    """List prims/attributes that look like physics material friction data.

    Returns a list of dicts with keys: `prim_path`, `attribute`, `value`.
    Uses `pxr` if available, otherwise falls back to running `usdcat`
    (if present) and text-parsing the USDA output.
    """
    if _has_pxr():
        return _list_with_pxr(usd_path)
    usdcat = shutil.which('usdcat')
    if usdcat:
        return _list_with_usdcat(usd_path, usdcat)
    raise RuntimeError(
        'No USD Python API available and `usdcat` not found. Install OpenUSD or enable `pxr`.'
    )


def _list_with_pxr(usd_path: str) -> List[Dict[str, object]]:
    from pxr import Usd, Sdf

    out = []
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return out
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            name = attr.GetBaseName()
            if 'friction' in name.lower():
                try:
                    val = attr.Get()
                except Exception:
                    val = None
                out.append({'prim_path': str(prim.GetPath()), 'attribute': name, 'value': val})
    return out


def _list_with_usdcat(usd_path: str, usdcat: str) -> List[Dict[str, object]]:
    """Run `usdcat` and text-parse the output for friction attributes."""
    cmd = [usdcat, usd_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"usdcat failed: {proc.stderr}")
    text = proc.stdout
    # Find lines like: something:staticFriction = 0.5
    pattern = re.compile(r"(?P<attr>[\w:\.]*friction[\w\.:]*)\s*=\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
    results: List[Dict[str, object]] = []
    for m in pattern.finditer(text):
        results.append({'prim_path': None, 'attribute': m.group('attr'), 'value': float(m.group('val'))})
    return results


def set_friction(usd_path: str, prim_path: str, static: Optional[float] = None, dynamic: Optional[float] = None) -> None:
    """Set `staticFriction` and/or `dynamicFriction` on `prim_path` in `usd_path`.

    This requires the USD Python API (`pxr`). If `pxr` is not available the
    function will raise an error with guidance.
    """
    if not _has_pxr():
        raise RuntimeError('Setting attributes requires the USD Python API (pxr).')
    from pxr import Usd, Sdf

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f'Unable to open USD: {usd_path}')
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f'Prim not found: {prim_path}')
    if static is not None:
        attr = prim.GetAttribute('staticFriction')
        if not attr:
            attr = prim.CreateAttribute('staticFriction', Sdf.ValueTypeNames.Double)
        attr.Set(float(static))
    if dynamic is not None:
        attr = prim.GetAttribute('dynamicFriction')
        if not attr:
            attr = prim.CreateAttribute('dynamicFriction', Sdf.ValueTypeNames.Double)
        attr.Set(float(dynamic))
    # Save layer
    stage.GetRootLayer().Save()


def compute_mu_from_slope(theta_degrees: float) -> float:
    """Compute static friction coefficient µ ≈ tan(theta) from slope angle in degrees."""
    theta = math.radians(theta_degrees)
    return math.tan(theta)


def compute_mu_from_motion(theta_degrees: float, acceleration: float, g: float = 9.81) -> float:
    """Estimate dynamic friction µ_k from measured acceleration down a slope.

    Formula derived from: m*a = m*g*sin(theta) - µ_k*m*g*cos(theta)
    => µ_k = (g*sin(theta) - a) / (g*cos(theta))
    """
    theta = math.radians(theta_degrees)
    return (g * math.sin(theta) - acceleration) / (g * math.cos(theta))


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Quick USD friction inspector')
    p.add_argument('usd', help='USD file to inspect')
    args = p.parse_args()
    print('Searching for friction-like attributes in', args.usd)
    try:
        found = list_physics_materials(args.usd)
    except Exception as e:
        print('Error:', e)
        raise
    if not found:
        print('No friction attributes found.')
    else:
        for f in found:
            print(f)
