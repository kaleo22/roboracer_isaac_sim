import cv2
import numpy as np

def extract_two_track_edges(img):
    # Immer als Graustufen interpretieren
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # OTSU + INVERT:
    #   - Hintergrund -> schwarz
    #   - Streckenlinien -> weiß
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Hierarchie holen, damit wir zwischen äußeren Rändern und "Löchern" unterscheiden können
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None or len(contours) == 0:
        raise RuntimeError("Keine Konturen gefunden.")

    hierarchy = hierarchy[0]  # Shape (N, 4) : [next, prev, child, parent]

    # Wir wollen die "inneren" Konturen (die Löcher in den weißen Flächen),
    # also alle mit parent != -1
    track_contours = [contours[i] for i, h in enumerate(hierarchy) if h[3] != -1]

    if len(track_contours) != 2:
        raise RuntimeError(
            f"Erwarte 2 Track-Konturen, gefunden: {len(track_contours)} "
            f"(gesamt {len(contours)} Konturen)."
        )

    c0, c1 = track_contours

    # Vereinfachen, damit du nicht zehntausende Punkte hast
    epsilon = 2.0  # Pixel-Toleranz, ggf. anpassen
    c0_simpl = cv2.approxPolyDP(c0, epsilon, True)
    c1_simpl = cv2.approxPolyDP(c1, epsilon, True)

    # Nach Fläche sortieren: kleinere als "inner", größere als "outer"
    if cv2.contourArea(c0_simpl) < cv2.contourArea(c1_simpl):
        inner = c0_simpl
        outer = c1_simpl
    else:
        inner = c1_simpl
        outer = c0_simpl

    return inner, outer

def resample_polyline(points, target_spacing=0.2):
    # points: list of Vec3f oder Nx3 np.array
    pts = np.array([[p[0], p[1], p[2]] for p in points])
    diffs = np.diff(pts, axis=0)
    d = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(d)])

    total_len = s[-1]
    num_samples = max(int(total_len / target_spacing), 3)
    s_new = np.linspace(0, total_len, num_samples)

    pts_new = np.zeros((num_samples, 3))
    for i in range(3):
        pts_new[:, i] = np.interp(s_new, s, pts[:, i])

    return pts_new

import math
from pxr import UsdGeom, UsdPhysics, Gf

def build_walls_from_polyline(stage,
                              parent_path,
                              name_prefix,
                              world_pts,
                                closed=False,
                              half_width=0.15,
                              height=0.25,
                              segment_length=0.2):

    UsdGeom.Xform.Define(stage, parent_path)

    num = len(world_pts)
    for i in range(num if closed else num - 1):
        j = (i + 1) % num if closed else i + 1
        p0 = world_pts[i]
        p1 = world_pts[j]

        # Richtung
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        if seg_len < 1e-4:
            continue

        yaw = math.degrees(math.atan2(dy, dx))

        # Mittelpunkt
        mid = Gf.Vec3f(
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5 + height * 0.5,
        )

        # Transform-Hierarchie
        seg_root = UsdGeom.Xform.Define(stage, f"{parent_path}/{name_prefix}_{i}")
        seg_xf = UsdGeom.Xformable(seg_root.GetPrim())
        seg_xf.AddTranslateOp().Set(mid)
        seg_xf.AddRotateZOp().Set(yaw)

        cube = UsdGeom.Cube.Define(stage, f"{parent_path}/{name_prefix}_{i}/Mesh")
        cube.CreateSizeAttr(1.0)
        cube_xform = UsdGeom.Xformable(cube.GetPrim())
        cube_xform.AddScaleOp().Set(Gf.Vec3f(seg_len, 2 * half_width, height))

        # Physics
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())