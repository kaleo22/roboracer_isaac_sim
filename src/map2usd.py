#!/usr/bin/env python3

import math
from pathlib import Path

import yaml
import cv2
import numpy as np

from pxr import Usd, UsdGeom, Gf, Sdf
# Wenn im Isaac-Umfeld:
# from pxr import PhysxSchema

# Konfiguration für die Barriers (Luftschläuche)
BARRIER_HEIGHT = 0.4      # m
BARRIER_DIAMETER = 0.3    # m (Breite der "Schläuche")
SEGMENT_LENGTH_MIN = 0.2  # m, minimale Segmentlänge


def load_map(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_rel = cfg["image"]
    resolution = float(cfg["resolution"])
    origin = cfg.get("origin", [0.0, 0.0, 0.0])

    img_path = (yaml_path.parent / image_rel).resolve()
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Konnte Bild nicht laden: {img_path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cfg, img, resolution, origin


def extract_contours(img: np.ndarray, invert: bool = False):
    # Binarisieren: anpassen je nach Map (schwarz = Wand, weiß = frei usw.)
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if invert:
        thresh = 255 - thresh

    # Konturen der "Wände"
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def pixel_to_world(pt, img_shape, res, origin):
    """pt: (x_px, y_px), img_shape: (h, w)"""
    h, w = img_shape[:2]
    col, row = pt

    x = origin[0] + col * res
    # Bild-Y nach oben invertieren:
    y = origin[1] + (h - row) * res
    z = 0.0
    return Gf.Vec3f(x, y, z)


def create_stage(usd_path: Path):
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    return stage, world.GetPrim()


def add_ground_plane(stage, parent_prim, img_shape, res, origin):
    h, w = img_shape[:2]
    width_m = w * res
    height_m = h * res

    ground = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Ground"))
    # Wir nehmen den Mittelpunkt der Map als Position
    cx = origin[0] + width_m / 2.0
    cy = origin[1] + height_m / 2.0
    cz = -0.01  # etwas unter 0
    ground.AddTranslateOp().Set(Gf.Vec3f(cx, cy, cz))
    # Größe: X,Y = Kartenausdehnung, Z sehr dünn
    ground.AddScaleOp().Set(Gf.Vec3f(width_m, height_m, 0.02))

    # Hier könntest du ein PhysX Static Collider draus machen, falls nötig
    # collision_api = PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
    # ...

    return ground


def add_barrier_segment(stage, parent_path: Sdf.Path,
                        p0: Gf.Vec3f, p1: Gf.Vec3f,
                        index: int):
    # Mittelpunkt & Länge
    mid = (p0 + p1) * 0.5
    dir_vec = p1 - p0
    length = dir_vec.GetLength()
    if length < SEGMENT_LENGTH_MIN:
        return None

    # Normieren für Rotation
    dir_norm = dir_vec / length

    # Wir nehmen an: USD-Cube liegt in Y-Richtung lang und Z nach oben.
    # Lokale Länge: scale.y = length, scale.x = BARRIER_DIAMETER, scale.z = Höhe
    prim_path = parent_path.AppendChild(f"Barrier_{index}")
    cube = UsdGeom.Cube.Define(stage, prim_path)

    # Translation
    cube.AddTranslateOp().Set(mid)

    # Rotation um Z, damit die Y-Achse dem Richtungsvektor folgt
    angle = math.atan2(dir_norm[1], dir_norm[0])  # Richtung in XY
    rot_deg = math.degrees(angle)
    cube.AddRotateZYXOp().Set(Gf.Vec3f(0.0, 0.0, rot_deg))

    # Scale (X=Breite, Y=Länge, Z=Höhe)
    cube.AddScaleOp().Set(Gf.Vec3f(BARRIER_DIAMETER,
                                   length,
                                   BARRIER_HEIGHT))

    # PhysX:
    # prim = cube.GetPrim()
    # body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    # collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    # collision_api.CreateShapeTypeAttr("convexHull")

    return cube


def add_barriers_from_contours(stage, img, contours, res, origin):
    barriers_root = UsdGeom.Xform.Define(stage, "/World/Barriers")
    root_path = barriers_root.GetPath()
    img_shape = img.shape

    seg_index = 0
    for contour in contours:
        # contour: Nx1x2 → flatten
        pts = contour.reshape(-1, 2)

        # jeden Punkt in Weltkoordinaten umrechnen
        world_pts = [
            pixel_to_world((int(x), int(y)), img_shape, res, origin)
            for (x, y) in pts
        ]

        # Segmente entlang der Kontur erzeugen (Loop schließen)
        for i in range(len(world_pts)):
            p0 = world_pts[i]
            p1 = world_pts[(i + 1) % len(world_pts)]  # Loop
            added = add_barrier_segment(stage, root_path, p0, p1, seg_index)
            if added is not None:
                seg_index += 1


def convert_map_to_usd(yaml_path: Path, usd_path: Path):
    cfg, img, res, origin = load_map(yaml_path)
    contours = extract_contours(img, invert=False)

    stage, world_prim = create_stage(usd_path)
    add_ground_plane(stage, world_prim, img.shape, res, origin)
    add_barriers_from_contours(stage, img, contours, res, origin)

    stage.GetRootLayer().Save()
    print(f"USD gespeichert unter: {usd_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Benutzung:")
        print("  python map2usd.py path/zur/map.yaml output.usd")
        sys.exit(1)

    yaml_path = Path(sys.argv[1]).resolve()
    usd_path = Path(sys.argv[2]).resolve()
    convert_map_to_usd(yaml_path, usd_path)
