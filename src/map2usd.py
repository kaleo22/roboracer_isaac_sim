import math
from pathlib import Path

import yaml
import cv2
import numpy as np
import racetrack
import boundariesinterpolation as bi

from pxr import Usd, UsdGeom, Gf, Sdf
# Wenn im Isaac-Umfeld:
# from pxr import PhysxSchema

# Konfiguration für die Barriers (Luftschläuche)
BARRIER_DIAMETER = 0.3    # m (Breite der "Schläuche")
SEGMENT_LENGTH_MIN = 0.2  # m, minimale Segmentlänge
BARRIER_HEIGHT = 0.2      # m, Höhe der Barriers


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


def convert_map_to_usd(yaml_path: Path, usd_path: Path):
    _, img, res, origin = racetrack.load_map(yaml_path)

    #inner_cnt, outer_cnt = racetrack.extract_contours(img, invert=False)
    inner_cnt, outer_cnt = bi.extract_two_track_edges(img)
    print("Inner contour points:", len(inner_cnt))
    print("Outer contour points:", len(outer_cnt))

    stage, _ = racetrack.create_usd_stage(usd_path)
    racetrack.add_ground_plane(stage, img.shape, res, origin)

    # Innere Begrenzung
    #inner_pts = racetrack.contour_to_world_points(inner_cnt, img.shape, res, origin)
    #racetrack.create_basis_curve(stage, "/World/Track", "InnerBoundary", inner_pts)
    inner_pts = bi.resample_polyline(racetrack.contour_to_world_points(inner_cnt, img.shape, res, origin), target_spacing=0.2)
    bi.build_walls_from_polyline(stage, "/World/Track/InnerBarrier", "Inner", inner_pts, closed=True,
                                half_width=0.15, height=0.25, segment_length=0.2)

    # Äußere Begrenzung
    outer_pts = bi.resample_polyline(racetrack.contour_to_world_points(outer_cnt, img.shape, res, origin), target_spacing=0.2)
    bi.build_walls_from_polyline(stage, "/World/Track/OuterBarrier", "Outer", outer_pts, closed=True,
                                half_width=0.15, height=0.25, segment_length=0.2)
    #racetrack.create_basis_curve(stage, "/World/Track", "OuterBoundary", outer_pts)
    # Barriers hinzufügen
    # racetrack.build_barrier_segments(stage, "/World/Track/InnerBarrier", "Inner", inner_pts,
    #                              segment_step=5, half_width=0.15, height=0.25)
    # racetrack.build_barrier_segments(stage, "/World/Track/OuterBarrier", "Outer", outer_pts,
    #                              segment_step=5, half_width=0.15, height=0.25)
    print("=== STAGE PRIMS ===")
    for prim in stage.Traverse():
        print(prim.GetPath())
    print("=== END ===")

    stage.GetRootLayer().Save()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Benutzung:")
        print("  python map2usd.py path/zur/map.yaml output.usd")
        sys.exit(1)

    yaml_path = Path(sys.argv[1]).resolve()
    usd_path = Path(sys.argv[2]).resolve()
    convert_map_to_usd(yaml_path, usd_path)
