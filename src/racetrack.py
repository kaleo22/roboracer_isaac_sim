from pathlib import Path
import yaml
import cv2
import numpy as np
import math
from pxr import Usd, UsdGeom, Gf, Sdf
from pxr import UsdShade, UsdPhysics

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
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) < 2:
        raise RuntimeError(f"Erwarte mindestens 2 Konturen, habe aber nur {len(contours)} gefunden.")

    if hierarchy is None or len(contours) == 0:
        raise RuntimeError("Keine Konturen gefunden.")

    hierarchy = hierarchy[0]  # shape (N, 4)

    outer_candidates = []
    inner_candidates = []

    for i, h in enumerate(hierarchy):
        parent = h[3]
        if parent == -1:
            # Top-Level – potenzieller äußerer Rand
            outer_candidates.append(contours[i])
        else:
            # Hat einen Parent – Loch → innerer Rand
            inner_candidates.append(contours[i])

    if not outer_candidates:
        raise RuntimeError("Keine äußeren Konturen gefunden.")
    if not inner_candidates:
        # Fallback: wenn es nur einen Blob ohne Loch gibt,
        # nimm diese eine Kontur doppelt.
        print("Warnung: Keine inneren Konturen gefunden – verwende äußere Kontur als inner & outer.")
        main = max(outer_candidates, key=cv2.contourArea)
        return main, main

    # Wähle jeweils die größte Kontur in jeder Gruppe
    outer = max(outer_candidates, key=cv2.contourArea)
    inner = max(inner_candidates, key=cv2.contourArea)
    epsilon = 3.0  # in Pixeln, bei Bedarf erhöhen
    inner_simpl = cv2.approxPolyDP(inner, epsilon, True)
    outer_simpl = cv2.approxPolyDP(outer, epsilon, True)

    return inner_simpl, outer_simpl

def create_basis_curve(stage, parent_path: str, name: str, world_pts):
    curve_path = f"{parent_path}/{name}"
    curve = UsdGeom.BasisCurves.Define(stage, curve_path)

    curve.CreatePointsAttr(world_pts)
    curve.CreateCurveVertexCountsAttr([len(world_pts)])
    curve.CreateTypeAttr(UsdGeom.Tokens.cubic)
    curve.CreateBasisAttr(UsdGeom.Tokens.bspline)
    curve.CreateWrapAttr(UsdGeom.Tokens.periodic)  # wenn Strecke geschlossen

    widths = [0.1] * len(world_pts)
    curve.CreateWidthsAttr(widths)

    return curve

def build_barrier_segments(stage,
                           parent_path: str,
                           name_prefix: str,
                           world_pts,
                           segment_step: int = 5,
                           half_width: float = 0.15,
                           height: float = 0.25):
    """
    Baut viele schmale Box-Segmente entlang der Polyline world_pts.
    segment_step: wie viele Punkte wir pro Segment überspringen (Größenordnung der Segmentlänge)
    half_width: halbe Breite des Schlauchs (seitlich)
    height: Höhe der Begrenzung
    """
    parent_prim = UsdGeom.Xform.Define(stage, parent_path).GetPrim()

    num_pts = len(world_pts)
    if num_pts < 2:
        print("Zu wenige Punkte für Begrenzung.")
        return

    for i in range(0, num_pts - segment_step, segment_step):
        p0 = world_pts[i]
        p1 = world_pts[i + segment_step]

        # Richtung im XY
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        if seg_len < 1e-3:
            continue

        # Mittelpunkt des Segments
        mid = Gf.Vec3f(
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5 + height * 0.5  # Mitte in Z
        )

        # Rotation um Z, damit Cube in Fahrtrichtung ausgerichtet ist
        yaw_rad = math.atan2(dy, dx)
        yaw_deg = math.degrees(yaw_rad)

        seg_path = f"{parent_path}/{name_prefix}_{i}"
        xf = UsdGeom.Xform.Define(stage, seg_path)
        xf_prim = xf.GetPrim()
        xf_api = UsdGeom.Xformable(xf_prim)

        # erst Rotation, dann Translation (oder umgekehrt, je nach Geschmack)
        xf_api.AddRotateZOp().Set(yaw_deg)
        xf_api.AddTranslateOp().Set(mid)

        # Box-Mesh als Kind
        box_path = seg_path + "/Mesh"
        box = UsdGeom.Cube.Define(stage, box_path)

        # Größe anpassen: X-Ausdehnung = Segmentlänge, Y = 2*half_width
        # UsdGeom.Cube hat uniform size → wir skalieren mit Xform
        box.CreateSizeAttr(1.0)  # Basisgröße 1m

        box_xf_api = UsdGeom.Xformable(box.GetPrim())
        box_xf_api.AddScaleOp().Set(Gf.Vec3f(seg_len, 2.0 * half_width, height))

        # Physics: Kollisionsobjekt + Material direkt am Cube
        try:
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            mat_api = UsdPhysics.MaterialAPI.Apply(box.GetPrim())
            mat_api.CreateStaticFrictionAttr(1.0)
            mat_api.CreateDynamicFrictionAttr(0.9)
            mat_api.CreateRestitutionAttr(0.1)  # leicht elastisch
        except Exception as e:
            print("Warnung (Barrier Physics):", e)

def bind_material(prim, mtl):
    # Setup a MaterialBindingAPI on the mesh prim
    bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
    # Use the constructed binding API to bind the material
    bindingAPI.Bind(mtl)

def pixel_to_world(pt, img_shape, res, origin):
    """pt: (x_px, y_px), img_shape: (h, w)"""
    h, w = img_shape[:2]
    col, row = pt

    x = origin[0] + col * res
    # Bild-Y nach oben invertieren:
    y = origin[1] + (h - row) * res
    z = origin[2]  # Bodenhöhe anpassen, falls nötig
    return (x, y, z)

def contour_to_world_points(contour, img_shape, res, origin, z_offset=0.01):
    """
    Wandelt eine OpenCV-Kontur in eine Liste von Gf.Vec3f-Weltpunkten um.
    contour: np.ndarray mit Shape (N, 1, 2), wie von cv2.findContours
    img_shape: Shape des Originalbildes (H, W)
    res: Auflösung [m/Pixel] aus der YAML
    origin: [x0, y0, z0] aus der YAML
    z_offset: kleiner Offset, damit die Punkte nicht genau im Boden liegen
    """
    # Konturpunkte extrahieren (x_px, y_px)
    pts_px = contour[:, 0, :]  # -> (N, 2)

    world_pts = []
    for col, row in pts_px:
        x, y, z = pixel_to_world((col, row), img_shape, res, origin)
        world_pts.append(Gf.Vec3f(x, y, z + z_offset))

    return world_pts

def create_usd_stage(usda_path: Path):
    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    # Physics Scene
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    return stage, world

def add_ground_plane(stage, img_shape, res, origin, texture_path: str = None):

    h, w = img_shape[:2]
    width_m = w * res
    depth_m = h * res
    ground_xf = UsdGeom.Xform.Define(stage, "/World/GroundPlane")
    ground_prim = ground_xf.GetPrim()
    # Eckpunkte (gegen den Uhrzeigersinn)
    points = [
        Gf.Vec3f(origin[0], origin[1], origin[2]),
        Gf.Vec3f(origin[0] + width_m, origin[1], origin[2]),
        Gf.Vec3f(origin[0] + width_m, origin[1] + depth_m, origin[2]),
        Gf.Vec3f(origin[0], origin[1] + depth_m, origin[2]),
    ]

    ground_mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
    ground_mesh.CreatePointsAttr(points)
    ground_mesh.CreateFaceVertexCountsAttr([4])
    ground_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground_mesh.CreateExtentAttr([
        Gf.Vec3f(origin[0], origin[1], origin[2]),
        Gf.Vec3f(origin[0] + width_m, origin[1] + depth_m, origin[2])
    ])
    ground_mesh.CreateDoubleSidedAttr(True)
    # Normale (nach oben, Z-Up)
    normals = [Gf.Vec3f(0.0, 0.0, 1.0)] * 4
    ground_mesh.CreateNormalsAttr(normals)

    # UVs (st) in [0,1] Raum für eine Textur
    primvars_api = UsdGeom.PrimvarsAPI(ground_mesh.GetPrim())
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.vertex)
    st.Set([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ])

    # Einfaches Material: UsdPreviewSurface (asphalt-ähnlich)
    mat_path = "/World/Looks/AsphaltMat"
    mat = UsdShade.Material.Define(stage, mat_path)

    shader = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Default-Farb-/Rauheitswerte
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.12, 0.12))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    # Optional: wenn texture_path gesetzt ist, benutze UsdUVTexture -> primvar reader -> connect to diffuse
    if texture_path:
        uv_tex = UsdShade.Shader.Define(stage, mat_path + "/AsphaltTexture")
        uv_tex.CreateIdAttr("UsdUVTexture")
        uv_tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        # Primvar reader für "st"
        primvar_reader = UsdShade.Shader.Define(stage, mat_path + "/PrimvarReader_st")
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        # Connect primvar reader -> uv_tex.st
        uv_tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(primvar_reader, "result")
        # Connect texture rgb -> previewSurface.diffuseColor (überschreibt obiges diffuseColor)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(uv_tex, "rgb")

    # Material binden
    bind_material(ground_mesh.GetPrim(), mat)


    # Physics: eine kollisionsfähige Fläche erstellen / markieren.
    UsdPhysics.CollisionAPI.Apply(ground_mesh.GetPrim())
    mat_api = UsdPhysics.MaterialAPI.Apply(ground_mesh.GetPrim())
    mat_api.CreateStaticFrictionAttr(1.0)   # Haftreibung
    mat_api.CreateDynamicFrictionAttr(0.9)  # Gleitreibung
    mat_api.CreateRestitutionAttr(0.0)      # "Bounciness"

    return ground_xf
