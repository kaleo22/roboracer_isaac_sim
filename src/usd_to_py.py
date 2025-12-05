import sys
from pxr import Usd, Sdf

in_usd = sys.argv[1]
out_py = sys.argv[2] if len(sys.argv) > 2 else 'recreate_usd.py'

stage = Usd.Stage.Open(in_usd)
lines = [
    "from pxr import Usd, Sdf",
    "stage = Usd.Stage.CreateNew('out.usd')",
    ""
]

for prim in stage.Traverse():
    path = str(prim.GetPath())
    typ = prim.GetTypeName() or "Xform"
    lines.append(f"prim = stage.DefinePrim('{path}', '{typ}')")
    for attr in prim.GetAttributes():
        name = attr.GetBaseName()
        val = attr.Get()
        if val is None:
            continue
        # simple handling for common python literals
        lines.append(f"attr = prim.CreateAttribute('{name}', Sdf.ValueTypeNames.String)")
        lines.append(f"attr.Set({repr(val)})")
    lines.append("")

lines.append("stage.GetRootLayer().Save()")

with open(out_py, 'w') as f:
    f.write('\\n'.join(lines))
print('Wrote', out_py)