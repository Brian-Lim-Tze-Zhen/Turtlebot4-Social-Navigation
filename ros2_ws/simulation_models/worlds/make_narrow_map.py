#!/usr/bin/env python3
"""Generate conversation_test_narrow.pgm/.yaml from the SDF wall geometry.

Exact map (no SLAM noise), reproducible. Constants MUST match the walls in
simulation_models/worlds/conversation_test_narrow.sdf.
Cells: 0 = occupied (walls), 254 = free (inside corridor), 205 = unknown.
People are NOT drawn - the static map must not contain them (Fix 2 probes it).
"""
RES = 0.05
ORIGIN_X, ORIGIN_Y = -2.5, -1.5        # lower-left corner of the image
SIZE_X, SIZE_Y = 11.0, 3.0             # m -> x -2.5..8.5, y -1.5..1.5
INNER_Y = 1.00                          # corridor inner faces at +/-INNER_Y
INNER_X0, INNER_X1 = -2.0, 8.0          # end-cap inner faces
THICK = 0.10                            # wall thickness
NAME = "conversation_test_narrow"

W, H = round(SIZE_X / RES), round(SIZE_Y / RES)

def cell_value(x, y):
    in_outer = (INNER_X0 - THICK <= x <= INNER_X1 + THICK) and abs(y) <= INNER_Y + THICK
    in_inner = (INNER_X0 < x < INNER_X1) and abs(y) < INNER_Y
    if in_inner:
        return 254
    if in_outer:
        return 0
    return 205

rows = []
for r in range(H):                      # row 0 = top of image = max y
    y = ORIGIN_Y + (H - 1 - r + 0.5) * RES
    rows.append(bytes(cell_value(ORIGIN_X + (c + 0.5) * RES, y) for c in range(W)))

with open(f"{NAME}.pgm", "wb") as f:
    f.write(f"P5\n{W} {H}\n255\n".encode())
    f.write(b"".join(rows))

with open(f"{NAME}.yaml", "w") as f:
    f.write(f"image: {NAME}.pgm\nmode: trinary\nresolution: {RES:.3f}\n"
            f"origin: [{ORIGIN_X}, {ORIGIN_Y}, 0]\nnegate: 0\n"
            "occupied_thresh: 0.65\nfree_thresh: 0.196\n")
print(f"wrote {NAME}.pgm ({W}x{H}) and {NAME}.yaml")
