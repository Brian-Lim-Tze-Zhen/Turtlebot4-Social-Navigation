#!/usr/bin/env python3
"""Generate corridor_headon_aligned.pgm/.yaml from the SDF wall geometry.

Exact map (no SLAM noise), reproducible, and in the GAZEBO WORLD FRAME.
The SLAM-recorded maps/corridor_headon.yaml is shifted ~3 m in x from the
world frame (it spans x -1.46..13.44, the corridor spans x -4.5..10.5), so
/person_ground_truth and /sim_ground_truth_pose (both world frame) do not
line up with it. Constants MUST match the walls in
simulation_models/worlds/corridor_headon.sdf.
Cells: 0 = occupied (walls), 254 = free (inside corridor), 205 = unknown.
People are NOT drawn - the static map must not contain them.

Run from ros2_ws/maps:
  python3 ../simulation_models/worlds/make_corridor_headon_map.py
"""
RES = 0.05
ORIGIN_X, ORIGIN_Y = -5.0, -1.75       # lower-left corner of the image
SIZE_X, SIZE_Y = 16.0, 3.5             # m -> x -5.0..11.0, y -1.75..1.75
INNER_Y = 1.25                          # walls centred at +/-1.30, 0.1 thick
INNER_X0, INNER_X1 = -4.45, 10.45       # end caps centred at -4.5 and 10.5
THICK = 0.10                            # wall thickness
NAME = "corridor_headon_aligned"

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
