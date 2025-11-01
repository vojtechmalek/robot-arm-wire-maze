import ezdxf
import numpy as np
import json

################################
# CONFIG
################################

DXF_FILE = "E_path.dxf"        # your DXF
OUTPUT_FILE = "waypoints_E.json"      # output JSON, working file
RESAMPLE_STEP = 0.001                 # 1 mm in meters
UNIT_SCALE = 0.001                    # DXF is probably mm -> convert to m
TOOL_AXIS = 'X'                       # which tool axis should align with path tangent
WORLD_UP = np.array([0, 0, 1.0])      # reference "up" to stabilize roll
OFFSET_X = -0.005   # meters
OFFSET_Y = -0.005  # meters
OFFSET_Z = 0.000   # meters

################################
# GEOMETRY HANDLING
################################

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n


def extract_ordered_path_points(dxf_doc):
    """
    Collect all path-like geometry (LINE, LWPOLYLINE, POLYLINE, SPLINE),
    then stitch into one ordered path.
    Returns Nx3 in original DXF units.
    """
    msp = dxf_doc.modelspace()

    polylines = []

    # LWPOLYLINE (often 2D, with optional elevation)
    for e in msp.query("LWPOLYLINE"):
        elevation = getattr(e.dxf, "elevation", 0.0)
        pts = []
        for v in e.get_points():
            x, y, *_ = v
            z = elevation
            pts.append([x, y, z])
        if len(pts) > 1:
            polylines.append(np.array(pts))

    # 3D POLYLINE
    for e in msp.query("POLYLINE"):
        pts = []
        for v in e.vertices:
            x, y, z = v.dxf.location.xyz
            pts.append([x, y, z])
        if len(pts) > 1:
            polylines.append(np.array(pts))

    # SPLINE (sample it densely to get approx points)
    for e in msp.query("SPLINE"):
        spline_pts = e.approximate(200)  # list of (x,y,z)
        pts = np.array(spline_pts)
        if len(pts) > 1:
            polylines.append(pts)

    # LINE segments (each will be its own tiny 2-point polyline)
    for e in msp.query("LINE"):
        p0 = np.array(e.dxf.start, dtype=float)
        p1 = np.array(e.dxf.end, dtype=float)
        pts = np.vstack([p0, p1])
        polylines.append(pts)

    if not polylines:
        raise RuntimeError("No usable path entities found.")

    # Now stitch all polylines into one long ordered chain.
    unused = polylines[:]
    current = unused.pop(0)
    ordered = [p for p in current]

    while unused:
        last_point = ordered[-1]
        best_i = None
        best_flip = False
        best_dist = 1e9

        for i, pl in enumerate(unused):
            start_dist = np.linalg.norm(pl[0]  - last_point)
            end_dist   = np.linalg.norm(pl[-1] - last_point)

            # pick whichever end is closer to continue the chain
            if start_dist < best_dist:
                best_dist = start_dist
                best_i = i
                best_flip = False  # keep forward
            if end_dist < best_dist:
                best_dist = end_dist
                best_i = i
                best_flip = True   # reverse this segment

        seg = unused.pop(best_i)
        if best_flip:
            seg = np.flip(seg, axis=0)

        # Avoid duplicating the connection point
        for p in seg[1:]:
            ordered.append(p)

    pts_all = np.array(ordered, dtype=float)  # Nx3, still in DXF units
    return pts_all


def interpolate_polyline(points, step_m):
    """
    Resample polyline to roughly fixed arc-length spacing (meters).
    points: Nx3 in meters
    """
    pts = np.array(points)
    segvecs = pts[1:] - pts[:-1]
    seglens = np.linalg.norm(segvecs, axis=1)
    cumdist = np.concatenate(([0.0], np.cumsum(seglens)))

    total_len = cumdist[-1]
    num_samples = int(np.floor(total_len / step_m)) + 1
    sample_ds = np.linspace(0.0, total_len, num_samples)

    resampled = []
    for d in sample_ds:
        idx = np.searchsorted(cumdist, d) - 1
        if idx < 0:
            idx = 0
        if idx >= len(segvecs):
            idx = len(segvecs) - 1

        d0 = cumdist[idx]
        seglen = seglens[idx] if seglens[idx] > 1e-12 else 1e-12
        t = (d - d0) / seglen
        p = pts[idx] + t * segvecs[idx]
        resampled.append(p)

    return np.array(resampled)


def compute_tool_frame_from_tangent(tangent, world_up, tool_axis='Z'):
    """
    Build rotation matrix so chosen tool axis aligns with path tangent.
    - If TOOL_AXIS == 'Z', we align tool +Z with tangent.
    """
    z_axis = normalize(tangent)

    # find x_axis close to world_up projected perpendicular to z_axis
    proj = world_up - np.dot(world_up, z_axis) * z_axis
    if np.linalg.norm(proj) < 1e-6:
        proj = np.array([1.0, 0.0, 0.0]) - np.dot([1.0,0.0,0.0], z_axis) * z_axis
    x_axis = normalize(proj)

    # y = z × x
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)

    # re-orthogonalize x = y × z
    x_axis = np.cross(y_axis, z_axis)
    x_axis = normalize(x_axis)

    if tool_axis.upper() == 'Z':
        R = np.column_stack((x_axis, y_axis, z_axis))  # columns are X, Y, Z of tool frame
    elif tool_axis.upper() == 'X':
        # if you want +X along tangent instead:
        X = z_axis
        Y = x_axis
        Z = np.cross(X, Y); Z = normalize(Z)
        Y = np.cross(Z, X); Y = normalize(Y)
        R = np.column_stack((X, Y, Z))
    else:
        raise ValueError("Unsupported TOOL_AXIS")

    return R


def rotm_to_rpy_zyx(R):
    """
    Convert rotation matrix to ZYX Euler angles:
    yaw (Z), pitch (Y), roll (X)
    returns [roll, pitch, yaw] in radians
    """
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    yaw   = np.arctan2(r21, r11)
    pitch = np.arcsin(-r31)
    roll  = np.arctan2(r32, r33)

    return [float(roll), float(pitch), float(yaw)]


def main():
    # 1. read the DXF
    dxf_doc = ezdxf.readfile(DXF_FILE)

    # 2. extract and stitch all segments into one long polyline
    raw_pts_mm = extract_ordered_path_points(dxf_doc)  # Nx3 in DXF units (likely mm)

    # 3. convert to meters
    raw_pts_m = raw_pts_mm * UNIT_SCALE  # Nx3 in meters

    # 4. resample ~every 1 mm
    path_pts = interpolate_polyline(raw_pts_m, RESAMPLE_STEP)  # Mx3

    # reverse the order so we start at the other end
    path_pts = np.flip(path_pts, axis=0)

    # 5. build waypoints with orientation
    waypoints = []
    for i in range(len(path_pts)):
        # pick two sample indices for tangent estimation:
        i_a = max(i - 2, 0)
        i_b = min(i + 2, len(path_pts)-1)

        p_a = path_pts[i_a]
        p_b = path_pts[i_b]

        tangent = p_b - p_a
        if np.linalg.norm(tangent) < 1e-12 and i < len(path_pts)-1:
            # fallback: use immediate next if degenerate
            tangent = path_pts[i+1] - path_pts[i]

        p_curr = path_pts[i]

        R_tool = compute_tool_frame_from_tangent(
            tangent,
            WORLD_UP,
            tool_axis=TOOL_AXIS
        )

        R_world_to_tool = R_tool.T

        rpy = rotm_to_rpy_zyx(R_world_to_tool)  # radians

        # apply XY(Z) offset
        p_curr_offset = np.array([
            p_curr[0] + OFFSET_X,
            p_curr[1] + OFFSET_Y,
            p_curr[2] + OFFSET_Z,
        ])

        waypoints.append({
            "pos": p_curr_offset.tolist(),
            "rpy": rpy,
            "idx": int(i)
        })

    # last point, reuse last orientation
    p_last = path_pts[-1]
    p_last_offset = np.array([
        p_last[0] + OFFSET_X,
        p_last[1] + OFFSET_Y,
        p_last[2] + OFFSET_Z,
    ])
    waypoints.append({
        "pos": p_last_offset.tolist(),
        "rpy": waypoints[-1]["rpy"],
        "idx": len(path_pts)-1
    })


    # 6. save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)

    print(f"Saved {len(waypoints)} waypoints to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
