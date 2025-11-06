import ezdxf
import numpy as np
import json
import utils

################################
# CONFIG
################################

DXF_FILE = "dxf:paths/E_path.dxf"        # your DXF
OUTPUT_FILE = "waypoint_paths/waypoints_M.json"      # output JSON, working file
RESAMPLE_STEP = 0.001                 # 1 mm in meters
UNIT_SCALE = 0.001                    # DXF usualy mm -> convert to m
TOOL_AXIS = 'X'                       # which tool axis should align with path tangent
WORLD_UP = np.array([0, 0, 1.0])      # reference "up" to stabilize roll

# offset because model path can be on models edge
OFFSET_X = -0.005   # meters
OFFSET_Y = -0.005  # meters
OFFSET_Z = 0.000   # meters


################################
# DXF GEOMETRY HANDLING
################################


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


################################
# WAYPOINTS GENERATION
################################


# ---------- initial frame from world_up ----------
def initial_frame_from_up(tangent, world_up, tool_axis='X'):
    X = utils.normalize(tangent)  # we align tool +X with the path
    proj = world_up - np.dot(world_up, X) * X
    if np.linalg.norm(proj) < 1e-6:
        # fallback if world_up ~ X
        alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(alt, X)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        proj = alt - np.dot(alt, X) * X
    Y = utils.normalize(proj)
    Z = utils.normalize(np.cross(X, Y))
    Y = utils.normalize(np.cross(Z, X))
    return np.column_stack((X, Y, Z))  # columns = tool axes in WORLD


# ---------- minimal realignment (parallel transport) ----------
def transport_frame(prev_R, new_tangent):
    """
    Rotate the whole previous frame by the *smallest* rotation so that
    its +X matches new_tangent. Columns of prev_R are tool [X Y Z] in world.
    """
    X_old = prev_R[:, 0]
    X_new = utils.normalize(new_tangent)

    v = np.cross(X_old, X_new)
    s = np.linalg.norm(v)
    c = np.dot(X_old, X_new)

    if s < 1e-10:
        if c > 0.0:
            R_align = np.eye(3)  # unchanged
        else:
            # 180° flip about any axis ⟂ X_old; pick a stable one
            tmp = np.array([1.0, 0.0, 0.0]) if abs(X_old[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = utils.normalize(np.cross(X_old, tmp))
            R_align = utils.rodrigues(axis, np.pi)
    else:
        axis = v / s
        theta = np.arctan2(s, c)
        R_align = utils.rodrigues(axis, theta)

    R_new = R_align @ prev_R

    # re-orthonormalize
    X = utils.normalize(R_new[:, 0])
    Y = utils.normalize(R_new[:, 1] - np.dot(R_new[:, 1], X) * X)
    Z = utils.normalize(np.cross(X, Y))
    Y = utils.normalize(np.cross(Z, X))
    return np.column_stack((X, Y, Z))


def compute_tool_frame_from_tangent(tangent, world_up, tool_axis='Z'):
    """
    Build rotation matrix so chosen tool axis aligns with path tangent.
    - If TOOL_AXIS == 'Z', we align tool +Z with tangent.
    """
    z_axis = utils.normalize(tangent)

    # find x_axis close to world_up projected perpendicular to z_axis
    proj = world_up - np.dot(world_up, z_axis) * z_axis
    if np.linalg.norm(proj) < 1e-6:
        proj = np.array([1.0, 0.0, 0.0]) - np.dot([1.0,0.0,0.0], z_axis) * z_axis
    x_axis = utils.normalize(proj)

    # y = z × x
    y_axis = np.cross(z_axis, x_axis)
    y_axis = utils.normalize(y_axis)

    # re-orthogonalize x = y × z
    x_axis = np.cross(y_axis, z_axis)
    x_axis = utils.normalize(x_axis)

    if tool_axis.upper() == 'Z':
        R = np.column_stack((x_axis, y_axis, z_axis))  # columns are X, Y, Z of tool frame
    elif tool_axis.upper() == 'X':
        # if you want +X along tangent instead:
        X = z_axis
        Y = x_axis
        Z = np.cross(X, Y); Z = utils.normalize(Z)
        Y = np.cross(Z, X); Y = utils.normalize(Y)
        R = np.column_stack((X, Y, Z))
    else:
        raise ValueError("Unsupported TOOL_AXIS")

    return R


################################
# MAIN  
################################


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

    # 5. build waypoints with smooth, continuous orientation
    waypoints = []
    R_prev = None

    for i in range(len(path_pts)):
        # tangent estimate like before
        i_a = max(i - 2, 0)
        i_b = min(i + 2, len(path_pts) - 1)
        p_a = path_pts[i_a]
        p_b = path_pts[i_b]
        tangent = p_b - p_a
        if np.linalg.norm(tangent) < 1e-12 and i < len(path_pts)-1:
            tangent = path_pts[i+1] - path_pts[i]

        p_curr = path_pts[i]

        if R_prev is None:
            R_tool_world = initial_frame_from_up(tangent, WORLD_UP, tool_axis='X')
        else:
            R_tool_world = transport_frame(R_prev, tangent)

        # store for next step
        R_prev = R_tool_world

        # expected world->tool, so we transpose R.
        R_world_to_tool = R_tool_world.T

        # RPY only for legacy usage
        rpy = utils.rotm_to_rpy_zyx(R_world_to_tool)

        # Provide continuous rotation matrix 
        mat_flat = R_tool_world.reshape(-1).tolist()

        # apply offset (same as before)
        p_curr_offset = np.array([
            p_curr[0] + OFFSET_X,
            p_curr[1] + OFFSET_Y,
            p_curr[2] + OFFSET_Z,
        ])

        waypoints.append({
            "pos": p_curr_offset.tolist(),
            "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
            "mat": mat_flat,             
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
        "mat": waypoints[-1]["mat"],
        "idx": len(path_pts)-1
    })



    # 6. save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)

    print(f"Saved {len(waypoints)} waypoints to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
