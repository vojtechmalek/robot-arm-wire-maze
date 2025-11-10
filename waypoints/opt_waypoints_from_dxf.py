import ezdxf
import numpy as np
import json
import utils

################################
# CONFIG
################################

MAZE = "E"
DXF_FILE = f"dxf_paths/{MAZE}_path.dxf"        # your DXF
OUTPUT_FILE = "opt_waypoint_paths/waypoints_E.json"      # output JSON, working file
TOOL_AXIS = 'X'                       # which tool axis should align with path tangent
UNIT_SCALE = 0.001                    # DXF usualy mm -> convert to m
WORLD_UP = np.array([0, 0, 1.0])      # reference "up" to stabilize roll

#################### Constants that tweak waypoint generation, all in meters ####################

RESAMPLE_STEP = 0.005               # step between points when sampling freshly extracted path

# offset because model path can be on models edge
OFFSET_X = 0.000                    # use if your dxf is off in X axis from origin, in meters
OFFSET_Y = 0.000                    # use if your dxf is off in Y axis from origin, in meters
OFFSET_Z = 0.000                    # use if your dxf is off in Z axis from origin, in meters

# optimalization of the path, safety guide offset and deletion of collinear waypoints
GUIDE_OFFSET = 0.08                 # offset in -maze-direction before the maze start to prevent colision
HORIZONTAL_GUIDE_OFFSET = 0.12      # extra +Z offset for horizontal starts to prevent colision
TWO_POINTS_DIST_TRESHOLD = 0.02     # minimal distance between two final waypoints to account for interpolation problems
HORIZONTAL_START = True if MAZE == "E" or MAZE == "D" else False # True if you maze has a horizontal section at the start and in not D or E


################################
# DXF GEOMETRY HANDLING
################################


import numpy as np

def extract_ordered_path_points(dxf_doc, tol=1e-6):
    """
    Extracts polylines/lines/splines and stitches them into one ordered path
    using endpoint-node grouping + graph walk to guarantee correct orientation.
    tol: distance tolerance to consider two endpoints identical.
    Returns Nx3 numpy array in original DXF units.
    """
    msp = dxf_doc.modelspace()
    polylines = []

    # collect geometry
    for e in msp.query("LWPOLYLINE"):
        elevation = getattr(e.dxf, "elevation", 0.0)
        pts = []
        for v in e.get_points():
            x, y, *_ = v
            z = elevation
            pts.append([x, y, z])
        if len(pts) > 1:
            polylines.append(np.array(pts, dtype=float))

    for e in msp.query("POLYLINE"):
        pts = []
        for v in e.vertices:
            x, y, z = v.dxf.location.xyz
            pts.append([x, y, z])
        if len(pts) > 1:
            polylines.append(np.array(pts, dtype=float))

    for e in msp.query("SPLINE"):
        spline_pts = e.approximate(200)
        pts = np.array(spline_pts, dtype=float)
        if len(pts) > 1:
            polylines.append(pts)

    for e in msp.query("LINE"):
        p0 = np.array(e.dxf.start, dtype=float)
        p1 = np.array(e.dxf.end, dtype=float)
        pts = np.vstack([p0, p1])
        polylines.append(pts)

    if not polylines:
        raise RuntimeError("No usable path entities found.")

    nodes = []          # list of representative 3D points for nodes
    seg_end_nodes = []  # for each segment, tuple (node_idx_start, node_idx_end)

    def find_or_add_node(pt):
        """Return node index for pt, creating new node if none within tol."""
        for i, n in enumerate(nodes):
            if np.linalg.norm(n - pt) <= tol:
                return i
        nodes.append(pt.copy())
        return len(nodes) - 1

    for seg in polylines:
        start = seg[0].astype(float)
        end = seg[-1].astype(float)
        si = find_or_add_node(start)
        ei = find_or_add_node(end)
        seg_end_nodes.append((si, ei))

    adjacency = {i: [] for i in range(len(nodes))}
    for s_idx, (si, ei) in enumerate(seg_end_nodes):
        adjacency[si].append((s_idx, True))
        adjacency[ei].append((s_idx, False))

    # find start node: prefer a node with degree 1 (open path).
    start_node = None
    for n, neigh in adjacency.items():
        if len(neigh) == 1:
            start_node = n
            break
    if start_node is None:
        start_node = 0

    used_seg = [False] * len(polylines)
    ordered_pts = []

    cur_node = start_node
    prev_node = None

    # If start_node has degree > 0, determine the first segment to take.
    while True:
        choices = [(s_idx, orient) for (s_idx, orient) in adjacency[cur_node] if not used_seg[s_idx]]

        if not choices:
            break
        seg_idx, orient_at_node = choices[0]

        seg = polylines[seg_idx]
        forward = orient_at_node  # True => seg is already in forward orientation relative to cur_node

        used_seg[seg_idx] = True

        # produce points in the correct orientation
        if forward:
            pts_to_add = seg.copy()
            next_node = seg_end_nodes[seg_idx][1]  # end node
        else:
            pts_to_add = np.flip(seg, axis=0).copy()
            next_node = seg_end_nodes[seg_idx][0]  # after flipping, previous end becomes start, next is original start

        # append points avoiding duplication at junction
        if not ordered_pts:
            for p in pts_to_add:
                ordered_pts.append(p)
        else:
            for p in pts_to_add[1:]:
                ordered_pts.append(p)

        prev_node = cur_node
        cur_node = next_node

        # Continue until all segments used
        if all(used_seg):
            break

    # If some segments remain unused (e.g., graph had branches), attempt to stitch remaining components
    if not all(used_seg):
        for seg_idx, used in enumerate(used_seg):
            if not used:
                si, ei = seg_end_nodes[seg_idx]
                # representative points
                rep_si = nodes[si]
                rep_ei = nodes[ei]
                # find index in ordered_pts with minimal distance
                ordered_arr = np.array(ordered_pts)
                d_si = np.linalg.norm(ordered_arr - rep_si, axis=1).min()
                d_ei = np.linalg.norm(ordered_arr - rep_ei, axis=1).min()
                # choose orientation that minimizes gap
                seg = polylines[seg_idx]
                if d_si <= d_ei:
                    seg_oriented = seg.copy()
                else:
                    seg_oriented = np.flip(seg, axis=0).copy()
                # append with simple duplication avoidance
                for p in seg_oriented:
                    if len(ordered_pts) == 0 or np.linalg.norm(ordered_pts[-1] - p) > tol:
                        ordered_pts.append(p)
                used_seg[seg_idx] = True

    pts_all = np.array(ordered_pts, dtype=float)
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
    """
    Ensures that first point will start facing up in every path (later gets fliped)
    """
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
    if MAZE == "B" or MAZE == "D":
        pass  #placeholder, no fliping
    else:
        path_pts = np.flip(path_pts, axis=0)

    # 5. build waypoints with smooth, continuous orientation
    waypoints = []
    R_prev = None
    # smoothin_factor*2 +1 = window lenght, whis window is used to smooth out rotation in sharp corners
    smoothing_factor = 1  

    for i in range(len(path_pts)):
        # tangent estimate like before
        i_a = max(i - smoothing_factor, 0)
        i_b = min(i + smoothing_factor, len(path_pts) - 1)
        p_a = path_pts[i_a]
        p_b = path_pts[i_b]
        tangent = p_b - p_a
        if np.linalg.norm(tangent) < 1e-12 and i < len(path_pts) - 1:
            tangent = path_pts[i + 1] - path_pts[i]

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

        # at first add initial waypoint to guide into maze 
        idx_shift = 1 # due to adding guide-in points

        if len(waypoints) < 1:
            p_shifted = utils.move_back_along_x(p_curr_offset, R_tool_world, distance=GUIDE_OFFSET)

            if HORIZONTAL_START:
                idx_shift += 1
                p_horizontal_shifted = utils.move_up_along_z(p_shifted, R_tool_world, distance=GUIDE_OFFSET)
                waypoints.append({
                    "pos": p_horizontal_shifted.tolist(),
                    "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                    "mat": mat_flat,
                    "idx": int(i)
                })

            waypoints.append({
                "pos": p_shifted.tolist(),
                "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                "mat": mat_flat,
                "idx": int(i)
            })

        # compress if last two mats equal to current mat == colinear points
        if len(waypoints) >= 2:
            same_orientation = (
                utils.mats_equal(waypoints[-2]["mat"], waypoints[-1]["mat"]) and
                utils.mats_equal(waypoints[-1]["mat"], mat_flat)
            )
            close_points = utils.points_dist_threshold_check(waypoints[-2]["pos"], waypoints[-1]["pos"], threshold=TWO_POINTS_DIST_TRESHOLD)

            if same_orientation and close_points:
                waypoints.pop()


        # now append the new waypoint with correct id
        waypoints.append({
            "pos": p_curr_offset.tolist(),
            "rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
            "mat": mat_flat,
            "idx": int(i) + idx_shift
        })

    # last point, reuse last orientation — also run the same compression logic
    p_last = path_pts[-1]
    p_last_offset = np.array([
        p_last[0] + OFFSET_X,
        p_last[1] + OFFSET_Y,
        p_last[2] + OFFSET_Z,
    ])

    mat_last = waypoints[-1]["mat"]            # reuse orientation
    rpy_last = waypoints[-1]["rpy"]

    if len(waypoints) >= 2:
        if utils.mats_equal(waypoints[-2]["mat"], waypoints[-1]["mat"]) and utils.mats_equal(waypoints[-1]["mat"], mat_last):
            waypoints.pop()

    waypoints.append({
        "pos": p_last_offset.tolist(),
        "rpy": rpy_last,
        "mat": mat_last,
        "idx": len(path_pts) - 1
    })



    # 6. save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(waypoints, f, indent=2)

    print(f"Saved {len(waypoints)} waypoints to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()