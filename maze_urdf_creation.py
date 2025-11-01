#!/usr/bin/env python3
"""
Generate a URDF maze where each straight segment becomes both a visual
and a collision box (aligned with XYZ direction from waypoints).
Also adds a small visual arrow at the origin pointing +X for orientation.
"""

import os
import json
import math

# ======= USER CONFIGURATION =======
WAYPOINT_NAME = "A"   # <-- Change this (A, B, C, etc.)
WAYPOINT_DIR  = "waypoints"
THICKNESS_X   = 0.01
THICKNESS_Y   = 0.01
ANGLE_TOL_DEG = 2.0
DUPE_DIST     = 1e-9
# ==================================

def load_waypoints(filepath):
    """Load waypoints, ignoring any rpy values."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    data = sorted(data, key=lambda d: d.get("idx", 0))
    pts = []
    for w in data:
        if "pos" not in w or len(w["pos"]) != 3:
            raise ValueError(f"{filepath}: each waypoint must have 'pos':[x,y,z]")
        x, y, z = map(float, w["pos"])
        pts.append((x, y, z))
    return pts

def dist(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def normalize(v):
    n = math.sqrt(sum(c*c for c in v))
    if n == 0: return None
    return (v[0]/n, v[1]/n, v[2]/n)

def dot(a, b):
    return sum(a[i]*b[i] for i in range(3))

def midpoint(a, b):
    return tuple((a[i]+b[i])/2 for i in range(3))

def pitch_yaw_from_dir(u):
    """Compute roll, pitch, yaw so local Z aligns with vector u."""
    ux, uy, uz = u
    uz = max(-1.0, min(1.0, uz))
    pitch = math.acos(uz)
    yaw = math.atan2(uy, ux)
    return (0.0, pitch, yaw)

def remove_duplicates(points, dupe_dist):
    if not points: return points
    result = [points[0]]
    for p in points[1:]:
        if dist(p, result[-1]) > dupe_dist:
            result.append(p)
    return result

def simplify_collinear(points, angle_tol_deg):
    """Merge points that are nearly straight."""
    if len(points) <= 2:
        return points
    tol = math.radians(angle_tol_deg)
    keep = [True] * len(points)
    for i in range(1, len(points) - 1):
        a, b, c = points[i-1], points[i], points[i+1]
        v1 = normalize((b[0]-a[0], b[1]-a[1], b[2]-a[2]))
        v2 = normalize((c[0]-b[0], c[1]-b[1], c[2]-b[2]))
        if not v1 or not v2:
            continue
        ang = math.acos(max(-1.0, min(1.0, dot(v1, v2))))
        if ang <= tol:
            keep[i] = False
    return [p for p, k in zip(points, keep) if k]

def generate_urdf(points, robot_name, link_name):
    """Build URDF with visual+collision boxes for maze and a small +X arrow."""
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{robot_name}">')
    lines.append(f'  <link name="{link_name}">')
    lines.append('    <material name="maze_white"><color rgba="1 1 1 1"/></material>')
    lines.append('    <material name="maze_arrow_red"><color rgba="1 0 0 1"/></material>')

    # Maze boxes
    for i in range(len(points)-1):
        p0, p1 = points[i], points[i+1]
        L = dist(p0, p1)
        if L == 0:
            continue
        u = normalize((p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]))
        r, p, y = pitch_yaw_from_dir(u)
        m = midpoint(p0, p1)

        # VISUAL
        lines.append('    <visual>')
        lines.append(f'      <origin xyz="{m[0]:.6f} {m[1]:.6f} {m[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}" />')
        lines.append('      <geometry>')
        lines.append(f'        <box size="{THICKNESS_X:.6f} {THICKNESS_Y:.6f} {L:.6f}" />')
        lines.append('      </geometry>')
        lines.append('      <material name="maze_white" />')
        lines.append('    </visual>')

        # COLLISION
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="{m[0]:.6f} {m[1]:.6f} {m[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}" />')
        lines.append('      <geometry>')
        lines.append(f'        <box size="{THICKNESS_X:.6f} {THICKNESS_Y:.6f} {L:.6f}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')

    # Visual arrow marker at origin pointing +X (no collision)
    lines.append('    <visual name="direction_arrow">')
    # a simple cylinder “arrow” centered near the origin, lying along X axis
    lines.append('      <origin xyz="0.05 0 0" rpy="0 0 1.5708" />  <!-- rotate so cylinder axis is +X -->')
    lines.append('      <geometry>')
    lines.append('        <cylinder radius="0.002" length="0.1" />')
    lines.append('      </geometry>')
    lines.append('      <material name="maze_arrow_red" />')
    lines.append('    </visual>')

    lines.append('  </link>')
    lines.append('</robot>')
    return "\n".join(lines)

def main():
    waypoint_file = os.path.join(WAYPOINT_DIR, f"waypoint_{WAYPOINT_NAME}.json")
    if not os.path.exists(waypoint_file):
        alt = os.path.join(WAYPOINT_DIR, f"waypoints_{WAYPOINT_NAME}.json")
        if os.path.exists(alt):
            waypoint_file = alt
        else:
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_file} (also tried {alt})")

    pts = load_waypoints(waypoint_file)
    pts = remove_duplicates(pts, DUPE_DIST)
    pts = simplify_collinear(pts, ANGLE_TOL_DEG)

    urdf_text = generate_urdf(pts, robot_name=f"maze_{WAYPOINT_NAME}", link_name="mazeLink")
    out_path = f"maze_{WAYPOINT_NAME}.urdf"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(urdf_text)

    print(f"✅ Generated {out_path} with {len(pts)-1} segments and a +X arrow visual.")

if __name__ == "__main__":
    main()
