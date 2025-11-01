#!/usr/bin/env python3
"""
Generate a URDF collision model of a maze segment from waypoint JSON files.

Usage:
  Just edit the `WAYPOINT_NAME` variable at the top (e.g. "A", "B", "C"...)
  and run:  python generate_maze_urdf.py

This reads all JSON waypoints in the folder:
    waypoints/waypoint_<WAYPOINT_NAME>.json
and generates:
    maze_<WAYPOINT_NAME>.urdf
"""

import os
import json
import math

# ======= USER CONFIGURATION =======
WAYPOINT_NAME = "B"   # <-- Change this (A, B, C, etc.)
WAYPOINT_DIR  = "waypoints"
THICKNESS_X   = 0.01
THICKNESS_Y   = 0.01
ANGLE_TOL_DEG = 2.0    # tolerance to merge straight lines
DUPE_DIST     = 1e-9
# ==================================

def load_waypoints(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    data = sorted(data, key=lambda d: d.get("idx", 0))
    return [(float(p["pos"][0]), float(p["pos"][1]), float(p["pos"][2])) for p in data]

def dist(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))

def normalize(v):
    n = math.sqrt(sum(c*c for c in v))
    if n == 0: return None
    return (v[0]/n, v[1]/n, v[2]/n)

def dot(a, b):
    return sum(a[i]*b[i] for i in range(3))

def midpoint(a, b):
    return tuple((a[i]+b[i])/2 for i in range(3))

def pitch_yaw_from_dir(u):
    """Return roll, pitch, yaw so local Z aligns with vector u."""
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
    """Build URDF XML string for straight segments."""
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{robot_name}">')
    lines.append(f'  <link name="{link_name}">')

    for i in range(len(points)-1):
        p0, p1 = points[i], points[i+1]
        L = dist(p0, p1)
        if L == 0: continue
        u = normalize((p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]))
        r, p, y = pitch_yaw_from_dir(u)
        m = midpoint(p0, p1)

        lines.append('    <collision>')
        lines.append(f'      <origin xyz="{m[0]:.6f} {m[1]:.6f} {m[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}" />')
        lines.append('      <geometry>')
        lines.append(f'        <box size="{THICKNESS_X:.6f} {THICKNESS_Y:.6f} {L:.6f}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')

    lines.append('  </link>')
    lines.append('</robot>')
    return "\n".join(lines)

def main():
    waypoint_file = os.path.join(WAYPOINT_DIR, f"waypoints_{WAYPOINT_NAME}.json")
    if not os.path.exists(waypoint_file):
        raise FileNotFoundError(f"Waypoint file not found: {waypoint_file}")

    pts = load_waypoints(waypoint_file)
    pts = remove_duplicates(pts, DUPE_DIST)
    pts = simplify_collinear(pts, ANGLE_TOL_DEG)

    urdf_text = generate_urdf(pts, robot_name=f"maze_{WAYPOINT_NAME}", link_name="mazeLink")
    out_path = f"maze_{WAYPOINT_NAME}.urdf"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(urdf_text)

    print(f"✅ Generated {out_path} with {len(pts)-1} collision segments.")

if __name__ == "__main__":
    main()
