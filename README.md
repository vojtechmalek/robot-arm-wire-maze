# Robot arm: contact-free path through a 3D wire maze

Semester project for the Robotics course (B3B33ROB1) at the Czech Technical University in Prague, 2026.

A CRS robot arm has to guide a loop end-effector along a twisted wire ("maze") without touching it.

**What it does**

- **Camera calibration and homography:** detects calibration markers in the camera image and maps 2D image coordinates into the robot's workspace to locate the maze base.
- **Path planning:** converts the wire's skeleton into 3D waypoints and searches a discretised 3D workspace for a valid route between them.
- **Collision checking:** every candidate path is checked in simulation before it runs on the real arm.
- **Result:** the final trajectory ran on the physical CRS arm and got through curves and changing wire segments without contact.

**Report:** [Report_ROB_semestral_project.pdf](Report_ROB_semestral_project.pdf)
