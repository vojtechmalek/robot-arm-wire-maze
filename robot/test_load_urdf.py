import os
import pybullet as p

p.connect(p.DIRECT)
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, 'crs97.urdf')
print('Loading URDF:', urdf_path)
try:
    uid = p.loadURDF(urdf_path)
    print('Loaded URDF id:', uid)
except Exception as e:
    print('Exception while loading URDF:', e)
finally:
    p.disconnect()
