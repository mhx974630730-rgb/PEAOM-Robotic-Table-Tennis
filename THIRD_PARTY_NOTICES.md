# Third-party and excluded components

This repository intentionally excludes the following components:

1. **Physical robot communication source code.** The physical robot bridge was built from source code supplied directly by the robot manufacturer and communicates through the manufacturer's serial protocol. The authors are not permitted to redistribute that source code.
2. **Camera SDK and Python binding.** The optional camera tools import `mvsdk`, which must be obtained from the camera vendor and installed according to the vendor's terms. The SDK dynamic libraries and the vendor-provided Python wrapper are not included.
3. **Device-specific camera profiles.** Generated `.mvdat` camera configuration files are hardware-specific and are not included.

The PyBullet simulation, observation models, training code, ROS message definition, URDF assets, and author-written analysis utilities do not require the excluded robot communication source code.
