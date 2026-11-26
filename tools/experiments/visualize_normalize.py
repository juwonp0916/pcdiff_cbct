import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# points_1 = np.load(r"C:\Users\stick\OneDrive\Documents\HKUST\UROP\output\20251230\epoch_3199\DBT2_3348L_[13]_condition.npy")
# points_2 = np.load(r"C:\Users\stick\OneDrive\Documents\HKUST\UROP\output\20251230\epoch_3199\DBT2_3348L_[13]_gt.npy")

points_1 = np.load(r"/Users/juwonpark/Documents/HKUST/UROP/pcdiff_cbct/validation/Without_Conditioning/FDI 42/syn/sample_000/DBT2_2952L_[11]_condition.npy")
points_2 = np.load(r"/Users/juwonpark/Documents/HKUST/UROP/pcdiff_cbct/validation/Without_Conditioning/FDI 42/syn/sample_000/DBT2_2952L_[11]_gt.npy")

# First window - Generated
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(points_1[:, 0], points_1[:, 1], points_1[:, 2], s=0.5)  # s=1 for small dots
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
ax1.set_zlim([-2, 2])
ax1.grid(False)
ax1.set_axis_off()
plt.title("Generated Point Cloud")

# Second window - Ground Truth
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(points_2[:, 0], points_2[:, 1], points_2[:, 2], s=2)  # s=1 for small dots
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
ax2.set_zlim([-2, 2])

# ax2.grid(False)
# ax2.set_axis_off()
plt.title("Ground Truth Point Cloud")

plt.show()
