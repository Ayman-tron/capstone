import numpy as np
import matplotlib.pyplot as plt

# Generate random data points for demonstration
np.random.seed(42)
healthy_points = np.random.rand(30, 2) * 50
unhealthy_points = (np.random.rand(30, 2) * 50) + 50

# Example white data point
white_data_point = np.array([40, 60])

# Set the size of the data points
point_size = 100

# Plot healthy points in red
plt.scatter(healthy_points[:, 0], healthy_points[:, 1],
            color='red', label='Healthy', s=point_size)

# Plot unhealthy points in blue
plt.scatter(unhealthy_points[:, 0], unhealthy_points[:, 1],
            color='blue', label='Unhealthy', s=point_size)

# Plot the white data point
plt.scatter(white_data_point[0], white_data_point[1], color='white',
            edgecolors='black', label='Unknown', s=point_size)

# Set axis labels
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')

# Add a legend
plt.legend()

# Show the plot
plt.show()
