import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img.convert("RGBA")) / 255

def plot_cylinder(images, radius=None, angle_gap=5):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    num_images = len(images)
    angle_interval = (360 - angle_gap * num_images) / num_images

    for idx, image_file in enumerate(images):
        # Load the image
        img = load_image(image_file)

        if radius is None:
            radius = img.shape[1] / (2 * np.pi)
            
        z_range = img.shape[0] * radius / img.shape[1]
        
        # Generate cylinder coordinates
        theta = np.linspace(idx * (angle_interval + angle_gap), (idx + 1) * angle_interval + idx * angle_gap, img.shape[1], endpoint=False) * np.pi / 180
        z = np.linspace(0, z_range, img.shape[0])
        theta_grid, z_grid = np.meshgrid(theta, z)
        x = radius * np.cos(theta_grid)
        y = radius * np.sin(theta_grid)

        # Plot the image on the cylinder surface
        ax.plot_surface(x, y, z_grid, facecolors=img, shade=False)

    ax.set_axis_off()
    ax.set_box_aspect([1,1,z_range / (2 * radius)])
    
#     plt.show()
    plt.savefig("/home/liurui/mp3dbev/tools/vis/parano/rgb.png", transparent=True)



# Example usage:
image_num = 11
images = [f"/home/liurui/mp3dbev/tools/vis/parano/{i}.jpg" for i in range(image_num)]
# images = ['new/1.png', 'new/2.png', 'new/3.png', 'new/3.png', 'new/3.png', 'new/3.png']
# images = ['new/1.png']
plot_cylinder(images, 1, angle_gap=5)