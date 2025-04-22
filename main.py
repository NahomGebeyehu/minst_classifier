import numpy as np
from matplotlib import pyplot as plt

# Load image data
with open('./mnist_dataset/t10k-images.idx3-ubyte', 'rb') as f:
    raw_data = f.read()

# Skip 16-byte header, then reshape
image_data = np.frombuffer(raw_data, dtype=np.uint8, offset=16)
with open ('./falafel/image_data.txt','w') as f:
    f.write(str(image_data))

images = image_data.reshape(-1, 28, 28)

# Load label data
with open('./mnist_dataset/t10k-labels.idx1-ubyte', 'rb') as f:
    raw_labels = f.read()

# Skip 8-byte header
labels = np.frombuffer(raw_labels, dtype=np.uint8, offset=8)

# Display the first 10 images with labels
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(str(labels[i]), fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()