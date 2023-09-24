import numpy as np


class Conv2D:
    def __init__(self) -> None:
        pass

    def __call__(self, image: np.ndarray, kernel: np.ndarray, stride: int = 0, padding: int = 0):
        assert image is not None
        assert kernel is not None

        image_height, image_width, img_channels = image.shape
        kernel_height, kernel_width = kernel.shape

        # Reshape the Kernel to (kernel_size, kernel_size, 1)
        kernel = np.expand_dims(kernel, axis=2)

        # print(image[:, :, 3])

        output_height = (image_height - kernel_height +
                         2 * padding) // stride + 1
        output_width = (image_width - kernel_width + 2 * padding) // stride + 1

        # Using a single kernel, so the same kernel is applied to each input channel. All channel outputs are summed to form a single scalar value.
        result = np.zeros((output_height, output_width))

        if padding > 0:
            image = np.pad(image,
                           ((padding, padding), (padding, padding), (0, 0)),
                           mode='constant')

        for i in range(0, output_height, stride):
            for j in range(0, output_width, stride):
                # Extract the image portion
                img_region = image[i:i+kernel_height, j:j+kernel_width, :]

                # Element-wise multiplication and summation across channels.
                result[i // stride, j // stride] = np.sum(img_region * kernel)

        # Take channel-wise summation for a single filter output
        return result


if __name__ == "__main__":
    image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                      [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
                      [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]],
                      [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]]], dtype=np.float32)

    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)

    STRIDE = 1
    PADDING = 1

    IMG_HEIGHT = 4
    IMG_WIDTH = 4
    IMG_CHANNELS = 3

    KERNEL_HEIGHT = 3
    KERNEL_WIDTH = 3

    image = np.random.rand(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    kernel = np.random.rand(KERNEL_HEIGHT, KERNEL_WIDTH)

    print(f"IMAGE SHAPE: {image.shape}")
    print(f"KERNEL SHAPE: {kernel.shape}")

    convLayer = Conv2D()
    res = convLayer(image, kernel, stride=STRIDE, padding=PADDING,)
    print(f"OUTPUT SHAPE: {res.shape}")
    print()
    print(res)
