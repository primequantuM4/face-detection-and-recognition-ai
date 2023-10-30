from matplotlib import pyplot as plt
from util.file_exension_editor import add_file_extension, remove_file_extension
from util.image_cropper import crop_images


def visualize_cropped_images(cropped_images):
    """ This function visualizes the cropped images. """

    num_images = len(cropped_images)
    rows = 10
    cols = (num_images + 1) // rows

    fig, axes = plt.subplots(rows, cols)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < num_images:
            ax.imshow(cropped_images[i], cmap="gray")

    plt.show()


def main():
    """ This function is the main function of the module. """

    dataset_dir = "dataset/data"
    remove_file_extension(dataset_dir)
    add_file_extension(dataset_dir, ".png")
    visualize_cropped_images(crop_images(dataset_dir))


if __name__ == "__main__":
    main()

    