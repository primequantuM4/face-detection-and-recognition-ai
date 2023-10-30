from util.file_exension_editor import add_file_extension, remove_file_extension
from util.image_cropper import crop_images

def main():
    """ This function is the main function of the module. """

    dataset_dir = "dataset/data"
    remove_file_extension(dataset_dir)
    add_file_extension(dataset_dir, ".png")
    print(crop_images(dataset_dir))
    


if __name__ == "__main__":
    main()

    