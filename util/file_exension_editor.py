import os
import glob

image_extensions = [".jpg", ".jpeg", ".png", ".jfif"]

def add_file_extension(directory, extension):
    files = glob.glob(os.path.join(directory, "*"))

    for file_path in files:
        if os.path.isfile(file_path):
            file_name, file_ext = os.path.splitext(file_path)
            if file_ext.lower() not in image_extensions:
                new_file_path = file_name + file_ext + extension
                os.rename(file_path, new_file_path)
                
            elif file_ext.lower() in image_extensions and file_ext.lower() != extension:
                new_file_path = file_name + extension
                os.rename(file_path, new_file_path)


def remove_file_extension(directory):
    files = glob.glob(os.path.join(directory, "*"))

    for file_path in files:
        if os.path.isfile(file_path):
            if os.path.splitext(file_path)[1].lower() in image_extensions:
                new_file_path = os.path.splitext(file_path)[0]
                os.rename(file_path, new_file_path)