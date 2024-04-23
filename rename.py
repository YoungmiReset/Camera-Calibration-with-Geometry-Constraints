import os

def rename_images(folder_path):

    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    # jpg_files_sorted = sorted(jpg_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    jpg_files_sorted = sorted(jpg_files)

    for index, jpg_file in enumerate(jpg_files_sorted):

        new_name = f"{index+1}_color.png"

        old_path = os.path.join(folder_path, jpg_file)

        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = "D:/tamed/ThirdCalib/NO08/left_calib"
    rename_images(folder_path)
