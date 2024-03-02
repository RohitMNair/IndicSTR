import shutil
from pathlib import Path
import argparse

def modify_gt_files(input_file:Path, output_file:Path, folder_no:int, folder_to_rename:int):
    # read the gt file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # for merged gt file
    modified_lines = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            img_path, label = parts
            path_parts = img_path.split('/')
            if len(path_parts) >= 2 and path_parts[1] == folder_to_rename:
                path_parts[1] = str(folder_no)
                new_path = '/'.join(path_parts)
                modified_lines.append(f"{new_path}\t{label}\n")
            elif len(path_parts) < 2:
                raise Exception("Invalid image path {path}")
        else:
            raise Exception(f"Invalid label line {line} found")

    with open(output_file, 'a') as f:
        f.writelines(modified_lines)    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    print(len(lines))

def merge_image_folder(src_img_folder:Path, dest_img_folder:Path, gt_file:str):
    # get the name of the last image folder
    last_img_folder = 0
    for item in dest_img_folder.iterdir():
        if item.is_dir():
            count = int(item.name)
            last_img_folder = count if count > last_img_folder else last_img_folder
        else:
            raise Exception(f"file found instead of image folders {item}")

    folder_no = last_img_folder + 1
    # start copying from last image folder name + 1
    for img_folder in src_img_folder.iterdir():
        if img_folder.is_dir():
            shutil.copytree(img_folder, dest_img_folder / str(folder_no))
            modify_gt_files(
                src_img_folder.parent / gt_file, 
                dest_img_folder.parent / gt_file, 
                folder_no, 
                img_folder.name,
            )
            folder_no += 1
        else:
            raise Exception(f"{item} is a file not a folder")

def merge_dataset(folder1:str, folder2:str, dest_folder:str):
    # Create a new folder for merged contents
    merged_folder = Path(dest_folder)
    merged_folder.mkdir()

    # instantiate the folder paths
    folder1 = Path(folder1)
    folder2 = Path(folder2)

    # check if folders exist or not
    if not folder1.exists():
        raise Exception(f"folder1 does not exist {folder1}")
    if not folder2.exists():
        raise Exception(f"folder2 does not exist {folder2}")
    
    # Iterate through the contents of the folder1 and copy it to
    # merged folder
    for item in folder1.iterdir():
        # If item is a file, copy it to the destination folder
        if item.is_file() and item.name == 'gt.txt':
            print(f"Copied {item}")
            shutil.copy(item, merged_folder)
        # If item is a folder, recursively copy its contents
        elif item.is_dir() and item.name == 'images':
            print(f"Copied {item}")
            new_destination = merged_folder / item.name
            shutil.copytree(item, new_destination)
    
    print(f"Contents of {folder1} copied successfully.")

    # merge images
    merge_image_folder(folder2 / "images", merged_folder / "images", "gt.txt")
    print(f"Contents of {folder2} gt.txt copied successfully")
    # merge_image_folder(folder2 / "glyph_masks", merged_folder / "glyph_masks", "glyph_coords.txt")
    # print(f"Contents of {folder2} glyph_coords.txt copied successfully")
    # merge_image_folder(folder2 / "masks", merged_folder / "masks", "coords.txt")      
    # print(f"Contents of {folder2} coords.txt copied successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program that Merges to Datasets in SynthTiger Dataset format')

    parser.add_argument('-d1', '--dataset1', help= 'Path to the 1st dataset to be merged', required= True)
    parser.add_argument('-d2', '--dataset2', help= 'Path to the 2nd dataset to be merged', required= True)
    parser.add_argument('-m', '--mergeFolder', help= 'Path to the destination folder where dataset will be merged', required= True)

    # Parse the command line arguments
    args = parser.parse_args()

    folder1 = args.dataset1
    folder2 = args.dataset2
    dest_folder = args.mergeFolder
    merge_dataset(folder1, folder2, dest_folder)