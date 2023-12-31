import os
import math

def rename_images(image_folder):
    # List all files in the folder
    all_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.JPG')]
    total_files = len(all_files)

    # Calculate the number of files for validation
    num_valid = math.ceil(total_files / 10)

    # Loop through all files and rename
    for i, filename in enumerate(all_files):
        old_path = os.path.join(image_folder, filename)
        
        # Determine new filename
        if i < num_valid:
            new_filename = f"valid_{filename}"
        else:
            new_filename = f"train_{filename}"
        
        new_path = os.path.join(image_folder, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed '{old_path}' to '{new_path}'")


for f in os.listdir('images'):
    rename_images(os.path.join('images', f))
