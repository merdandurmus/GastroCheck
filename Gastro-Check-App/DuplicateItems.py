import os
import random
import shutil

def equalize_folder_item_count(parent_folder):
    # Step 1: Gather all subfolders in the parent folder
    subfolders = [os.path.join(parent_folder, folder) for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]

    if not subfolders:
        print("No subfolders found.")
        return

    # Step 2: Get the file count in each folder
    folder_file_counts = {}
    for folder in subfolders:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        folder_file_counts[folder] = len(files)

    # Step 3: Determine the maximum number of files in any folder
    max_file_count = max(folder_file_counts.values())
    
    # Step 4: Duplicate files in folders with fewer files than the max
    for folder, file_count in folder_file_counts.items():
        if file_count < max_file_count:
            print(f"Folder '{os.path.basename(folder)}' has {file_count} items. Duplicating to reach {max_file_count} items.")

            # List of files to duplicate
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            num_duplicates_needed = max_file_count - file_count

            for _ in range(num_duplicates_needed):
                file_to_duplicate = random.choice(files)
                original_file_path = os.path.join(folder, file_to_duplicate)

                # Create a new file name for the duplicate
                file_name, file_ext = os.path.splitext(file_to_duplicate)
                new_file_name = f"{file_name}_copy{random.randint(1000, 9999)}{file_ext}"
                new_file_path = os.path.join(folder, new_file_name)

                # Copy the file to create a duplicate
                shutil.copy2(original_file_path, new_file_path)

            print(f"Folder '{os.path.basename(folder)}' now has {max_file_count} items.")

    print("All folders now have the same number of items.")

# Example usage
parent_folder = "Data/Training/Training_Images"  # Replace with the actual path to your "Training_Images" folder
equalize_folder_item_count(parent_folder)
