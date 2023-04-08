import os

# Set the path to the folder containing the files
folder_path = "capstone\\actual_test\\sensor_i"

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file name starts with "ACC_F232"
    if file_name.startswith("ACC_F232"):
        # Create the new file name by replacing "ACC_F232" with "ACC_F1"
        new_file_name = file_name.replace("ACC_F232", "ACC_F1")

        # Get the full paths for the original and new file names
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed file: {file_name} -> {new_file_name}")
