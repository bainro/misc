import os

# Get all files in the current directory
files = os.listdir()

# Filter to only include .jpg files (if necessary)
image_files = [f for f in files if f.endswith('.jpg')]

# Sort the files numerically
image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Rename the files
for i, file in enumerate(image_files):
    # Generate the new filename
    new_name = f"{i}.jpg"
    
    # Rename the file
    os.rename(file, new_name)
    print(f"Renamed {file} to {new_name}")