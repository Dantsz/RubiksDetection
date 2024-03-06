from pickle import load
from rpd.metafeatures import Face
import os

#read each color file in the test_data folder
# Get the path to the test_data folder
test_data_folder = './test_data'

# Iterate over each file in the test_data folder
for filename in os.listdir(test_data_folder):
    # Construct the full path to the file
    file_path = os.path.join(test_data_folder, filename)

    # Check if the file is a pickle file
    if filename.endswith('.pickle'):
        # Open the file in binary mode
        with open(file_path, 'rb') as file:
            # Unpickle the face array
            face_array = load(file, fix_imports=False)

            # Process the face array as needed
            # ...
            print(len(face_array))
