import pandas as pd
import glob
import os

# Define column names
column_names = ['ID', 'X', 'rho', 'v1', 'Ttrans', 'p', 'p1', 'q1', 'mv1v1v1', 'mvvv1v1']

def process_file(file_path):
    # Get the name of the subfolder containing the file
    subfolder_name = os.path.basename(os.path.dirname(file_path))
    
    # Read the file header to extract the value of I
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Extract the value of ITEM: NUMBER OF CELLS
        i_value = int(lines[3].strip())  # The 4th line stores the value of I

    # Skip the first 9 lines and read the data section
    data = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=9)
    
    # Sort the data by ID
    sorted_data = data.sort_values(by='ID', ascending=True)
    
    # Construct the output filename for the forward shock data
    fwd_output_file = os.path.join(
        os.path.dirname(file_path),
        f"fwd_{os.path.basename(file_path).replace('tmp_flowvss.', '').split('.')[-1]}.dat"
    )
    
    # Write the Tecplot header and forward shock data
    with open(fwd_output_file, 'w') as f:
        f.write("TITLE = \"DSMC\"\n")
        f.write("VARIABLES = \"ID\" \"X\" \"rho\" \"v1\" \"Ttrans\" \"p\" \"p1\" \"q1\" \"mv1v1v1\" \"mvvv1v1\"\n")
        f.write(f"ZONE T=\"{subfolder_name}\", I={i_value}, J=1, K=1, F=POINT\n")  # Use the subfolder name
        sorted_data.to_csv(f, sep=' ', index=False, header=False)
    
    print(f"Forward shock data from '{file_path}' has been sorted and saved to '{fwd_output_file}' with headers.")

    # Create the inverted shock data
    inverted_data = sorted_data.copy()
    inverted_data['X'] = -inverted_data['X']  # Negate X
    inverted_data['v1'] = -inverted_data['v1']  # Negate v1
    inverted_data['q1'] = -inverted_data['q1']  # Negate q1
    inverted_data['mv1v1v1'] = -inverted_data['mv1v1v1']  # Negate mv1v1v1
    # Keep the other variables unchanged

    # Sort by X in ascending order
    inverted_data = inverted_data.sort_values(by='X', ascending=True)

    # Construct the output filename for the inverted shock data
    inv_output_file = os.path.join(
        os.path.dirname(file_path),
        f"inv_{os.path.basename(file_path).replace('tmp_flowvss.', '').split('.')[-1]}.dat"
    )

    # Write the Tecplot header and inverted shock data
    with open(inv_output_file, 'w') as f:
        f.write("TITLE = \"DSMC\"\n")
        f.write("VARIABLES = \"ID\" \"X\" \"rho\" \"v1\" \"Ttrans\" \"p\" \"p1\" \"q1\" \"mv1v1v1\" \"mvvv1v1\"\n")
        f.write(f"ZONE T=\"{subfolder_name}_inv\", I={i_value}, J=1, K=1, F=POINT\n")
        inverted_data.to_csv(f, sep=' ', index=False, header=False)
    
    print(f"Inverted shock data from '{file_path}' has been saved to '{inv_output_file}'.")

# Recursively collect paths of all tmp_flowvss.* files in subfolders
file_paths = glob.glob('./**/tmp_flowvss.*', recursive=True)

# Check whether any files match the pattern
if file_paths:
    for file_path in file_paths:
        # Skip files named tmp_flowvss.0
        if file_path.endswith("tmp_flowvss.0"):
            print(f"Skipping file '{file_path}'")
            continue
        process_file(file_path)
else:
    print("No files found matching pattern './**/tmp_flowvss.*'.")

