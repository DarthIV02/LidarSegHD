import os

def find_best_accuracy(folder_path):
    best_accuracy = 0.0
    best_file = ""
    
    # Iterate over each subfolder
    for subdir in os.listdir(folder_path):
        for file in os.listdir(os.path.join(folder_path, subdir)):
            # Check if the file is named result.txt
            if file == "result.txt":
                file_path = os.path.join(folder_path, subdir, file)
                # Read the last line of the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        try:
                            accuracy = float(last_line.split(',')[-1])
                            # Update best accuracy and file if found a better one
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_file = file_path
                        except ValueError:
                            pass
    
    return best_accuracy, best_file

# Example usage:
folder_path = "LiDARSegHD/save_2/LidarSeg/TLS_models/"
best_accuracy, best_file = find_best_accuracy(folder_path)

if best_file:
    print("Best accuracy found:", best_accuracy)
    print("File with best accuracy:", best_file)
else:
    print("No result.txt files found in the specified folder.")
