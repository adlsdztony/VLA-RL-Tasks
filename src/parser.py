import os
import re

# List to store pairs (file path, class name)
env_classes = []

# Function to recursively find Python files and parse for the env decorator and class
def find_envs_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Relative path to runner.py
                relative_file_path = os.path.relpath(file_path, start=directory)
                find_env_in_file(relative_file_path, file_path)

# Function to parse each Python file line by line to find the env decorator and class
def find_env_in_file(relative_file_path, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Loop through the lines to find the @register_env decorator and the class definition
    for i in range(len(lines)):
        line = lines[i].strip()
        
        # Check if the line contains the @register_env decorator
        match = re.match(r'@register_env\("([^"]+)"', line)
        if match:
            # Extract the env_name from the decorator
            env_name = match.group(1)
            
            # Look for the next line to find the class definition
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                class_match = re.match(r'class\s+(\w+)\s*\(BaseEnv\):', next_line)
                if class_match:
                    # Extract the class name
                    class_name = class_match.group(1)
                    # Store the result as a tuple (relative file path, class name)
                    import_path = relative_file_path.replace("\\", ".").replace(".py", "")
                    env_classes.append((import_path, env_name, class_name))

# Run the search in the current directory (or a specific path)
find_envs_in_directory(".\\src\\tasks")

# Print out the collected pairs
for file_path, env_name, class_name in env_classes:
    print(f"{file_path},{env_name},{class_name}")
