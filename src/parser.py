import os
import re
import csv
import sys

# List to store pairs (file path, class name)
env_classes = []

# Function to recursively find Python files and parse for the env decorator and class
def find_envs_in_directory(directory):
    # Normalize the directory path to match the OS
    directory = os.path.normpath(directory)
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Relative path to runner.py
                relative_file_path = os.path.relpath(file_path, start=directory)
                find_env_in_file(relative_file_path, file_path)

# Function to parse each Python file line by line to find the env decorator and class
def find_env_in_file(relative_file_path, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
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
                        # Convert file path to import path format using proper separator
                        import_path = relative_file_path.replace(os.sep, ".")
                        import_path = import_path.replace(".py", "")
                        env_classes.append((import_path, env_name, class_name))
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    # Use the correct path format for any operating system
    tasks_dir = os.path.join("src", "tasks")
    
    # Check if the directory exists
    if not os.path.isdir(tasks_dir):
        print(f"Error: Directory {tasks_dir} not found", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        print("Available directories:", file=sys.stderr)
        print(os.listdir(), file=sys.stderr)
        sys.exit(1)
    
    # Find environment classes
    find_envs_in_directory(tasks_dir)
    
    # Save collected data to CSV
    csv_file = 'env_classes.csv'
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Import Path', 'Env Name', 'Class Name'])
            # Write data
            for file_path, env_name, class_name in env_classes:
                writer.writerow([f"tasks.{file_path}", env_name, class_name])
        print(f"Data saved to {csv_file}")
    except Exception as e:
        print(f"Error writing to CSV file {csv_file}: {str(e)}", file=sys.stderr)

    # If no environments were found, report an error
    if not env_classes:
        print("Warning: No environment classes were found", file=sys.stderr)
