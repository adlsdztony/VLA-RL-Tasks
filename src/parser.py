import os
import re
import csv
import sys

env_classes = []
def find_envs_in_directory(directory):
    directory = os.path.normpath(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                relative_file_path = os.path.relpath(file_path, start=directory)
                find_env_in_file(relative_file_path, file_path)

def find_env_in_file(relative_file_path, file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            match = re.match(r'@register_env\("([^"]+)"', line)
            if match:
                env_name = match.group(1)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    class_match = re.match(r'class\s+(\w+)\s*\(BaseEnv\):', next_line)
                    if class_match:
                        class_name = class_match.group(1)
                        import_path = relative_file_path.replace(os.sep, ".")
                        import_path = import_path.replace(".py", "")
                        env_classes.append((import_path, env_name, class_name))
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    tasks_dir = os.path.join("src", "tasks")
    
    if not os.path.isdir(tasks_dir):
        print(f"Error: Directory {tasks_dir} not found", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        print("Available directories:", file=sys.stderr)
        print(os.listdir(), file=sys.stderr)
        sys.exit(1)
    
    find_envs_in_directory(tasks_dir)
    
    csv_file = 'env_classes.csv'
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Import Path', 'Env Name', 'Class Name'])
            for file_path, env_name, class_name in env_classes:
                writer.writerow([f"tasks.{file_path}", env_name, class_name])
        print(f"Data saved to {csv_file}")
    except Exception as e:
        print(f"Error writing to CSV file {csv_file}: {str(e)}", file=sys.stderr)

    if not env_classes:
        print("Warning: No environment classes were found", file=sys.stderr)
