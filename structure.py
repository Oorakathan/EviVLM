import os
from pathlib import Path

def print_architecture(dir_path, prefix=""):
    path = Path(dir_path)
    
    try:
        # Get all items, sort alphabetically
        all_items = sorted(path.iterdir(), key=lambda x: x.name)
    except PermissionError:
        print(f"{prefix}└── [Access Denied]")
        return

    # Filter into directories and NIfTI files
    dirs = [x for x in all_items if x.is_dir()]
    nii_files = [x for x in all_items if x.is_file() and x.name.endswith(('.nii', '.nii.gz'))]
    
    # Cap the .nii files to top 3
    if len(nii_files) > 3:
        files_to_show = nii_files[:3]
        show_ellipsis = True
        ellipsis_text = f"... (and {len(nii_files) - 3} more .nii files)"
    else:
        files_to_show = nii_files
        show_ellipsis = False
        
    # Calculate total items to print at this level to format the tree branches correctly
    total_lines = len(dirs) + len(files_to_show) + (1 if show_ellipsis else 0)
    current_line = 0
    
    # 1. Print directories recursively
    for d in dirs:
        current_line += 1
        is_last = (current_line == total_lines)
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}📁 {d.name}/")
        
        # Recursive call for sub-folders
        extension = "    " if is_last else "│   "
        print_architecture(d, prefix + extension)
        
    # 2. Print top 3 .nii files
    for f in files_to_show:
        current_line += 1
        is_last = (current_line == total_lines)
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}📄 {f.name}")
        
    # 3. Print the "..." if there are more than 3 files
    if show_ellipsis:
        print(f"{prefix}└── {ellipsis_text}")

if __name__ == "__main__":
    import sys
    
    # Use the path provided as an argument, or default to the current directory
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    base_path = Path(target_dir).absolute()
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{target_dir}' does not exist.")
    else:
        print(f"\n📁 {base_path.name}/")
        print_architecture(base_path)