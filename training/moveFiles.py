import os
import shutil

output_dir = 'combination'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir('healthy_test'):
    if file.endswith('.jpg'):  # Check if the file is a .jpg image
        original_path = os.path.join('healthy_test', file)
        new_filename = 'healthy_' + file
        new_path = os.path.join(output_dir, new_filename)
        
        shutil.copy(original_path, new_path)

for file in os.listdir('ms_test'):
    if file.endswith('.jpg'):
        original_path = os.path.join('ms_test', file)
        new_filename = 'ms_' + file
        new_path = os.path.join(output_dir, new_filename)
        
        shutil.copy(original_path, new_path)