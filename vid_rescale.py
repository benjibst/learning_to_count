import os

# iterate over files in a directory
for file in [f for f in os.listdir(".") if os.path.isfile(f) and f.endswith('.mp4')]:
    #execute the shell command
    #ffmpeg -i in.mp4 -vf "crop=in_w/2:in_h/2:in_w/2:in_h/2" -c:a copy out.mp4
    fpscut_name = file.split('.')[0] + '_fpscut.' + file.split('.')[1]
    cropped_name = file.split('.')[0] + '_cropped.' + file.split('.')[1]
    scaled_name = file.split('.')[0] + '_scaled.' + file.split('.')[1]
    os.system(f"ffmpeg -i {file} -vf \"fps=1/10\" -c:a copy {fpscut_name}")
    #os.system(f"ffmpeg -i {fpscut_name} -vf \"crop=in_h:in_h:(in_w-in_h)/2:0\" -c:a copy {cropped_name}")
    #os.system(f"ffmpeg -i {cropped_name} -s 240x240 -c:a copy {scaled_name}")
os.remove("*_fpscut.*")
os.remove("*_cropped.*")
