import os
#base = "/home/benjamin/webcam"
base = "/home/benni/dev/learning_to_count_data/vids"
#out = "/home/benjamin/learning_to_count/images"
out = "/home/benni/dev/learning_to_count_data/images"
files =  [x for x in os.listdir(base) if x.endswith(".mp4")]
for f in files:
    f_name = f.split(".")[0]
    os.system(f"ffmpeg -hwaccel cuda -i {base}/{f} -vf \"crop=w='min(iw\,ih)':h='min(iw\,ih)',scale=500:500,setsar=1\" -r 0.5 '{out}/{f_name}_%06d.jpg'")