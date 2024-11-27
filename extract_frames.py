import os
base = "/home/benjamin/webcam"
out = "/home/benjamin/learning_to_count/images"
files =  [x for x in os.listdir(f"{base}/vids") if x.endswith(".mp4")]
for f in files:
    f_name = f.split(".")[0]
    os.system(f"ffmpeg -i {base}/vids/{f} -r 0.2 '{out}/{f_name}_%06d.jpg'")