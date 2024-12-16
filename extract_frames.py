import os
from multiprocessing import Pool
import time
if False:
    base = "/home/benni/dev/learning_to_count_data/vids"
    out = "/home/benni/dev/learning_to_count_data/images"
else:
    base = "/home/benjamin/learning_to_count_data/vids"
    out = "/home/benjamin/learning_to_count_data/images"   
files =  [x for x in os.listdir(base) if x.endswith(".mp4")]
files = [x for x in files if os.stat(f"{base}/{x}").st_ctime > time.time()-0.3*3600]
print(files)

def extract_frames_from_video(vid):
    f_name = vid.split(".")[0]
    os.system(f"ffmpeg -hide_banner -loglevel error -i {base}/{vid} -vf \"crop=w='min(iw\,ih)':h='min(iw\,ih):x=0:y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_l_%06d.jpg'")
    os.system(f"ffmpeg -hide_banner -loglevel error -i {base}/{vid} -vf \"crop=w='min(iw\,ih)':h='min(iw\,ih):x=(iw/2-min(iw\,ih)/2):y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_c_%06d.jpg'")
    os.system(f"ffmpeg -hide_banner -loglevel error -i {base}/{vid} -vf \"crop=w='min(iw\,ih)':h='min(iw\,ih):x=(iw-min(iw\,ih)):y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_r_%06d.jpg'")

if __name__ == '__main__':
    with Pool(4) as p:
        p.map(extract_frames_from_video,files)
        p.close()

