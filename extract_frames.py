import os
from multiprocessing import Pool
import sys

base = os.environ.get("LTC_DATA")
print(base)
vids = f"{base}/vids"
out = f"{base}/images"

filter = sys.argv[1]
files =  [x for x in os.listdir(vids) if filter in x]
print(files)

def extract_frames_from_video(vid):
    f_name = vid.split(".")[0]
    os.system(f"ffmpeg -hide_banner -loglevel error -i {vids}/{vid} -vf \"crop=w='min(iw,ih)':h='min(iw,ih):x=0:y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_l_%06d.jpg'")
    os.system(f"ffmpeg -hide_banner -loglevel error -i {vids}/{vid} -vf \"crop=w='min(iw,ih)':h='min(iw,ih):x=(iw/2-min(iw,ih)/2):y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_c_%06d.jpg'")
    os.system(f"ffmpeg -hide_banner -loglevel error -i {vids}/{vid} -vf \"crop=w='min(iw,ih)':h='min(iw,ih):x=(iw-min(iw,ih)):y=0',scale=640:640,setsar=1\" -r 0.5 '{out}/{f_name}_r_%06d.jpg'")

if __name__ == '__main__':
    with Pool(4) as p:
        p.map(extract_frames_from_video,files)
        p.close()

