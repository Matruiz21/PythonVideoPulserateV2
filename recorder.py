import cv2,os,time
from tqdm import tqdm

name = "recording_rest2"
video_path = "D:/User/Nerdex/Documentos/ITBA/Tesis/Videos/source_videos_part_16-001/source_videos/W135/BlendShape/camera_front/W135_BlendShape_camera_front.mp4"
if not os.path.exists(name):
    os.mkdir(name)

cap = cv2.VideoCapture(video_path)
timestamps = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for idx in tqdm(range(frame_count)):
    ret, frame = cap.read()
    timestamps+=[str(time.time())]


    #cv2.imshow('Camera',frame)
    cv2.imwrite(f"{name}/{idx}.png",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f = open(f"{name}/timestamps.txt","w")
f.write(",".join(timestamps))
f.close()
cap.release()
cv2.destroyAllWindows()