import os.path
import cv2


def fix_ucf_sports(base_dir):
    video_files = [
        "ucf action/Golf-Swing-Front/002/7608-9_70(2)151.avi",
        "ucf action/Golf-Swing-Back/001/3283-8_700741.avi",
        "ucf action/Golf-Swing-Back/002/3283-8_701201.avi",
        "ucf action/Golf-Swing-Back/003/7608-12_70275.avi",
        "ucf action/Golf-Swing-Back/004/7616-7_70270.avi",
        "ucf action/Golf-Swing-Back/005/RF1-13903_70070.avi"
    ]
    for video_file in video_files:
        _extract_frames(os.path.join(base_dir, video_file))

    image_folders = [
        "ucf action/Diving-Side/008", "ucf action/Diving-Side/008",
        "ucf action/Diving-Side/009", "ucf action/Diving-Side/010",
        "ucf action/Diving-Side/011", "ucf action/Diving-Side/012",
        "ucf action/Diving-Side/013", "ucf action/Diving-Side/014"
    ]
    for image_folder in image_folders:
        _create_video(os.path.join(base_dir, image_folder))


def _create_video(image_folder):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    filelist = sorted(os.listdir(image_folder))
    for filename in filelist:
        if filename.endswith(".jpg"):
            frame = cv2.imread(os.path.join(image_folder, filename))
            if out is None:
                out = cv2.VideoWriter(
                    os.path.join(image_folder, filename[:-4] + ".avi"), fourcc,
                    10.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)
    out.release()


def _extract_frames(video_file):
    frame_filename = video_file[:-7]
    frame_counter = video_file[-7:-4]
    counter_len = len(frame_counter)
    frame_counter = int(frame_counter)
    video = cv2.VideoCapture(video_file)
    while video.grab():
        __, frame = video.retrieve()
        cv2.imwrite(
            frame_filename + f'{frame_counter:0{counter_len}}' + ".jpg", frame)
        frame_counter += 1
    video.release()
