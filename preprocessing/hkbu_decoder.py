import os
import glob
from tqdm.auto import tqdm

import cv2
import argparse
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image

'''
root
 |-attack (target these folders)
   |-- 01
   |-- 02
 |-real (target these folders)
   |-- 01
   |-- 02
'''


def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


if __name__ == "__main__":
    default_labels_file = "/root/datasets/HKBU-V2/labels.txt"
    default_output_csv = "/root/datasets/HKBU-V2/labels.csv"
    default_device = 'cuda:0'

    parser = argparse.ArgumentParser(
            description='Extract frames from HKBU videos.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('videos_folder', type=str, help='Absolute path to folder to process videos.')
    parser.add_argument('-t', '--file_type', type=str, help='File type of video.', default="avi")
    parser.add_argument('-f', '--frames', type=int, help='Number of frames to extract.', default=5)
    parser.add_argument('-o', '--output_csv', type=str, help='File to output CSV label file.', default=default_output_csv)
    parser.add_argument('-l', '--label', type=str, help='The label for all files.', default=1)
    parser.add_argument('-d', '--device', type=str, help='The device to run MTCNN face cropper.', default=default_device)
    parser.add_argument('-s', '--output_size', type=int, help='Output size of cropped images.', default=224)
    parser.add_argument('-de', '--debug', type=bool, help='Debug mode.', default=False)

    args = parser.parse_args()

    video_folders = glob.glob(args.videos_folder + "/[0-9]*")

    # img_folder = args.videos_folder.rstrip("/") + "/imgs/"
    # make_folder(img_folder)

    data = {
        'path': [],
        'target': []
    }

    vertical_sensors = ["04", "05", "06"]

    face_detector = MTCNN(image_size=args.output_size, device=args.device)

    vf_pbar = tqdm(video_folders)
    for video_folder in vf_pbar:
        vf_pbar.set_description("Processing person folder %s " % video_folder)
        abs_path = os.path.abspath(video_folder)

        video_files = glob.glob(abs_path + "" + "/*." + args.file_type)

        img_folder = abs_path.rstrip("/") + "/imgs/"
        make_folder(img_folder)

        pbar2 = tqdm(video_files)
        for video_file in pbar2:
            video_folder_name = video_file.split("/")[-1].split(".")[0]

            label = args.label
            pbar2.set_description("Processing video %s " % video_folder_name)

            sensor = video_folder_name.split("_")[0]
            video_folder = img_folder + video_folder_name

            make_folder(video_folder)

            vidcap = cv2.VideoCapture(video_file)

            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            increment = int(length / args.frames)

            success, image = vidcap.read()
            count = 1
            while success:
                if count == 3 or count % increment == 0:
                    file_name = video_folder + "/%d.jpg" % count

                    data['path'].append(file_name)
                    data['target'].append(label)

                    # cv2.imwrite(uncropped_path, image)  # save uncropped frame

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if sensor == "04":
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    elif sensor == "05" or sensor == "06":
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    image_pil = Image.fromarray(image)
                    cropped_image = face_detector(image_pil, save_path=file_name)   # save cropped frame

                success, image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1

            vidcap.release()

    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)


