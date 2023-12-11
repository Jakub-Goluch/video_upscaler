import cv2
from tkinter import filedialog
from os import getcwd
import numpy as np

""""https://learnopencv.com/super-resolution-in-opencv/"""

def CPUprocessing(video):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # for sharpening
    result_video = cv2.VideoWriter('rezultat.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.get(cv2.CAP_PROP_FPS), (1920,1080))

    # setting up optical flow vector params

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.5,
                          minDistance=3,
                          blockSize=9)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors for movement detection
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it after interpolating it
    ret, first_frame = video.read()
    first_frame = cv2.resize(first_frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    first_frame = cv2.bilateralFilter(first_frame, 10, 100, 100)
    first_frame = cv2.filter2D(first_frame, -1, kernel)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    counter = 0 # for refreshing reference frame for optical flow

    # setting enhancing model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    path = "FSRCNN_x3.pb"

    sr.readModel(path)

    sr.setModel("fsrcnn", 3)
    print("model loaded")

    while video.isOpened():
        success, frame = video.read()

        if success:
            # interpolation
            frame_final = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_NEAREST_EXACT)
            # denoising
            frame_final = cv2.bilateralFilter(frame_final, 10, 100, 100)
            # sharpening
            frame_final = cv2.filter2D(frame_final, -1, kernel)

            counter += 1
            if counter == 30:
                counter = 1
                old_gray = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # calculating optical flow
            frame_gray = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            # print(good_new.shape)

            # find the bound box for better image enhancement
            offset = 20
            x1 = int(np.min(good_new[:, 0])) - offset
            x2 = int(np.max(good_new[:, 0])) + offset
            y1 = int(np.min(good_new[:, 1])) - offset
            y2 = int(np.max(good_new[:, 1])) + offset

            image_to_supersample = frame[x1:x2, y1:y2]
            image_to_supersample = sr.upsample(image_to_supersample)
            print("frame supersampled")
            shape = (frame_final[x1:x2, y1:y2].shape[1], frame_final[x1:x2, y1:y2].shape[0])
            image_to_supersample = cv2.resize(image_to_supersample, shape, interpolation=cv2.INTER_LANCZOS4)
            # cv2.imshow('image', image_to_supersample)
            print(frame_final[x1:x2, y1:y2].shape[:2], image_to_supersample.shape[:2])
            print("frame resized")

            # replacing the better interpolated image to the final frame
            frame_final[x1:x2, y1:y2] = image_to_supersample
            print("frame replaced")

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame_final = cv2.circle(frame_final, (int(a), int(b)), 5, color[i].tolist(), -1)
            frame_final = cv2.add(frame_final, mask)

            # showing both original and upscaled frames as one
            both_images = np.concatenate((frame, frame_final), axis=1)
            both_images = cv2.resize(both_images, (1920, 1080), interpolation=cv2.INTER_NEAREST_EXACT)
            cv2.imshow('Video', both_images)
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            result_video.write(both_images)
            key = cv2.waitKey(33) & 0xFF
            if key == 27:
                break
        else:
            break

    result_video.release()

video_path = filedialog.askopenfilename(title="Select a video", filetypes=[("MP4 files", "*.mp4")], initialdir=getcwd())
video_original = cv2.VideoCapture(video_path)

CPUprocessing(video_original)