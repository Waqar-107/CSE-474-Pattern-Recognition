# from dust i have come, dust i will be

import cv2
import numpy as np
import time


def calculate_dist(ref, test, idx, jdx):
    M = ref.shape[0]
    N = ref.shape[1]
    test_partial = test[idx: idx + M, jdx: jdx + N].astype(np.int64)

    # sum(diff in each cell ^ 2)
    ans = np.absolute(ref.astype(np.int64) - test_partial)
    ans = ans * ans

    return np.sum(ans)


def exhaustive_search(ref, frames, p, fps):
    start = time.time()

    M = ref.shape[0]
    N = ref.shape[1]
    I = frames[0].shape[0]
    J = frames[0].shape[1]

    print("shape of ref", ref.shape)
    print("shape of video", (I, J))
    print("total frames:", len(frames))

    num_of_frames = len(frames)
    search = 0
    output_frames = []
    prev_i = prev_j = -1

    for f in range(num_of_frames):
        if f == 0:
            i_start = 0
            i_end = I - M + 1
            j_start = 0
            j_end = J - N + 1

        else:
            i_start = prev_i - p
            i_end = prev_i + p
            j_start = prev_j - p
            j_end = prev_j + p

        min_dist = np.inf
        selected_i = 0
        selected_j = 0

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if i < 0 or i >= I - M or j < 0 or j >= J - N:
                    continue

                dist = calculate_dist(ref, frames[f], i, j)
                search += 1
                if dist < min_dist:
                    min_dist = dist
                    selected_i = i
                    selected_j = j

        rgb_test = cv2.cvtColor(frames[f], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rgb_test, (selected_j, selected_i), (selected_j + N, selected_i + M), (0, 0, 255), 2)
        output_frames.append(rgb_test)

        prev_i = selected_i
        prev_j = selected_j

    # ---------------------------------------------
    # write the video file
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("./io/exhaustive.mov", fourcc, fps, (J, I))

    for output_frame in output_frames:
        out.write(output_frame)

    out.release()

    print("exhaustive done in", time.time() - start, "seconds")

    return search / len(frames)


if __name__ == "__main__":
    # -------------------------------------------
    # capture the video
    capture = cv2.VideoCapture("./io/input.mov")
    video_frames = []
    _fps = capture.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        # save in greyscale
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    capture.release()
    # -------------------------------------------

    # -------------------------------------------
    # read the reference image in greyscale
    ref_image = cv2.imread("./io/reference.jpg", 0)
    # -------------------------------------------

    # -------------------------------------------
    # run the algorithms
    _p = 2

    # exhaustive search
    exhaustive_search(ref_image, video_frames, _p, _fps)
    # -------------------------------------------
