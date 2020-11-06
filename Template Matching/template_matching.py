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


def save_video(output_frames, fps, J, I, directory):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(directory, fourcc, fps, (J, I))

    for output_frame in output_frames:
        out.write(output_frame)

    out.release()


def exhaustive_search(ref, frames, p, fps, video: bool):
    start = time.time()

    M = ref.shape[0]
    N = ref.shape[1]
    I = frames[0].shape[0]
    J = frames[0].shape[1]

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
                if i < 0 or i > I - M or j < 0 or j > J - N:
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
    if video:
        save_video(output_frames, fps, J, I, "./io/exhaustive.mov")
    print("exhaustive done in", time.time() - start, "seconds")

    return search / len(frames)


def logarithmic_search(ref, frames, p, fps, video: bool):
    start = time.time()

    M = ref.shape[0]
    N = ref.shape[1]
    I = frames[0].shape[0]
    J = frames[0].shape[1]

    num_of_frames = len(frames)
    search = 0
    output_frames = []
    prev_i = prev_j = -1

    for f in range(num_of_frames):
        min_dist = np.inf
        selected_i = 0
        selected_j = 0

        if f == 0:
            for i in range(I - M + 1):
                for j in range(J - N + 1):
                    if i < 0 or i > I - M or j < 0 or j > J - N:
                        continue

                    dist = calculate_dist(ref, frames[f], i, j)
                    search += 1
                    if dist < min_dist:
                        min_dist = dist
                        selected_i = i
                        selected_j = j

        else:
            temp_p = p
            while True:
                k = np.ceil(np.log2(temp_p))
                d = int(np.power(2, k - 1))

                if d < 1:
                    break

                min_dist = np.inf
                selected_i = 0
                selected_j = 0

                points_x = [prev_j - d, prev_j, prev_j + d]
                points_y = [prev_i - d, prev_i, prev_i + d]
                for y in points_y:
                    for x in points_x:
                        if y < 0 or y > I - M or x < 0 or x > J - N:
                            continue

                        dist = calculate_dist(ref, frames[f], y, x)
                        search += 1
                        if dist < min_dist:
                            min_dist = dist
                            selected_i = y
                            selected_j = x

                prev_i = selected_i
                prev_j = selected_j
                temp_p //= 2

        rgb_test = cv2.cvtColor(frames[f], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rgb_test, (selected_j, selected_i), (selected_j + N, selected_i + M), (0, 0, 255), 2)
        output_frames.append(rgb_test)

        prev_i = selected_i
        prev_j = selected_j

    # ---------------------------------------------
    # write the video file
    if video:
        save_video(output_frames, fps, J, I, "./io/logarithmic_search.mov")
    print("2D logarithmic search done in", time.time() - start, "seconds")

    return search / len(frames)


def hierarchical_search(ref, frames, p, fps, video: bool):
    start = time.time()

    M = ref.shape[0]
    N = ref.shape[1]
    I = frames[0].shape[0]
    J = frames[0].shape[1]

    num_of_frames = len(frames)
    search = 0
    output_frames = []
    prev_i = prev_j = -1

    for f in range(num_of_frames):
        min_dist = np.inf
        selected_i = 0
        selected_j = 0

        if f == 0:
            for i in range(I - M + 1):
                for j in range(J - N + 1):
                    if i < 0 or i >= I - M or j < 0 or j >= J - N:
                        continue

                    dist = calculate_dist(ref, frames[f], i, j)
                    search += 1
                    if dist < min_dist:
                        min_dist = dist
                        selected_i = i
                        selected_j = j

        else:
            # step 1
            # create level 0, 1, 2
            image_levels_ref = [ref_image]
            image_levels_test = [frames[f]]
            for i in range(1, 3):
                image_levels_ref.append(cv2.pyrDown(image_levels_ref[i - 1]))
                image_levels_test.append(cv2.pyrDown(image_levels_test[i - 1]))

            temp_p = [p // 4, 1, 1]
            center_i = [prev_i // 4]
            center_j = [prev_j // 4]

            # step 2-4
            for i in range(3):
                min_dist = np.inf
                selected_i = 0
                selected_j = 0

                points_x = [center_j[i] - temp_p[i], center_j[i], center_j[i] + temp_p[i]]
                points_y = [center_i[i] - temp_p[i], center_i[i], center_i[i] + temp_p[i]]

                temp_I = image_levels_test[2 - i].shape[0]
                temp_J = image_levels_test[2 - i].shape[1]
                temp_M = image_levels_ref[2 - i].shape[0]
                temp_N = image_levels_ref[2 - i].shape[1]

                for y in points_y:
                    for x in points_x:
                        if y < 0 or y > temp_I - temp_M or x < 0 or x > temp_J - temp_N:
                            continue

                        dist = calculate_dist(image_levels_ref[2 - i], image_levels_test[2 - i], y, x)
                        search += 1
                        if dist < min_dist:
                            min_dist = dist
                            selected_i = y
                            selected_j = x

                center_i.append(selected_i * 2)
                center_j.append(selected_j * 2)

        rgb_test = cv2.cvtColor(frames[f], cv2.COLOR_GRAY2BGR)
        cv2.rectangle(rgb_test, (selected_j, selected_i), (selected_j + N, selected_i + M), (0, 0, 255), 2)
        output_frames.append(rgb_test)

        prev_i = selected_i
        prev_j = selected_j

    # ---------------------------------------------
    # write the video file
    if video:
        save_video(output_frames, fps, J, I, "./io/hierarchical_search.mov")
    print("hierarchical search done in", time.time() - start, "seconds")

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
    _p = 8

    # exhaustive search
    # exhaustive_search(ref_image, video_frames, _p, _fps, True)

    # 2D logarithmic search
    # logarithmic_search(ref_image, video_frames, _p, _fps, True)

    # hierarchical search
    hierarchical_search(ref_image, video_frames, _p, _fps, True)
    # -------------------------------------------
