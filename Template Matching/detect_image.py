# from dust i have come, dust i will be

import cv2
import numpy as np


def calculate_dist(ref, test, idx, jdx):
    M = ref.shape[0]
    N = ref.shape[1]
    test_partial = test[idx: idx + M, jdx: jdx + N].astype(np.int64)

    # sum(diff in each cell ^ 2)
    ans = np.absolute(ref.astype(np.int64) - test_partial)
    ans = ans * ans

    return np.sum(ans)


def exhaustive_search(ref, test):
    M = ref.shape[0]
    N = ref.shape[1]
    I = test.shape[0]
    J = test.shape[1]

    min_dist = np.inf
    selected_i = 0
    selected_j = 0

    for i in range(I - M + 1):
        for j in range(J - N + 1):
            dist = calculate_dist(ref, test, i, j)
            if dist < min_dist:
                min_dist = dist
                selected_i = i
                selected_j = j

    print(selected_i, selected_j, min_dist)
    # now that we have i, j selected we draw a red box
    # note that cv2 keeps the image in BGR format, not RGB!
    rgb_test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)

    # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # i denotes row, so Y-axis, j denotes column, so X-axis
    cv2.rectangle(rgb_test, (selected_j, selected_i), (selected_j + N, selected_i + M), (0, 0, 255), 2)

    cv2.imshow("exhaustive search result", rgb_test)
    cv2.waitKey()
    cv2.destroyWindow("exhaustive search result")


if __name__ == "__main__":
    ref_img = cv2.imread("./io/reference.jpg", 0)
    test_img = cv2.imread("./io/main.PNG", 0)

    exhaustive_search(ref_img, test_img)
