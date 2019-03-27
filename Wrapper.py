#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework1: AutoCalib

Author:
Srinidhi Sreenath (ssreenat@terpmail.umd.edu)
Masters in Engineering Robotics
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import math
import argparse
from glob import glob


def getCalibMatrix(b):
    v = (b[0][1]*b[0][3] - b[0][0]*b[0][4])/(b[0][0]*b[0][2] - b[0][1]**2)
    lamda = b[0][5] - (b[0][3]**2 +
                       v*(b[0][1]*b[0][3] - b[0][0]*b[0][4]))/b[0][0]
    alpha = math.sqrt(lamda/b[0][0])
    beta = math.sqrt(lamda*b[0][0]/(b[0][0]*b[0][2] - b[0][1]**2))
    gamma = (-1*b[0][1]*alpha**2*beta)/(lamda)
    u = (gamma*v)/beta - (b[0][3]*alpha**2)/lamda

    print("u = {}\nv = {}\nlamda = {}\naplha = {}\nbeta = {}\ngamma = {}\n".format(
        u, v, lamda, alpha, beta, gamma))

    A = np.array([[alpha, gamma, u], [0, beta, v], [0, 0, 1]])
    return A


def getBMatrix(V):
    u, s, vh = np.linalg.svd(V, full_matrices=True)
    # print vh
    # solve Vb = 0 for b
    b = vh[-1:]
    return b


def updateVMatrix(H, V):
    v_12 = [H[0][0]*H[0][1], (H[0][0]*H[1][1] + H[1][0]*H[0][1]), H[1][0]*H[1][1],
            (H[2][0]*H[0][1] + H[0][0]*H[2][1]), (H[2][0]*H[1][1] + H[1][0]*H[2][1]), H[2][0]*H[2][1]]

    # print v_12

    trm1 = H[0][0]*H[0][0] - H[0][1]*H[0][1]
    trm2 = 2*(H[0][0]*H[1][0] - H[0][1]*H[1][1])
    trm3 = H[1][0]*H[1][0] - H[1][1]*H[1][1]
    trm4 = 2*(H[2][0]*H[0][0] - H[0][1]*H[2][1])
    trm5 = 2*(H[2][0]*H[1][0] - H[1][1]*H[2][1])
    trm6 = H[2][0]*H[2][0] - H[2][1]*H[2][1]

    v_1122 = []
    v_1122.append(trm1)
    v_1122.append(trm2)
    v_1122.append(trm3)
    v_1122.append(trm4)
    v_1122.append(trm5)
    v_1122.append(trm6)

    # print v_1122

    V.append(v_12)
    V.append(v_1122)

    # print V
    # print "\n \n"


def computeHomography(matched_points):
    homography = np.zeros((3, 3))

    # select 4 matched points
    pair1 = matched_points[0, ]
    pair2 = matched_points[1, ]
    pair3 = matched_points[2, ]
    pair4 = matched_points[3, ]

    # define x1, y1 ... x4, y4 and x1',y1' ... x4',y4'
    x1, y1, x1_dash, y1_dash = pair1[0,
                                     0], pair1[0, 1], pair1[1, 0], pair1[1, 1]
    x2, y2, x2_dash, y2_dash = pair2[0,
                                     0], pair2[0, 1], pair2[1, 0], pair2[1, 1]
    x3, y3, x3_dash, y3_dash = pair3[0,
                                     0], pair3[0, 1], pair3[1, 0], pair3[1, 1]
    x4, y4, x4_dash, y4_dash = pair4[0,
                                     0], pair4[0, 1], pair4[1, 0], pair4[1, 1]

    # define P matrix
    P = np.zeros((9, 9))
    P[0][0], P[2][0], P[4][0], P[6][0] = -x1, -x2, -x3, -x4
    P[0][1], P[2][1], P[4][1], P[6][1] = -y1, -y2, -y3, -y4
    P[0][2], P[2][2], P[4][2], P[6][2] = -1, -1, -1, -1

    P[1][3], P[3][3], P[5][3], P[7][3] = -x1, -x2, -x3, -x4
    P[1][4], P[3][4], P[5][4], P[7][4] = -y1, -y2, -y3, -y4
    P[1][5], P[3][5], P[5][5], P[7][5] = -1, -1, -1, -1

    P[0][6], P[1][6], P[2][6], P[3][6], P[4][6], P[5][6], P[6][6], P[7][6] = x1*x1_dash, x1 * \
        y1_dash, x2*x2_dash, x2*y2_dash, x3*x3_dash, x3*y3_dash, x4*x4_dash, x4*y4_dash
    P[0][7], P[1][7], P[2][7], P[3][7], P[4][7], P[5][7], P[6][7], P[7][7] = y1*x1_dash, y1 * \
        y1_dash, y2*x2_dash, y2*y2_dash, y3*x3_dash, y3*y3_dash, y4*x4_dash, y4*y4_dash
    P[0][8], P[1][8], P[2][8], P[3][8], P[4][8], P[5][8], P[6][8], P[7][8] = x1_dash, y1_dash, x2_dash, y2_dash, x3_dash, y3_dash, x4_dash, y4_dash

    # Since we know that last element of homography is 1, I define another row in P matrix to make it easier to solve for H.
    P[8][8] = 1

    # define b matrix
    b = np.zeros((9, 1))
    b[8][0] = 1

    # PH = b. Solve for H.
    h = np.linalg.solve(P, b)
    homography[0][0], homography[0][1], homography[0][2] = h[0], h[1], h[2]
    homography[1][0], homography[1][1], homography[1][2] = h[3], h[4], h[5]
    homography[2][0], homography[2][1], homography[2][2] = h[6], h[7], h[8]

    return homography


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--basepath', default="Data",
                        help='Path to directory with images to stitch for panaroma')

    Args = Parser.parse_args()
    dirpath = Args.basepath
    completepath = str(dirpath) + str("/*.JPG")
    images = sorted(glob(completepath))

    _3d_points = []
    _2d_points = []
    V = []

    x, y = np.meshgrid(range(7), range(5))
    world_points = np.hstack((x.reshape(35, 1), y.reshape(
        35, 1))).astype(np.float32)
    world_points = world_points*23
    world_points = np.asarray(world_points)
    # print world_points

    H1 = np.array([[-19.0845, 981.0899, 100.7805], [-1704.8808, -
                                                    64.26607, 285.5835], [-0.008419, -0.1168, 1.0]])
    H2 = np.array([[167.74933, 421.35797, 117.690033],
                   [-1196.27269, -468.71546, 271.218], [0.778903, -1.7984935, 1.0]])
    H3 = np.array([[-400.095, 470.74177, 133.72457], [-1206.2368, -
                                                      283.841, 240.074313], [-1.52232934, -1.0556823, 1.0]])
    H4 = np.array([[-359.29829, 940.502344, 126.0333], [-1646.69602, -
                                                        272.976406, 305.891843], [-1.7148713, 0.879775592, 1.0]])

    updateVMatrix(H1, V)
    updateVMatrix(H2, V)
    updateVMatrix(H3, V)
    updateVMatrix(H4, V)

    # for imagepath in images:
    #     image = cv2.imread(imagepath)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.resize(gray, (400, 300))

    #     ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

    #     if ret == True:
    #         cv2.drawChessboardCorners(gray, (7, 5), corners, ret)
    #         cv2.imshow("image", gray)
    #         cv2.waitKey(0)

    #         corners = corners.reshape(-1, 2)
    #         # _2d_points.append(corners)  # append current 2D points
    #         # _3d_points.append(world_points)  # 3D points are always the same
    #         # print corners[34], world_points[34]

    #         src = np.asarray(world_points[: 4])  # world
    #         dst = np.asarray(corners[: 4])  # image

    #         src[:, [0, 1]] = src[:, [1, 0]]
    #         dst[:, [0, 1]] = dst[:, [1, 0]]

    #         matched_inliers = []

    #         for i in range(0, 4):
    #             matched_inliers.append([src[i], dst[i]])

    #         matched_inliers = np.asarray(matched_inliers, dtype=np.float32)

    #         H = cv2.getPerspectiveTransform(dst, src)
    #         print H
    #         print "\n \n"
    #         # homography_matrix = computeHomography(matched_inliers)
    #         # print homography_matrix
    #         # print "\n \n"

    #         updateVMatrix(H, V)

    #     cv2.destroyAllWindows()

    V = np.asarray(V)
    b = getBMatrix(V)
    print b

    K = getCalibMatrix(b)


if __name__ == '__main__':
    main()
