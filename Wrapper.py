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
import math
import argparse
from glob import glob


def getExtrinsicParams(K, lamda):
    K_inv = np.linalg.inv(K)


def getCalibMatrix(b):
    v = (b[0][1]*b[0][3] - b[0][0]*b[0][4])/(b[0][0]*b[0][2] - b[0][1]**2)
    lamda = b[0][5] - (b[0][3]**2 +
                       v*(b[0][1]*b[0][3] - b[0][0]*b[0][4]))/b[0][0]
    alpha = math.sqrt(lamda/b[0][0])
    beta = math.sqrt(lamda*b[0][0]/(b[0][0]*b[0][2] - b[0][1]**2))
    gamma = (-1*b[0][1]*alpha**2*beta)/(lamda)
    u = (gamma*v)/beta - (b[0][3]*alpha**2)/lamda

    print("u = {}\nv = {}\nlamda = {}\nalpha = {}\nbeta = {}\ngamma = {}\n".format(
        u, v, lamda, alpha, beta, gamma))

    A = np.array([[alpha, gamma, u], [0, beta, v], [0, 0, 1]])
    return A, lamda


def getBMatrix(V):
    _, _, vh = np.linalg.svd(V, full_matrices=True)
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


def computeHomography(corners, world_points):
    n = 20
    src = np.asarray(world_points[: n])  # world
    dst = np.asarray(corners[: n])  # image

    src[:, [0, 1]] = src[:, [1, 0]]
    dst[:, [0, 1]] = dst[:, [1, 0]]

    P = np.zeros((2*n, 9))

    i = 0
    for (srcpt, dstpt) in zip(src, dst):
        x, y, x_dash, y_dash = srcpt[0], srcpt[1], dstpt[0], dstpt[1]

        P[i][0], P[i][1], P[i][2] = -x, -y, -1
        P[i+1][0], P[i+1][1], P[i+1][2] = 0, 0, 0

        P[i][3], P[i][4], P[i][5] = 0, 0, 0
        P[i+1][3], P[i+1][4], P[i+1][5] = -x, -y, -1

        P[i][6], P[i][7], P[i][8] = x*x_dash, y*x_dash, x_dash
        P[i+1][6], P[i+1][7], P[i+1][8] = x*y_dash, y*y_dash, y_dash

        i = i+2

    _, _, vh = np.linalg.svd(P, full_matrices=True)
    h = vh[-1:]
    h.resize((3, 3))

    homography = h/h[2, 2]

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
    for imagepath in images:
        image = cv2.imread(imagepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (400, 300))

        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

        if ret == True:
            cv2.drawChessboardCorners(gray, (7, 5), corners, ret)
            cv2.imshow("image", gray)
            cv2.waitKey(0)

            corners = corners.reshape(-1, 2)

            homography_matrix = computeHomography(corners, world_points)

            updateVMatrix(homography_matrix, V)

        cv2.destroyAllWindows()

    V = np.asarray(V)
    b = getBMatrix(V)

    K, lamda = getCalibMatrix(b)
    print("Calibration matrix: \n\n {} \n".format(K))

    R, t = getExtrinsicParams(K, lamda)


if __name__ == '__main__':
    main()
