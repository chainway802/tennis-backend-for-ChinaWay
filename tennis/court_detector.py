# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import time
import numpy as np
from sympy import Line
from itertools import combinations

from .lib.models.CourtDetectionModel import CourtDetectionModel

__all__ = [
    "CourtDetector"
]

court_detection_model = CourtDetectionModel()


class CourtDetector(object):
    """
    球场检测和跟踪
    """

    def __init__(self):
        self.colour_threshold = 200   # 二值化阈值
        self.dist_tau = 3  # 控制线宽的参数
        self.intensity_threshold = 40  # 判断球场线的参数
        self.minLineLength = 100  # 最短线段长度
        self.maxLineGap = 20  # 被判别为同一直线的最大距离
        self.v_width = 0
        self.v_height = 0
        self.gray = None
        self.pmatrix_list = []
        self.inv_pmatrix_list = []
        self.max_age = 40
        self.best_conf = 12
        self.court_match_threshold = 2000
        self.frame_points = None

        # 测试全局模型是否可用
        court_detection_model.init_model("")
        court_detection_model.inference(None)


    def detect_court(self, frame):
        """
        对外暴露的检测球场线接口
        :param frame: 当前帧的图像
        :return:
        """
        if len(self.pmatrix_list) == 0:
            return self._detect_court(frame)
        else:
            return self._track_court(frame, cdist=None)

    def _detect_court(self, frame):
        """
        检测球场线
        :param frame: 当前帧的图像
        :return: 球场线
        """
        # 获取二值化阈值分割图
        self.gray = self._threshold(frame)
        # 获取图像长宽
        self.v_height, self.v_width = self.gray.shape

        # 过滤出球场线
        filtered = self._filter_pixels(self.gray)

        # 霍夫变换检测水平和垂直的直线
        horizontal_lines, vertical_lines = self._detect_lines(filtered)

        # 获取实际球场线和参照球场线之间透视变换的变换矩阵
        pmatrix, inv_pmatrix, score = self._find_homography(horizontal_lines, vertical_lines)
        # 变换矩阵添加到列表中
        self.pmatrix_list.append(pmatrix)
        self.pmatrix_list = self.pmatrix_list[-self.max_age:]
        self.inv_pmatrix_list.append(inv_pmatrix)
        self.inv_pmatrix_list = self.inv_pmatrix_list[-self.max_age:]

        # 将参照球场变换后返回
        lines = self._find_lines_location()

        return lines

    def _track_court(self, frame, cdist=None):
        """
        检测后跟踪球场线位置
        :param frame: 当前帧的图像
        :param cdist: 微调阈值
        :return: 球场线
        """
        # 拷贝原始图像，获取灰度图
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 初始化参数
        dist = (5 if cdist is None else cdist)
        if self.frame_points is None:
            # 获取参考球场4个关键点映射到图像的点
            conf_points = np.array(COURT_CONFERENCE.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points, self.pmatrix_list[-1]).squeeze().round()

        # 获取4个关键点对应的4条直线
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]

        # 在参考线条的基础上微调,获得新的线条
        new_lines = []
        for line in lines:
            # 获取帧中每一行的100个样本点
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]
            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # 如果直线的一端超出框架，则只得到框架内的点
            if p1 is not None or p2 is not None:
                print("点在屏幕外")
                points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[1:-1]

            new_points = []
            # 查找每个点附近的最大强度像素
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))
                top_y, top_x = max(p[1] - dist, 0), max(p[0] - dist, 0)
                bottom_y, bottom_x = min(p[1] + dist, self.v_height), min(p[0] + dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]
                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > 150:
                    new_p = (x + top_x + 1, y + top_y + 1)
                    new_points.append(new_p)
                    cv2.circle(copy, p, 1, (255, 0, 0), 1)
                    cv2.circle(copy, new_p, 1, (0, 0, 255), 1)
            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
            # 求新点的拟合线
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),
                              (int(x + vx * self.v_width), int(y + vy * self.v_width))))

            # 如果发现微调的采样点少于50个，则重新检测球场线，而不是跟踪球场线
            if len(new_points) < 50:
                if dist > 20:
                    print("球场线变化较大，重新检测")
                    return self.detect_court(frame)
                else:
                    print("微调球场线失败，增大调整阈值")
                    dist += 5
                    return self.track_court(frame, cdist=dist)

        # 从新线条中获得透视变换矩阵
        i1 = self._line_intersection(new_lines[0], new_lines[2])
        i2 = self._line_intersection(new_lines[0], new_lines[3])
        i3 = self._line_intersection(new_lines[1], new_lines[2])
        i4 = self._line_intersection(new_lines[1], new_lines[3])
        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        # 计算透视变换矩阵及其逆矩阵
        pmatrix, _ = cv2.findHomography(np.float32(COURT_CONFERENCE.court_conf[self.best_conf]), intersections, method=0)
        inv_pmatrix = cv2.invert(pmatrix)[1]
        # 将变换矩阵添加到对象列表
        self.pmatrix_list.append(pmatrix)
        self.pmatrix_list = self.pmatrix_list[-self.max_age:]
        self.inv_pmatrix_list.append(inv_pmatrix)
        self.inv_pmatrix_list = self.inv_pmatrix_list[-self.max_age:]
        # 保存当前的4个交点
        self.frame_points = intersections

        # 将参照球场变换后返回
        lines = self._find_lines_location()

        return lines

    def _threshold(self, frame):
        """
        白色像素的简单阈值处理
        :param frame: 当前帧的图像
        :return: 当前帧的灰度图
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        return gray

    def _filter_pixels(self, gray):
        """
        使用球场线结构筛选像素
        :param gray: 当前帧的灰度图
        :return: 过滤后的球场线灰度图
        """
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold) and (gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue
                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold) and (gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, filtered):
        """
        使用 Hough 变换查找画面中的所有直线
        :param filtered: 过滤后的灰度图
        :return:
        """
        # 霍夫变换提取直线
        lines = cv2.HoughLinesP(filtered, 1, np.pi / 180, 80, minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
        lines = np.squeeze(lines)

        # 划分水平和垂直的直线
        horizontal, vertical = self._classify_lines(lines)

        # 合并直线
        horizontal, vertical = self._merge_lines(horizontal, vertical)

        return horizontal, vertical

    def _classify_lines(self, lines):
        """
        将直线分为垂直线和水平线
        :param lines: 检测出的所有直线
        :return: 水平直线，垂直直线
        """
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # 使用垂直直线的最低点和最高点过滤水平线
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        """
        合并属于同一框架的直线
        :param horizontal_lines: 水平直线
        :param vertical_lines: 垂直直线
        :return: 合并后的水平直线，合并后的垂直直线
        """
        # 合并水平直线
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # 合并垂直直线
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = self._line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = self._line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_vertical_lines.append(line)

        return new_horizontal_lines, new_vertical_lines

    def _line_intersection(self, line1, line2):
        """
        查找两条直线的交点
        :param line1: 第一条直线
        :param line2: 第二条直线
        :return: 返回两条直线的交点
        """
        l1 = Line(line1[0], line1[1])
        l2 = Line(line2[0], line2[1])
        intersection = l1.intersection(l2)

        return intersection[0].coordinates

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        使用 4 对匹配点查找从参照球场到框架球场的变换
        :param horizontal_lines: 水平直线
        :param vertical_lines: 垂直直线
        :return:
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None

        # 循环遍历每一对水平线和每一对垂直线
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                # 寻找所有直线的交点
                i1 = self._line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = self._line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = self._line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = self._line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[0:2]), tuple(v2[2:])))
                intersections = [i1, i2, i3, i4]
                intersections = self._sort_intersection_points(intersections)

                # 寻找变换矩阵
                for num, points in COURT_CONFERENCE.court_conf.items():
                    matrix, _ = cv2.findHomography(np.float32(points), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    # 获取变换得分
                    confi_score = self._get_confi_score(matrix)
                    # 维护得分最大的变换
                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = num
                        if max_score > self.court_match_threshold:
                            return max_mat, max_inv_mat, max_score

        return max_mat, max_inv_mat, max_score

    def _sort_intersection_points(self, intersections):
        """
        将交叉点从左上角向右下角排序
        :param intersections:
        :return:
        """
        """
        sort intersection points from top left to bottom right
        """
        y_sorted = sorted(intersections, key=lambda x: x[1])
        p12 = y_sorted[:2]
        p34 = y_sorted[2:]
        p12 = sorted(p12, key=lambda x: x[0])
        p34 = sorted(p34, key=lambda x: x[0])
        return p12 + p34

    def _get_confi_score(self, matrix):
        """
        计算转换得分
        :param matrix: 透视变换矩阵
        :return: 得分
        """
        court = cv2.warpPerspective(COURT_CONFERENCE.court, matrix, self.gray.shape[::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def _find_lines_location(self):
        """
        将标准球场线转换为图像中的球场线
        :return: 图像中的球场线
        """
        conference_court_lines = np.array(COURT_CONFERENCE.conference_court_lines, dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(conference_court_lines, self.pmatrix_list[-1]).reshape(-1)

        return lines


class CourtReference(object):
    """
    标准球场
    """

    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))  # ab
        self.baseline_bottom = ((286, 2935), (1379, 2935))  # cd
        self.net = ((286, 1748), (1379, 1748))  # st
        self.left_court_line = ((286, 561), (286, 2935))  # ac
        self.right_court_line = ((1379, 561), (1379, 2935))  # bd
        self.left_inner_line = ((423, 561), (423, 2935))  # eg
        self.right_inner_line = ((1242, 561), (1242, 2935))  # fh
        self.middle_line = ((832, 1110), (832, 2386))  # mn
        self.top_inner_line = ((423, 1110), (1242, 1110))  # ij
        self.bottom_inner_line = ((423, 2386), (1242, 2386))  # kl
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)

        # self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
        #                    2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
        #                        self.right_inner_line[1]],
        #                    3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
        #                        self.right_court_line[1]],
        #                    4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
        #                        self.right_inner_line[1]],
        #                    5: [*self.top_inner_line, *self.bottom_inner_line],
        #                    6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
        #                    7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
        #                    8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
        #                        self.right_court_line[1]],
        #                    9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
        #                        self.left_inner_line[1]],
        #                    10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
        #                         self.middle_line[1]],
        #                    11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
        #                         self.bottom_inner_line[1]],
        #                    12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}

        self.court_conf = {5: [*self.top_inner_line, *self.bottom_inner_line],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}

        self.line_width = 1
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        self.conference_court_lines = [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                                       *self.left_inner_line, *self.right_inner_line, *self.middle_line, *self.top_inner_line, *self.bottom_inner_line]

        conference_court_image_path = os.path.join(os.path.dirname(__file__), "lib/resource/court_reference.png")
        self.court = cv2.cvtColor(cv2.imread(conference_court_image_path), cv2.COLOR_BGR2GRAY)


COURT_CONFERENCE = CourtReference()
