import time

import cv2 as cv
import mss
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

if __name__ == "__main__":
    debug = True
    # debug = False
    game = True
    # game = False
    if game:
        set_up_simple_logging()

        bng = BeamNGpy(
            "localhost",
            64256,
            home="C:/Games/PC/BeamNG.drive",
            user="C:/Games/PC/BeamNG.drive",
            quit_on_close=True,
        )
        bng.open()

        # scenario = Scenario("driver_training", "lane detection")
        scenario = Scenario("west_coast_usa", "lane detection")

        # car = Vehicle("car", model="etk, part_config="gtx", licence="PYTHON", color=(0, 0, 0))
        car = Vehicle("car", model="scintilla", part_config="gtx", licence="PYTHON", color=(0, 0, 0))

        # scenario.add_vehicle(car, pos=(-270, 310, 53), cling=True)
        pos = (-737.9904854520128, -924.2250490613987, 163.45175623576097)
        rot_quad = (0.0038102599792182446, 0.03857719525694847, -0.5216534733772278, 0.8522763252258301)
        scenario.add_vehicle(car, pos=pos, rot_quat=rot_quad, cling=True)
        scenario.make(bng)

        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()

    with mss.mss() as sct:
        monitor = sct.monitors[2]
        while "Screen capturing":
            last_time = time.time()

            raw = np.array(sct.grab(monitor))
            raw = raw[50:-50, 0:-1]
            raw = cv.cvtColor(raw, cv.COLOR_BGRA2BGR)

            img = cv.resize(raw, (0, 0), fx=1, fy=1)
            img[: int(img.shape[0] / 2)] = 0

            mask = np.zeros_like(img)
            x = img.shape[1]
            y = img.shape[0]
            road_area = np.array(
                [
                    [
                        (x / 2 + 200, y / 2),
                        (x / 2 - 200, y / 2),
                        (100, y - 200),
                        (x - 100, y - 200),
                    ]
                ],
                np.int32,
            )
            cv.fillPoly(mask, road_area, (255, 255, 255))
            img = cv.bitwise_and(img, mask)

            mask = np.zeros_like(img)
            car_area = np.array(
                [
                    [
                        (0, img.shape[0] - 200),
                        (0, img.shape[0]),
                        (img.shape[1], img.shape[0]),
                        (img.shape[1], img.shape[0] - 200),
                    ]
                ],
                np.int32,
            )
            cv.fillPoly(mask, car_area, (255, 255, 255))
            mask = cv.bitwise_not(mask)
            img = cv.bitwise_and(img, mask)

            if debug:
                cv.addWeighted(raw, 0.5, img, 0.5, 0, raw)

            # only keep the yellow and white lanes
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([255, 80, 255])

            # make it more sensitive to white lines under a shadow (darker)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([255, 80, 255])

            mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
            mask_white = cv.inRange(hsv, lower_white, upper_white)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([70, 255, 255])
            mask_green = cv.inRange(hsv, lower_green, upper_green)
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

            # keep white and yellow, remove green and blue
            mask = cv.bitwise_or(mask_yellow, mask_white)
            mask = cv.bitwise_and(mask, cv.bitwise_not(mask_green))
            mask = cv.bitwise_and(mask, cv.bitwise_not(mask_blue))
            thresh = cv.bitwise_and(img, img, mask=mask)

            if debug:
                # cv.addWeighted(raw, 0.5, cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), 0.5, 0, raw)
                cv.addWeighted(raw, 0.5, thresh, 0.5, 0, raw)

            blur = cv.GaussianBlur(thresh, (5, 5), 0)
            canny = cv.Canny(blur, 100, 200)
            lines = cv.HoughLinesP(
                canny,
                1,
                np.pi / 180,
                100,
                np.array([]),
                minLineLength=40,
                maxLineGap=5,
            )

            left_lines = []
            right_lines = []
            if lines is not None:
                for line in lines:
                    slope = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
                    if abs(slope) < 0.2 or abs(slope) > 1.5:
                        continue
                    # if on left side and with a slope that is sloping towards the center
                    if line[0][0] < img.shape[1] / 2 and slope < 0:
                        left_lines.append(line)
                    elif line[0][0] > img.shape[1] / 2 and slope > 0:
                        right_lines.append(line)

            def closest_bunch(lines):
                if len(lines) == 0:
                    return None
                bunch = [lines[0]]
                for line in lines:
                    if abs(line[0][1] - bunch[0][0][1]) < 20:
                        bunch.append(line)
                return bunch

            left_lines = closest_bunch(left_lines)
            right_lines = closest_bunch(right_lines)

            # keep only lines which could be part of the lane
            def filter_lines(lines):
                if lines is None:
                    return None
                if len(lines) == 0:
                    return None
                # remove lines which are too far away from the center
                lines = [line for line in lines if abs(line[0][0] - img.shape[1] / 2) < 200]
                # remove lines which are too far away from each other
                lines = [
                    line
                    for line in lines
                    if abs(line[0][1] - lines[0][0][1]) < 20
                ]
                return lines
            # left_lines = filter_lines(left_lines)
            # right_lines = filter_lines(right_lines)


            def average_lines(lines):
                if lines is None:
                    return None
                if len(lines) == 0:
                    return None
                left_line = np.average(lines, axis=0)
                left_line = left_line.astype("int")
                return left_line

            left_line = average_lines(left_lines)
            right_line = average_lines(right_lines)

            def draw_lines(img, lines):
                if lines is not None:
                    for line in lines:
                        cv.line(
                            img,
                            (line[0][0], line[0][1]),
                            (line[0][2], line[0][3]),
                            (0, 255, 0),
                            3,
                        )

            draw_lines(raw, left_lines)
            draw_lines(raw, right_lines)


            def putTextOnRow(img, text, row):
                if not debug:
                    return
                cv.putText(
                    img,
                    text,
                    (10, 20 + row * 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            if left_line is not None and right_line is not None:
                # left_x = left_line[0][0]
                # right_x = right_line[0][0]
                # center_x = (left_x + right_x) / 2
                # offset = center_x - img.shape[1] / 2
                # better offset calculation which allows for more accurate steering
                left_x = left_line[0][0]
                right_x = right_line[0][2]
                center_x = (left_x + right_x) / 2
                offset = center_x - img.shape[1] / 2
                # print(offset)
                # print(left_x, right_x, center_x)
                throttle = 0.05
                # brake if we are too far off center
                if abs(offset) > 100:
                    throttle = 0.02
                if game:
                    car.sensors.poll()
                    vel = car.sensors["state"]["vel"]
                    putTextOnRow(raw, "Velocity: {}".format(vel), 1)
                    # if too fast, brake
                    # velocity is a triple (x, y, z)
                    # if [0]

                # show the offset on the screen
                putTextOnRow(raw, "Offset: {}".format(offset), 2)
                putTextOnRow(raw, "Throttle: {}".format(throttle), 3)


                if game:
                    car.control(throttle=throttle, steering=offset / 1000)

            putTextOnRow(raw, "FPS: {}".format(1 / (time.time() - last_time)), 0)
            cv.imshow("OpenCV/Numpy normal", raw)

            # Press "q" to quit
            if cv.waitKey(25) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break
