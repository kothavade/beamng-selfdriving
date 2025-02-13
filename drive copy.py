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

        def remove_horizontal(lines):
            if lines is None:
                return None
            if len(lines) == 0:
                return None
            final_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:
                    final_lines.append(line)
            return final_lines

        def average_lines(lines):
            if lines is None:
                return None
            if len(lines) == 0:
                return None
            avg_line = np.average(lines, axis=0)
            avg_line = avg_line.astype("int")
            return avg_line

        def draw_lines(img, lines, color):
            if lines is not None:
                for line in lines:
                    cv.line(
                        img,
                        (line[0][0], line[0][1]),
                        (line[0][2], line[0][3]),
                        color,
                        6,
                    )

        def putTextOnRow(img, text, row):
            if not debug:
                return
            scale = 3
            cv.putText(
                img,
                text,
                (10 * scale, 20 * scale + row * 20 * scale),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5 * scale,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        x = monitor["width"] - 1
        y = monitor["height"] - 2 * 50

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

        car_area = np.array(
            [
                [
                    (0, y - 200),
                    (0, y),
                    (x, y),
                    (x, y - 200),
                ]
            ],
            np.int32,
        )

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        lower_white = np.array([0, 150, 0])
        upper_white = np.array([255, 255, 255])

        lower_green = np.array([0, 0, 0])
        upper_green = np.array([100, 255, 255])

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

            cv.fillPoly(mask, road_area, (255, 255, 255))
            img = cv.bitwise_and(img, mask)

            mask = np.zeros_like(img)

            cv.fillPoly(mask, car_area, (255, 255, 255))
            mask = cv.bitwise_not(mask)
            img = cv.bitwise_and(img, mask)

            if debug:
                cv.addWeighted(raw, 0.5, img, 0.5, 0, raw)

            img = cv.GaussianBlur(img, (5, 5), 0)

            # only keep the yellow and white lanes
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
            thresh_yellow = cv.bitwise_and(img, img, mask=mask_yellow)

            # use hsl to detect white lines
            hsl = cv.cvtColor(img, cv.COLOR_BGR2HLS)

            mask_white = cv.inRange(hsl, lower_white, upper_white)

            # remove greenish yellowish grass from the white mask
            mask_green = cv.inRange(hsv, lower_green, upper_green)
            mask_white = cv.bitwise_and(mask_white, mask_white, mask=cv.bitwise_not(mask_green))

            thresh_white = cv.bitwise_and(img, img, mask=mask_white)

            # grayscale and canny edge detection
            gray_yellow = cv.cvtColor(thresh_yellow, cv.COLOR_BGR2GRAY)
            gray_white = cv.cvtColor(thresh_white, cv.COLOR_BGR2GRAY)

            canny_yellow = cv.Canny(gray_yellow, 100, 200)
            canny_white = cv.Canny(gray_white, 100, 200)

            lines_yellow = cv.HoughLinesP(canny_yellow, 1, np.pi / 180, 50, maxLineGap=50)
            lines_white = cv.HoughLinesP(canny_white, 1, np.pi / 180, 50, maxLineGap=50)

            # left lines = all yellow lines on left side of the image with a slope < 0
            # right lines = all white lines on right side of the image with a slope > 0
            left_lines = []
            right_lines = []
            if lines_yellow is not None:
                for line in lines_yellow:
                    x1, y1, x2, y2 = line[0]
                    slope = np.where((x2-x1) != 0, (y2-y1) / (x2-x1), 1e-10)
                    if slope < 0 and x1 < x / 2 and x2 < x / 2:
                        left_lines.append(line)
            if lines_white is not None:
                for line in lines_white:
                    x1, y1, x2, y2 = line[0]
                    slope = np.where((x2-x1) != 0, (y2-y1) / (x2-x1), 1e-10)
                    if slope > 0 and x1 > x / 2 and x2 > x / 2:
                        right_lines.append(line)

            # remove horizontal lines
            # left_lines = remove_horizontal(left_lines)
            right_lines = remove_horizontal(right_lines)

            # average the lines
            left_line = average_lines(left_lines)
            right_line = average_lines(right_lines)

            # draw the lines
            draw_lines(raw, left_lines, (255, 0, 0))
            draw_lines(raw, right_lines, (0, 0, 255))

            # draw using the lines
            if left_line is not None and right_line is not None:
                left_x = left_line[0][0]
                right_x = right_line[0][2]
                center_x = (left_x + right_x) / 2
                offset = center_x - x / 2
                throttle = 0.05
                # brake if we are too far off center
                if abs(offset) > 100:
                    throttle = 0.02
                if game:
                    car.sensors.poll()
                    vel = car.sensors["state"]["vel"]
                    # for every item in vel list (x, y, z) round to 4 decimal places
                    vel = [round(v, 4) for v in vel]
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
            cv.imshow("lane detection", raw)

            # Press "q" to quit
            if cv.waitKey(25) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break
