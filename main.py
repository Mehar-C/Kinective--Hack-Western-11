import cv2
import random
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the width and height of the camera feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up window
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Feed", width, height)
cv2.setWindowProperty("Webcam Feed", cv2.WND_PROP_TOPMOST, 1)

# Colors for circles and text
BURGUNDY = (31, 2, 141)  # #8D021F in BGR
SEA_GREEN = (87, 139, 46)  # #2E8B57 in BGR
YELLOW = (0, 255, 255)  # Highlight color for the user's score

# Duration of the game
duration = 30

# Scoreboard
scoreboard = []

# Function to check if two circles overlap
def circles_overlap(pos1, pos2, threshold=50):
    return np.linalg.norm(np.array(pos1) - np.array(pos2)) < threshold

# Function to display the scoreboard
def display_scoreboard(scoreboard, current_score):
    scoreboard.append(current_score)
    scoreboard = sorted(scoreboard, reverse=True)[:5]  # Keep top 5 scores

    # Display the scoreboard
    while True:
        dark_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dark_frame, "Scoreboard", (width // 2 - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, SEA_GREEN, 3)

        for i, score in enumerate(scoreboard):
            color = YELLOW if score == current_score else SEA_GREEN
            cv2.putText(dark_frame, f"{i + 1}. {score} points", (width // 2 - 200, 200 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(dark_frame, "Press Spacebar to Play Again", (width // 2 - 250, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, SEA_GREEN, 2)
        cv2.imshow("Webcam Feed", dark_frame)

        key = cv2.waitKey(1)
        if key == ord(' '):  # Spacebar to play again
            return scoreboard
        elif key == ord('q'):  # Quit the program
            return None

# Main game loop
while True:
    # Countdown screen with dark background
    for i in range(3, 0, -1):
        dark_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dark_frame, f"Get Ready! {i}", (width // 2 - 200, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, SEA_GREEN, 3)
        cv2.imshow("Webcam Feed", dark_frame)
        cv2.waitKey(1000)

    # Initial positions for Pair A circles
    red_circle_position = (random.randint(50, width - 50), random.randint(50, height - 50))
    blue_circle_position = (random.randint(50, width - 50), random.randint(50, height - 50))

    # Flags and counters
    red_matched = False
    blue_matched = False
    match_count = 0

    # Start time
    start_time = time.time()

    # Mediapipe Pose tracking
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            # Draw Pair A circles
            if not red_matched:
                cv2.circle(frame, red_circle_position, 30, BURGUNDY, -1)
            if not blue_matched:
                cv2.circle(frame, blue_circle_position, 30, SEA_GREEN, -1)

            if result.pose_landmarks:
                left_hand = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
                right_hand = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
                left_hand_coords = (int(left_hand.x * width), int(left_hand.y * height))
                right_hand_coords = (int(right_hand.x * width), int(right_hand.y * height))

                cv2.circle(frame, left_hand_coords, 20, BURGUNDY, -1)
                cv2.circle(frame, right_hand_coords, 20, SEA_GREEN, -1)

                if not red_matched and circles_overlap(left_hand_coords, red_circle_position):
                    red_matched = True
                    match_count += 1

                if not blue_matched and circles_overlap(right_hand_coords, blue_circle_position):
                    blue_matched = True
                    match_count += 1

                if red_matched and blue_matched:
                    red_circle_position = (random.randint(50, width - 50), random.randint(50, height - 50))
                    blue_circle_position = (random.randint(50, width - 50), random.randint(50, height - 50))
                    red_matched = False
                    blue_matched = False

            elapsed_time = time.time() - start_time
            remaining_time = max(0, int(duration - elapsed_time))

            cv2.putText(frame, f"Time: {remaining_time}s", (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, SEA_GREEN, 2)
            cv2.putText(frame, f"Matches: {match_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, SEA_GREEN, 2)

            cv2.imshow("Webcam Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or remaining_time <= 0:
                break

    # Show the scoreboard
    scoreboard = display_scoreboard(scoreboard, match_count)
    if scoreboard is None:  # Exit if the user presses 'q'
        break

cap.release()
cv2.destroyAllWindows()
