import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize Pygame and MediaPipe
pygame.init()
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Pygame window setup
screen_width, screen_height = 1200, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture-Controlled T-Rex Game")

# Define colors
WHITE = (255, 255, 255)

# Load images
def load_image(path, size):
    image = pygame.image.load(path)
    return pygame.transform.scale(image, size)

dino_run = load_image("dino_run.png", (70, 70))
dino_jump = load_image("dino_run.png", (70, 70))
dino_duck = load_image("dino_duck.png", (100, 50))
cactus_img = load_image("cactus.png", (50, 50))
bird_img = load_image("bird.png", (70, 50))

# Game variables
dino_x, dino_y = 100, screen_height - 150
dino_vel_y = 0
gravity = 1
floor_height = screen_height - 70
obstacles = []
score = 0
game_over = False
running = True
jumping = False

# Clock for controlling frame rate
clock = pygame.time.Clock()

def handle_obstacles():
    global score, game_over
    if not obstacles or obstacles[-1][1] < screen_width - 300:
        obstacle_type = random.choice(['cactus', 'bird'])
        obstacle_img = cactus_img if obstacle_type == 'cactus' else bird_img
        obstacle_y = floor_height - obstacle_img.get_height() if obstacle_type == 'cactus' else floor_height - 100 - obstacle_img.get_height()
        obstacles.append([obstacle_img, screen_width, obstacle_y])

    for obstacle in obstacles[:]:
        if not game_over:
            obstacle[1] -= 10  # Move obstacle left
            if obstacle[1] < -obstacle[0].get_width():
                obstacles.remove(obstacle)
                score += 10  # Increase score when an obstacle is passed
            # Check collision
            if pygame.Rect(obstacle[1], obstacle[2], obstacle[0].get_width(), obstacle[0].get_height()).colliderect(pygame.Rect(dino_x, dino_y, 70, 70)):
                game_over = True

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read camera frame
    ret, frame = cap.read()
    if not ret:
        continue  # Skip the frame if capture failed

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)  # Horizontal flip for mirror effect
    results = hands.process(frame)

    # Draw hand landmarks on the RGB frame before converting it to Pygame surface
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Convert the modified frame (with landmarks) to Pygame surface
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for Pygame compatibility
    frame = np.rot90(frame)  # Rotate frame for correct orientation
    frame_surface = pygame.surfarray.make_surface(frame)
    frame_surface = pygame.transform.scale(frame_surface, (300, 200))  # Resize to fit corner
    screen.blit(frame_surface, (screen_width - 300, 0))

    # Process hand gestures
    if results.multi_hand_landmarks and not game_over:
        for hand_landmarks in results.multi_hand_landmarks:
            all_fingers_up = all(hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in [8, 12, 16, 20])  # Index, middle, ring, pinky
            thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            if all_fingers_up and thumb_up:
                if not jumping:
                    jumping = True
                    dino_vel_y = -20  # Start jump
            elif not all_fingers_up:
                dino_y = floor_height  # Duck if not all fingers are up
                jumping = False

    if jumping:
        dino_y += dino_vel_y
        dino_vel_y += gravity
        if dino_y >= floor_height:
            dino_y = floor_height
            jumping = False

    # Update dinosaur position
    dino_rect = pygame.Rect(dino_x, dino_y, 70, 70)
    screen.blit(dino_jump if jumping else dino_run, (dino_x, dino_y))

    # Handle obstacles
    handle_obstacles()

    # Draw obstacles
    for obstacle in obstacles:
        screen.blit(obstacle[0], (obstacle[1], obstacle[2]))

    # Display score
    score_text = pygame.font.Font(None, 36).render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))
    
    if game_over:
        game_over_text = pygame.font.Font(None, 72).render("Game Over! Press 'R' to Restart", True, (0, 0, 0))
        screen.blit(game_over_text, (screen_width // 2 - game_over_text.get_width() // 2, screen_height // 2))
        pygame.display.flip()  # Update display to show game over text
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    waiting = False
                if event.type == pygame.QUIT:
                    pygame.quit()
                    cv2.destroyAllWindows()
                    cap.release()
                    exit()

        # Reset game to start over
        obstacles = []
        score = 0
        game_over = False
        dino_y = floor_height
        jumping = False
        continue

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
cv2.destroyAllWindows()
cap.release()
