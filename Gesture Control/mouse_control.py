import cv2
import mediapipe
import pyautogui

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)

# Set the desired frame rate to 60 frames per second
camera.set(cv2.CAP_PROP_FPS, 60)

# Set the box dimensions as a fraction of the screen size
box_width_fraction = 0.5  # Adjust this fraction as needed
box_height_fraction = 0.5  # Adjust this fraction as needed

# Calculate the actual box dimensions based on the screen size
box_width = int(screen_width * box_width_fraction)
box_height = int(screen_height * box_height_fraction)

# Move the box to the bottom right
x1 = screen_width - box_width
y1 = screen_height - box_height
x2 = x1 + box_width
y2 = y1 + box_height

# Variable to track whether the cursor is activated
cursor_activated = False

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            index_finger = one_hand_landmarks[8]
            thumb = one_hand_landmarks[4]

            x_index = int(index_finger.x * image_width)
            y_index = int(index_finger.y * image_height)

            x_thumb = int(thumb.x * image_width)
            y_thumb = int(thumb.y * image_height)

            # Check if the hand is open
            hand_open = y_thumb > y_index

            # Check if the hand is in the box and open
            if x1 < x_index < x2 and y1 < y_index < y2 and hand_open:
                if not cursor_activated:
                    cursor_activated = True
                    print("Trackpad Activated!")
            else:
                cursor_activated = False

            if cursor_activated:
                # Map hand coordinates to screen coordinates within the box
                mouse_x = int(screen_width * (x_index - x1) / box_width)
                mouse_y = int(screen_height * (y_index - y1) / box_height)
                cv2.circle(image, (x_index, y_index), 10, (0, 255, 255))
                pyautogui.moveTo(mouse_x, mouse_y)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Hand Recognition capture", image)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
