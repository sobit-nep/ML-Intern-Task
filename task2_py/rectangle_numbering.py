import cv2
import numpy as np

def get_line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Read the image
image = cv2.imread("task2.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection with adjusted parameters
edges = cv2.Canny(blur, 50, 150)

# Find contours of the rectangles
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out rectangles with line inside
rectangles = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        rectangles.append(contour)

# Apply Hough Line Transform to detect lines inside each rectangle
numbered_rectangles = image.copy()
names = ["1", "2", "3", "4"]  # You can add more names if needed
for i, rectangle_contour in enumerate(rectangles, 1):
    # Create a mask for the current rectangle
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [rectangle_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to the edge-detected image
    masked_edges = cv2.bitwise_and(edges, mask)

    # Filter out lines that are not inside the current rectangle
    lines = cv2.HoughLinesP(masked_edges, 0.688, np.pi / 180, threshold=45, minLineLength=56, maxLineGap=4)
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if cv2.pointPolygonTest(rectangle_contour, (float(x1), float(y1)), False) >= 0 and \
           cv2.pointPolygonTest(rectangle_contour, (float(x2), float(y2)), False) >= 0:
            filtered_lines.append(line)

    # Sort the lines based on their length
    sorted_lines = sorted(filtered_lines, key=get_line_length)

    # Draw the shortest line on the original image and label it with the rectangle name
    shortest_line = sorted_lines[0]
    x1, y1, x2, y2 = shortest_line[0]
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    name = names[i - 1] if i <= len(names) else str(i)
    cv2.putText(numbered_rectangles, name, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.line(numbered_rectangles, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show the image with names assigned to each rectangle
cv2.imshow("Numbered Rectangles", numbered_rectangles)
cv2.waitKey(0)
cv2.destroyAllWindows()