import cv2
import numpy as np

def align_rectangle_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny edge detection with adjusted parameters
    edges = cv2.Canny(blur, 50, 150)

    # Find contours of the rectangles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out rectangles with line inside and approximate to 4 vertices
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles.append(approx)

    # Filter rectangles based on aspect ratio (close to 1, roughly square)
    rectangles = [rect for rect in rectangles if cv2.isContourConvex(rect)]
    rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)

    # Draw rectangles on the original image
    for i, rectangle in enumerate(rectangles[:4]):  # Draw the largest four rectangles
        cv2.drawContours(image, [rectangle], -1, (0, 0, 255), 2)
        rect_points = rectangle.reshape(4, 2).astype(np.float32)

        # Define the target points for perspective transformation
        target_points = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float32)

        # Calculate the perspective transformation matrix
        M = cv2.getPerspectiveTransform(rect_points, target_points)

        # Apply the perspective transformation to the rectangle to align it
        aligned_rectangle = cv2.warpPerspective(image, M, (300, 300))
        
        # Display the aligned rectangle
        cv2.imshow(f"Aligned Rectangle {i+1}", aligned_rectangle)

    # Display the original image with rectangles drawn on it
    cv2.imshow("Original Image with Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read the image
image = cv2.imread("task2.png")

# Align and display rectangles on the original image
align_rectangle_image(image)
