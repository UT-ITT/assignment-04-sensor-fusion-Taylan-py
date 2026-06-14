import cv2
import numpy as np
import argparse
import sys

# Global variables
points = []
img = None
original_img = None
WINDOW_NAME = 'Image Extractor'
RESULT_WINDOW_NAME = 'Warped Result'

def mouse_callback(event, x, y, flags, param):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            # Draw line to previous point
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
            # If 4 points are selected, close the polygon
            if len(points) == 4:
                cv2.line(img, points[3], points[0], (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, img)

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # Return the ordered coordinates
    return rect

def draw_legend(image):
    # Add a slightly dark background rectangle for better text readability
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (200, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Write the commands
    cv2.putText(image, "Commands:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Click 4 points", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "ESC - Clear", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "S - Save result", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Q - Quit", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    global points, img, original_img

    parser = argparse.ArgumentParser(description="Extract and warp a region of an image.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_image", help="Path to the output destination.")
    parser.add_argument("resolution", help="Target resolution in format WxH (e.g., 800x600).")

    args = parser.parse_args()

    try:
        width, height = map(int, args.resolution.lower().split('x'))
    except ValueError:
        print("Error: Resolution must be in the format WxH (e.g., 800x600).")
        sys.exit(1)

    original_img = cv2.imread(args.input_image)
    if original_img is None:
        print(f"Error: Could not load image {args.input_image}")
        sys.exit(1)

    img = original_img.copy()
    draw_legend(img)
    warped = None

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    print("Click 4 points on the image to extract a region.")
    print("Press ESC to clear points and start over.")
    print("Press 's' or 'S' to save the extracted image once 4 points are selected.")
    print("Press 'q' or 'Q' to quit the program.")

    while True:
        # Check if the main window was closed by the user (clicking the 'X' button)
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        cv2.imshow(WINDOW_NAME, img)
        
        if len(points) == 4 and warped is None:
            # Perform perspective transform
            pts = np.array(points, dtype="float32")
            rect = order_points(pts)

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(original_img, M, (width, height))

            cv2.imshow(RESULT_WINDOW_NAME, warped)

        key = cv2.waitKey(10) & 0xFF

        if key == 27: # ESC
            points = []
            img = original_img.copy()
            draw_legend(img)
            warped = None
            try:
                cv2.destroyWindow(RESULT_WINDOW_NAME)
            except cv2.error:
                pass
            print("Points cleared. Start over.")
        elif key == ord('s') or key == ord('S'):
            if warped is not None:
                cv2.imwrite(args.output_image, warped)
                print(f"Saved warped image to {args.output_image}")
            else:
                print("Please select 4 points first before saving.")
        elif key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
