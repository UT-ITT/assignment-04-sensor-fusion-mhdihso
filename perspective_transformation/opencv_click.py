import argparse
import cv2
import numpy as np
import sys

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def main():
    parser = argparse.ArgumentParser(description="Extract and warp a region from an image.")
    parser.add_argument("input", help="Path to the input image.")
    parser.add_argument("output", help="Path to save the output image.")
    parser.add_argument("width", type=int, help="Target width of the output image.")
    parser.add_argument("height", type=int, help="Target height of the output image.")
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load image at {args.input}")
        sys.exit(1)
        
    clone = img.copy()
    points = []
    
    window_name = "Image Extractor - Select 4 points (ESC to reset, Q to quit)"
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                # Draw lines between points for better visual feedback
                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (0, 255, 0), 2)
                if len(points) == 4:
                    cv2.line(clone, points[-1], points[0], (0, 255, 0), 2)
                cv2.imshow(window_name, clone)
                
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Click 4 points in the image (e.g., Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
    
    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27: # ESC
            points = []
            clone = img.copy()
            cv2.imshow(window_name, clone)
            
        elif len(points) == 4:
            src_pts = np.array(points, dtype=np.float32)
            src_pts = order_points(src_pts)
            dst_pts = np.array([
                [0, 0],
                [args.width - 1, 0],
                [args.width - 1, args.height - 1],
                [0, args.height - 1]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (args.width, args.height))
            
            result_window = "Warped Result (S to save, ESC to discard)"
            cv2.imshow(result_window, warped)
            
            while True:
                res_key = cv2.waitKey(1) & 0xFF
                if res_key == 27: # ESC to discard and start over
                    cv2.destroyWindow(result_window)
                    points = []
                    clone = img.copy()
                    cv2.imshow(window_name, clone)
                    break
                elif res_key == ord('s') or res_key == ord('S'):
                    cv2.imwrite(args.output, warped)
                    print(f"Saved warped image to {args.output}")
                    cv2.destroyAllWindows()
                    return
                elif res_key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                    
        elif key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
