import cv2
import os
import time

def capture_from_webcam(num_samples=200):
    # 1. Setup Folder Structure
    # Path: custom_data/images/img_x.jpg
    base_dir = "custom_data"
    class_dir = os.path.join(base_dir, "images")
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
    # 2. Open Webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- WEBCAM DATA COLLECTOR ---")
    print(f"Goal: Capture {num_samples} images.")
    print("Press 'c' to Capture a frame.")
    print("Press 'a' to Auto-Capture (Fast mode).")
    print("Press 'q' to Quit.")

    count = 0
    auto_mode = False

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 64x64 immediately (saves space and processing time)
        # GANs need square images
        frame_resized = cv2.resize(frame, (64, 64))

        # Show the live feed (Original size for viewing)
        cv2.imshow('Data Collector - Press C or A', frame)

        key = cv2.waitKey(1) & 0xFF

        # --- CONTROLS ---
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"Auto Mode: {auto_mode}")
        elif key == ord('c') or auto_mode:
            # Save the frame
            img_name = os.path.join(class_dir, f"webcam_img_{count}.jpg")
            cv2.imwrite(img_name, frame_resized)
            count += 1
            print(f"Captured {count}/{num_samples}")
            
            # Small delay in auto mode to vary the angles
            if auto_mode:
                time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Success! {count} images saved to '{class_dir}'.")
    print("You can now run 'python train.py'")

if __name__ == "__main__":
    capture_from_webcam(num_samples=200)