from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time
import os
import cv2
import numpy as np
import threading
from datetime import datetime
from queue import Queue

class PanoramicCameraRecorder:
    def __init__(self):
        # Create directories for files
        os.makedirs("videos", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Initialize camera instances
        self.picam0 = Picamera2(0)  # First camera
        self.picam1 = Picamera2(1)  # Second camera
        
        # Set up configurations for both cameras - using same resolution for better stitching
        self.resolution = (1280, 720)
        
        self.config_cam0 = self.picam0.create_video_configuration(main={"size": self.resolution})
        self.config_cam1 = self.picam1.create_video_configuration(main={"size": self.resolution})
        
        self.picam0.configure(self.config_cam0)
        self.picam1.configure(self.config_cam1)
        
        # Get timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create encoder and output for each camera
        self.encoder0 = H264Encoder(10000000)  # 10 Mbps bitrate
        self.encoder1 = H264Encoder(10000000)
        
        self.temp_file0 = f"temp/cam0_{self.timestamp}.h264"
        self.temp_file1 = f"temp/cam1_{self.timestamp}.h264"
        
        self.output0 = FfmpegOutput(self.temp_file0)
        self.output1 = FfmpegOutput(self.temp_file1)
        
        # Final output file
        self.output_file = f"videos/panoramic_{self.timestamp}.mp4"
        
        # Queues for preview frames
        self.frame_queue0 = Queue(maxsize=2)
        self.frame_queue1 = Queue(maxsize=2)
        
        # Control flags
        self.recording = False
        self.stop_preview = False
        
        # Create feature detector for stitching
        self.feature_detector = cv2.SIFT_create()
        
        # Parameters for FLANN-based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Homography matrix storage
        self.homography_matrix = None
        self.calibrated = False
        
        # Store panorama dimensions for later use
        self.panorama_width = 0
        self.panorama_height = 0
        self.offset_x = 0
        self.offset_y = 0

    def calibrate_cameras(self):
        """Find the homography matrix between the two cameras for stitching"""
        print("Calibrating cameras for panoramic stitching...")
        
        # Start cameras briefly to capture calibration frames
        self.picam0.start()
        self.picam1.start()
        
        # Wait for cameras to settle
        time.sleep(2)
        
        # Capture frames
        frame0 = self.picam0.capture_array()
        frame1 = self.picam1.capture_array()
        
        # Stop cameras after calibration
        self.picam0.stop()
        self.picam1.stop()
        
        # Convert to grayscale for feature detection
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        keypoints0, descriptors0 = self.feature_detector.detectAndCompute(gray0, None)
        keypoints1, descriptors1 = self.feature_detector.detectAndCompute(gray1, None)
        
        # Match descriptors between the two images
        matches = self.flann.knnMatch(descriptors0, descriptors1, k=2)
        
        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) >= 4:
            # Get the matching points
            src_pts = np.float32([keypoints0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            self.homography_matrix = H
            
            # Calculate panorama dimensions
            h, w = frame0.shape[:2]
            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Combined all corners to find the size of the panorama
            all_corners = np.concatenate((transformed_corners, np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)))
            
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            # Translation matrix to shift to positive coordinates
            self.offset_x = abs(min(0, xmin))
            self.offset_y = abs(min(0, ymin))
            
            # Final panorama dimensions
            self.panorama_width = xmax + self.offset_x
            self.panorama_height = ymax + self.offset_y
            
            print(f"Calibration successful: Panorama dimensions will be {self.panorama_width}x{self.panorama_height}")
            self.calibrated = True
            
            # Show preview of the calibrated stitching
            translation_matrix = np.array([[1, 0, self.offset_x], [0, 1, self.offset_y], [0, 0, 1]])
            warped_img = cv2.warpPerspective(frame0, translation_matrix.dot(self.homography_matrix), 
                                            (self.panorama_width, self.panorama_height))
            
            # Create a mask for blending
            mask = np.zeros((self.panorama_height, self.panorama_width), dtype=np.uint8)
            cv2.warpPerspective(np.ones(frame0.shape[:2], dtype=np.uint8), 
                               translation_matrix.dot(self.homography_matrix), 
                               (self.panorama_width, self.panorama_height), 
                               dst=mask, flags=cv2.INTER_NEAREST)
            
            # Place second image in the panorama
            panorama = np.zeros((self.panorama_height, self.panorama_width, 3), dtype=np.uint8)
            panorama[self.offset_y:self.offset_y+frame1.shape[0], self.offset_x:self.offset_x+frame1.shape[1]] = frame1
            
            # Blend the warped image
            panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask))
            panorama = cv2.add(panorama, warped_img)
            
            # Display calibration result
            cv2.imshow("Calibration Result", cv2.resize(panorama, (1280, 720)))
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
            
            return True
        else:
            print("Calibration failed: Not enough matching features between cameras")
            print("Falling back to side-by-side mode")
            return False

    def start_recording(self):
        print(f"Starting recording to {self.output_file}")
        
        # Try to calibrate the cameras first
        self.calibrate_cameras()
        
        # Start both cameras
        self.picam0.start_encoder(self.encoder0, self.output0)
        self.picam1.start_encoder(self.encoder1, self.output1)
        self.picam0.start()
        self.picam1.start()
        
        self.recording = True
        
        # Start preview in a separate thread
        preview_thread = threading.Thread(target=self.show_preview)
        preview_thread.start()
        
        # Start frame capture threads
        cam0_thread = threading.Thread(target=self.capture_frames, args=(self.picam0, self.frame_queue0))
        cam1_thread = threading.Thread(target=self.capture_frames, args=(self.picam1, self.frame_queue1))
        cam0_thread.start()
        cam1_thread.start()
        
        try:
            while self.recording:
                time.sleep(0.1)  # Check for keyboard interrupt periodically
        except KeyboardInterrupt:
            print("Recording interrupted by user")
        finally:
            self.stop_recording()
            preview_thread.join()
            cam0_thread.join()
            cam1_thread.join()
    
    def capture_frames(self, camera, queue):
        """Capture frames from camera for preview"""
        while self.recording and not self.stop_preview:
            frame = camera.capture_array()
            # Keep queue fresh (discard old frames if needed)
            if queue.full():
                try:
                    queue.get_nowait()
                except:
                    pass
            queue.put(frame)
            time.sleep(0.033)  # ~30fps
    
    def stitch_frames(self, frame0, frame1):
        """Stitch two frames into a panoramic view"""
        if not self.calibrated:
            # If calibration failed, just return side-by-side images
            return np.hstack((frame0, frame1))
        
        # Create translation matrix to account for negative coordinates
        translation_matrix = np.array([[1, 0, self.offset_x], [0, 1, self.offset_y], [0, 0, 1]])
        
        # Warp first frame according to the homography
        warped_img = cv2.warpPerspective(frame0, translation_matrix.dot(self.homography_matrix), 
                                         (self.panorama_width, self.panorama_height))
        
        # Create a mask for the warped image
        mask = np.zeros((self.panorama_height, self.panorama_width), dtype=np.uint8)
        cv2.warpPerspective(np.ones(frame0.shape[:2], dtype=np.uint8), 
                           translation_matrix.dot(self.homography_matrix), 
                           (self.panorama_width, self.panorama_height), 
                           dst=mask, flags=cv2.INTER_NEAREST)
        
        # Place second image in the panorama
        panorama = np.zeros((self.panorama_height, self.panorama_width, 3), dtype=np.uint8)
        h1, w1 = frame1.shape[:2]
        panorama[self.offset_y:self.offset_y+h1, self.offset_x:self.offset_x+w1] = frame1
        
        # Blend the images
        # First, create region for second image without overlap from first
        panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask))
        # Then add the warped first image
        panorama = cv2.add(panorama, warped_img)
        
        return panorama
    
    def show_preview(self):
        """Show stitched preview of both cameras"""
        while self.recording and not self.stop_preview:
            try:
                if not self.frame_queue0.empty() and not self.frame_queue1.empty():
                    frame0 = self.frame_queue0.get()
                    frame1 = self.frame_queue1.get()
                    
                    # Stitch frames
                    panorama = self.stitch_frames(frame0, frame1)
                    
                    # Resize for display if too large
                    if panorama.shape[1] > 1600:
                        scale_factor = 1600 / panorama.shape[1]
                        panorama = cv2.resize(panorama, (1600, int(panorama.shape[0] * scale_factor)))
                    
                    # Add timestamp overlay
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(panorama, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Show preview
                    cv2.imshow("Panoramic Preview", panorama)
                    
                    # Check for 'q' key to stop recording
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.recording = False
            
            except Exception as e:
                print(f"Preview error: {e}")
            
            time.sleep(0.01)
        
        cv2.destroyAllWindows()
    
    def stop_recording(self):
        """Stop recording and create the final panoramic video"""
        print("Stopping recording...")
        self.recording = False
        self.stop_preview = True
        
        # Stop camera recording
        self.picam0.stop()
        self.picam1.stop()
        self.picam0.stop_encoder()
        self.picam1.stop_encoder()

        # Create the panoramic video using the recorded files
        self.create_panoramic_video()
        
        # Clean up
        print("Cleaning up temporary files...")
        os.remove(self.temp_file0)
        os.remove(self.temp_file1)
        print(f"Recording saved to {self.output_file}")
    
    def create_panoramic_video(self):
        """Process the individual camera recordings to create a panoramic video"""
        import subprocess
        
        # Convert h264 temp files to mp4 first
        mp4_file0 = f"temp/cam0_{self.timestamp}.mp4"
        mp4_file1 = f"temp/cam1_{self.timestamp}.mp4"
        
        print("Converting raw camera files to MP4...")
        subprocess.call(['ffmpeg', '-i', self.temp_file0, '-c:v', 'copy', mp4_file0], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.call(['ffmpeg', '-i', self.temp_file1, '-c:v', 'copy', mp4_file1],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        if self.calibrated:
            print("Creating panoramic video (this may take some time)...")
            # Process each frame to create the panoramic video
            self.process_videos_to_panorama(mp4_file0, mp4_file1, self.output_file)
        else:
            print("Creating side-by-side video...")
            # If calibration failed, just create side-by-side video using ffmpeg
            cmd = [
                'ffmpeg',
                '-i', mp4_file0,
                '-i', mp4_file1,
                '-filter_complex', '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]',
                '-map', '[vid]',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast',
                self.output_file
            ]
            subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Clean up temporary mp4 files
        os.remove(mp4_file0)
        os.remove(mp4_file1)
    
    def process_videos_to_panorama(self, video1_path, video2_path, output_path):
        """Process each frame of the videos to create a panoramic video"""
        # Open video files
        cap0 = cv2.VideoCapture(video1_path)
        cap1 = cv2.VideoCapture(video2_path)
        
        # Check if videos opened successfully
        if not cap0.isOpened() or not cap1.isOpened():
            print("Error: Could not open input videos")
            return
        
        # Get video properties
        fps = cap0.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.panorama_width, self.panorama_height))
        
        frame_count = 0
        total_frames = min(int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break  # End of one of the videos
            
            # Stitch frames
            panorama = self.stitch_frames(frame0, frame1)
            
            # Write to output video
            out.write(panorama)
            
            # Show progress
            frame_count += 1
            if frame_count % 30 == 0:
                percent = int((frame_count / total_frames) * 100)
                print(f"Processing: {percent}% complete ({frame_count}/{total_frames} frames)")
        
        # Release everything
        cap0.release()
        cap1.release()
        out.release()


def main():
    print("Panoramic Camera Recording with Picamera2")
    print("------------------------------------------")
    print("Press 'q' in the preview window to stop recording")
    
    recorder = PanoramicCameraRecorder()
    recorder.start_recording()


if __name__ == "__main__":
    main()