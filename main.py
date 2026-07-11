import cv2
import numpy as np
from ultralytics import YOLO

model_path = './assets/models/segmentation.pt'
model = YOLO(model_path)

lap_camera = cv2.VideoCapture(0)

while True:
    success, img = lap_camera.read()
    if not success:
        print("Fail to read the next frame")
        break  # Better to break if the camera fails
    
    # Get original image dimensions dynamically
    H, W, _ = img.shape

    # Run YOLO inference
    results = model(img, stream=True)
    
    # Each has a mask and a bounding box, use zip function 
    
    for result in results:
        # Check if any masks were detected to avoid NoneType errors
        if result.masks is not None:
            # result.masks.xy sets of point to draw the boundaries for polygons
            # this will be used for collision detection
            
            # so bounding box for identity
            # 
            print(result.masks.xy)
            for box, mask_tensor in zip(result.boxes, result.masks.data):
                # -------- Bounding Box --------
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                color = [127,127,127]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # -------- Mask ------------
                mask = mask_tensor.cpu().numpy() * 255
                mask = mask.astype('uint8')

                # Resize mask to match the original image size
                mask_resized = cv2.resize(mask, (W, H))

                # the reason is because the mask itself is single channel but our image is 
                # 3 channels
                color_mask = np.zeros_like(img)
                color_mask[mask_resized > 0.5] = [0, 255, 0]
     
                
                # Display the mask window
                cv2.addWeighted(color_mask, 0.4,img,1,0,img)
            
    
    # Display the original camera feed
    cv2.imshow("Webcam Feed", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows
lap_camera.release()
cv2.destroyAllWindows()