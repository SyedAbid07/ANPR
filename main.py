import cv2
import pytesseract
import os

# Set Tesseract Path (For Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  

# Frame Settings
frameWidth = 1000  
frameHeight = 480  

# Load Haarcascade for Number Plate Detection
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

if plateCascade.empty():
    print("Error: Haarcascade file not found!")
    exit()

# Minimum area for a valid detection
minArea = 500

# Create IMAGES folder if it doesn't exist
if not os.path.exists("IMAGES"):
    os.makedirs("IMAGES")

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  
cap.set(4, frameHeight)  
cap.set(10, 150)  
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame from webcam.")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgRoi = img[y:y + h, x:x + w]

            # Convert to grayscale and apply threshold to improve OCR
            imgRoiGray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
            _, imgRoiThresh = cv2.threshold(imgRoiGray, 150, 255, cv2.THRESH_BINARY)

            # Extract text from the number plate using Tesseract
            plateText = pytesseract.image_to_string(imgRoiThresh, config="--psm 7")

            # Display the detected text
            cv2.putText(img, plateText.strip(), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            print("Detected Number Plate:", plateText.strip())

            cv2.imshow("Number Plate", imgRoi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save Image when 's' key is pressed
        cv2.imwrite(f"./IMAGES/{count}.jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
    elif key == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
