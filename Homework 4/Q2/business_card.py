import cv2
import pytesseract
import webbrowser
 
pytesseract.pytesseract.tesseract_cmd = "C:\\Computer Vision\\Homework 4\\Homework 4\\tesseract.exe"

picture = cv2.imread("kirstyn_card.png")
 
gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
amplify = cv2.dilate(thresh1, rect_kernel, iterations = 1)
 
curves, rank = cv2.findContours(amplify, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
image2 = picture.copy()
 
index = open("business_data.txt", "w+")
index.write("")
index.close()

for cnt in curves:
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = image2[y:y + h, x:x + w]

    index = open("business_data.txt", "a")

    text = pytesseract.image_to_string(cropped)

    index.write(text)
    index.close

catch = cv2.QRCodeDetector()
url_data, bbox, straight_qrcode = catch.detectAndDecode(picture)
if url_data:
    index = open("cv_data.txt", "w+")
    index.write(url_data)
    index.close()
    webbrowser.open(url_data)