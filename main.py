import cv2
import face_recognition

# Load images
imgElon = face_recognition.load_image_file(r'C:\Users\Arnava\Downloads\Auto_Att-Opencv\ImageBasic\W.jpg')
if imgElon is None:
    raise Exception("Reference image not found!")

imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)  # Convert image to RGB

# Load the test image
imgTest = face_recognition.load_image_file(r'C:\Users\Arnava\Downloads\Auto_Att-Opencv\ImageBasic\Elon_Test.jpg')
if imgTest is None:
    raise Exception("Test image not found!")

imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Find face locations and encodings for the reference image
faceLocElon = face_recognition.face_locations(imgElon)
encodeElon = face_recognition.face_encodings(imgElon)

if faceLocElon and encodeElon:
    faceLocElon = faceLocElon[0]  # Get the first face location
    cv2.rectangle(imgElon, (faceLocElon[3], faceLocElon[0]), (faceLocElon[1], faceLocElon[2]), (255, 0, 255), 2)

# Find face locations and encodings for the test image
faceLocTest = face_recognition.face_locations(imgTest)
encodeTest = face_recognition.face_encodings(imgTest)

if faceLocTest and encodeTest:
    faceLocTest = faceLocTest[0]  # Get the first face location
    cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

    # Compare faces and calculate distance
    results = face_recognition.compare_faces([encodeElon[0]], encodeTest[0])  # Check if the faces match
    faceDis = face_recognition.face_distance([encodeElon[0]], encodeTest[0])  # Calculate the distance
    print(results, faceDis)
    cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

# Display the images
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()  # Close the image windows properly