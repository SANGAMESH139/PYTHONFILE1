import cv2
import numpy as np

def resize_image(image, width):
    """Resize the image to a given width while maintaining aspect ratio."""
    ratio = width / image.shape[1]
    dim = (width, int(image.shape[0] * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def find_edges(image):
    """Convert the image to grayscale, blur it, and find edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    return edges

def find_contours(edges):
    """Find contours in the edged image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return contours

def find_document_contour(contours):
    """Find the contour that represents the document."""
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
    return None

def perspective_transform(image, contour):
    """Apply a perspective transform to obtain a top-down view of the document."""
    contour = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = contour.sum(axis=1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def enhance_image(image):
    """Convert the image to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return enhanced

def main():
    # Load the image
    image = cv2.imread('document.jpg')

    # Resize the image to make processing faster
    resized_image = resize_image(image, 500)

    # Find edges in the image
    edges = find_edges(resized_image)

    # Find contours in the edged image
    contours = find_contours(edges)

    # Find the document contour
    document_contour = find_document_contour(contours)

    if document_contour is None:
        print("Document contour not found.")
        return

    # Perform perspective transform
    scanned = perspective_transform(image, document_contour)

    # Enhance the scanned document
    enhanced_scanned = enhance_image(scanned)

    # Show the original and scanned images
    cv2.imshow("Original", resize_image(image, 500))
    cv2.imshow("Scanned", resize_image(enhanced_scanned, 500))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
