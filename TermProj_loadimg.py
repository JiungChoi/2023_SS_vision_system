from tkinter import *
from tkinter import filedialog
import cv2

root = Tk()
path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
img1 = cv2.imread(path)
root.withdraw()

root = Tk()
path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
img2 = cv2.imread(path)
root.withdraw()

cv2.imshow("test1", img1)
cv2.imshow("test2", img2)


cv2.waitKey(0)
cv2.destroyAllWindows()
