import imutils
from scipy.misc import imsave
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2,Image
from autocorrect import spell
import os


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    #print s
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    print rect

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
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

file=open('a.txt','wb')
file1=open('a1.txt','wb')
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
imsave("input.jpg",image)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray1", gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("gray", gray)
edged = cv2.Canny(gray, 50, 200)

#print "STEP 1: Edge Detection"
#cv2.imshow("Image", image)
#cv2.imshow("Edged", edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt=[]
for c in cnts:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

#print "STEP 2: Find contours of paper"
if len(screenCnt) == 4:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow("Outline", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # warped = warped.astype("uint8") * 255
    #cv2.imshow("normal", imutils.resize(warped, height=650))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 251, offset=10)
    warped = warped.astype("uint8") * 255

    #print "STEP 3: Apply perspective transform"
    #cv2.imshow("Original", imutils.resize(orig, height=650))
    #cv2.imshow("Scanned", imutils.resize(warped, height=650))
    imsave("test.jpg", imutils.resize(warped, height=650))
    image = cv2.imread('test.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("resized", im_bw)
    imsave('new.jpg',im_bw)
    image=Image.open('new.jpg')
    col,row=image.size
    cropped=im_bw[40:row - 40, 40:col - 40]
    imsave("input.jpg",cropped)
    #cv2.waitKey(0)



image = cv2.imread('input.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)#raw_input() need to change by imager = 50.0 / image.shape[1]
'''
dim = (50, int(image.shape[0] * r))
ss=int(image.shape[0])
mat=[[0 for x in xrange(50)]for y in xrange(int(image.shape[0]*r))]
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
(h, w) = image.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, 90, 1.0)
image= cv2.warpAffine(image, M, (w, h))
'''
mat=[[0 for x in xrange(image.shape[1])]for y in xrange(image.shape[0])]
(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imshow("resized", im_bw)
#cv2.imshow("cropped",im_bw[7:45,3:47])
cv2.waitKey(0)
imsave("new.jpg", im_bw)
image = Image.open("new.jpg")
col, row = image.size
# data = np.zeros((row*col, 5))
pixels = image.load()
image=im_bw
for i in xrange(row):
    for j in xrange(col):
        r = pixels[j, i]
        if int(r) < 200:
            mat[i][j]=0
            #print "0  ",
        else:
            mat[i][j]=255
            #print "255",
    #print
rowline=0
FRow,LRow=0,0
Frow,Lrow=0,0
while(rowline!=row):
    #print "rowline"
    PLcol=0
    set, set1, set2, set3 = 0, 0, 0, 0
    for i in xrange(FRow, row):
        if set == 1 and set2 == 0:
            set1 = 2
        for j in xrange(PLcol, col):
            if mat[i][j] == 0 and set == 0:
                set = 1
                FRow = i
            if mat[i][j] == 0 and set == 1:
                set1 = 3
        if set1 == 2:
            set3 = 1
            LRow = i
            break
    if set3 == 0:
        LRow = row
        break
    FRow-=1
    words=''
    avg=0
    scanline=0
    #cv2.imshow("s",image[FRow:LRow,0:col])
    Fcol, Lcol,Frow,Lrow= 0, 0,FRow,LRow
    while(scanline!=col):
        Frow,Lrow=FRow,LRow
        PFrow,PFcol,PLrow,PLcol=Frow, Fcol, Lrow, Lcol
        set,set1,set2,set3=0,0,0,0
        for i in xrange(Lcol,col):
            if set==1 and set2==0:
                set1=2
            for j in xrange(Frow,Lrow):
                if mat[j][i]==0 and set==0:
                    set=1
                    Fcol=i
                if mat[j][i]==0 and set==1 :
                    set1=3
            if set1==2:
                if PLcol!=0:
                    if avg==0:
                        avg=2*(Fcol-PLcol)
                    #print avg,Fcol-PLcol
                    if (Fcol-PLcol)>(avg):
                        words+=" "
                Lcol=i
                set3=1
                break
        if set3==0:
            Lcol=col
        set, set1, set2,set3 = 0, 0, 0,0
        for i in xrange(FRow,LRow+1):
            if set == 1 and set2 == 0:
                set1 = 2
            for j in xrange(PLcol,Lcol):
                if mat[i][j] == 0 and set == 0:
                    set = 1
                    Frow = i
                if mat[i][j] == 0 and set == 1:
                    set1 = 3
            if set1 == 2:
                set3=1
                Lrow = i
                break
        if set3==0:
            Lrow=row
        if Frow==PFrow and Fcol==PFcol and (Lrow==PLrow or Lrow==row) and (Lcol==PLcol or Lcol==col):
            scanline=col
            words+=' '
        else:
            scanline=Lcol
            cropped=image[Frow:Lrow,Fcol:Lcol]
            #cv2.imshow('Cropped',cropped)
            imsave('aa.jpg',cropped)
            #cv2.waitKey(0)

            dim=(50,50)
            cropped = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            #cv2.imshow('adjusted',cropped)
            imsave('aa.jpg',cropped)
            #cv2.waitKey(0)
            methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                       'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            s = ['A.jpg', 'B.jpg', 'C.jpg', 'D.jpg', 'E.jpg', 'F.jpg', 'G.jpg', 'H.jpg', 'I.jpg', 'J.jpg', 'K.jpg',
                 'L.jpg', 'M.jpg', 'N.jpg', 'O.jpg', 'P.jpg', 'Q.jpg', 'R.jpg', 'S.jpg', 'T.jpg', 'U.jpg', 'V.jpg',
                 'W.jpg', 'X.jpg', 'Y.jpg', 'Z.jpg','a.jpg','b.jpg','c.jpg','d.jpg','e.jpg','f.jpg','g.jpg','h.jpg',
                 'k.jpg','l.jpg','m.jpg','n.jpg','o.jpg','p.jpg','q.jpg','r.jpg','s.jpg','t.jpg','u.jpg','v.jpg',
                 'w.jpg','x.jpg','y.jpg','z.jpg']
            minValue=1.0
            #for method in methods:
            for i in s:
                template = cv2.imread(i,0)
                res = cv2.matchTemplate(cropped,template,eval(methods[5]))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if min_val<minValue:
                    minValue=min_val
                    letter=i[0]
            #print minValue
            words+=letter
    file.write(words+'\n')
    word=''
    sentence=''
    for i in words:
        if i==" ":
            sentence+=spell(word)
            sentence+=' '
            word=''
        else:
            word+=i
    file1.write(sentence+'\n')
    FRow=LRow+1
    rowline=LRow
file.close()
file1.close()
