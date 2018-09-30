import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
def showfig(image, ucmap):
    #There is a difference in pixel ordering in OpenCV and Matplotlib.
    #OpenCV follows BGR order, while matplotlib follows RGB order.
    if len(image.shape)==3 :
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
    imgplot=plt.imshow(image, ucmap)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
image_path = ""
carsample=cv2.imread(image_path)
showfig(carsample,None)
plt.show()

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

bwtest = cv2.imread(image_path)
imgwidth = bwtest.shape[1]
imgheight = bwtest.shape[0]
imgarea = imgwidth * imgheight
lab = cv2.cvtColor(bwtest, cv2.COLOR_BGR2Lab)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab_planes[1] = clahe.apply(lab_planes[1])
lab_planes[2] = clahe.apply(lab_planes[2])
lab = cv2.merge(lab_planes)
bwtest = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
showfig(bwtest,None)
bwtest = adjust_gamma(bwtest,1)
bwtest = cv2.cvtColor(bwtest, cv2.COLOR_BGR2GRAY)
bwtest = cv2.GaussianBlur(bwtest,(5,5),0)
_,bwtest = cv2.threshold(bwtest,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
wpp = (cv2.countNonZero(bwtest) / imgarea) * 100
mode = cv2.THRESH_BINARY
flag = 0
print(wpp)
if(wpp > 40):
    if(wpp > 48):
        gammaval = 0.65
        #flag = 1
    else:
        gammaval = 0.85
elif(wpp<40):
    if(wpp<36):
        gammaval = 0.5
    else:
        gammaval = 1.5

def preprocess(carsample,gammaval):
    carsample = adjust_gamma(carsample,gammaval)
    lab = cv2.cvtColor(carsample, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(5,5))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_planes[1] = clahe.apply(lab_planes[1])
    lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    carsample = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    showfig(carsample,None)
    plt.show()
    gray_carsample = cv2.cvtColor(carsample, cv2.COLOR_BGR2GRAY)
    showfig(gray_carsample,plt.get_cmap('gray'))
    plt.show()
    blur = cv2.GaussianBlur(gray_carsample,(5,5),0)
    _, th3= cv2.threshold(blur,0,255,mode+cv2.THRESH_OTSU)
    showfig(th3,plt.get_cmap('gray'))
    """  if(flag == 1):
        se=cv2.getStructuringElement(cv2.MORPH_RECT,(2,1))
        showfig(se, plt.get_cmap('gray'))
        th3 = cv2.erode(th3,se,iterations = 5)
        showfig(th3,plt.get_cmap('gray'))    """
    

    plt.show()
    return th3
#se=cv2.getStructuringElement(cv2.MORPH_RECT,(40,60))
#showfig(se, plt.get_cmap('gray'))
#opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN,se)
th3 = preprocess(carsample,gammaval)
_,contours, _=cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    rect=cv2.minAreaRect(cnt)  
    box=cv2.boxPoints(rect)
    box=np.int0(box)  
    cv2.drawContours(carsample, [box], 0, (0,255,0),2)
showfig(carsample, None)
plt.show()

def validate(cnt):    
    rect=cv2.minAreaRect(cnt)  
    global imgarea
    box=cv2.boxPoints(rect)
    angle = rect[2]
    box=np.int0(box)  
    output=False
    width=rect[1][0]
    height=rect[1][1]
   #(angle > -45 or angle == -180)
    if (width < height):
        angle = angle - 90
    if ((width!=0) and (height!=0) and (angle >= -45 or ((angle >= -180) and angle <= -135))):
        if (((height/width>1.20) and (height>width)) or ((width/height>1.20) and (width>height))):
            if((height*width < 0.65 * imgarea) & (height*width> 0.0025* imgarea)): 
                output = True
    return output

#Lets draw validated contours with red.
for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)
        box=cv2.boxPoints(rect) 
        box=np.int0(box)  
        cv2.drawContours(carsample, [box], 0, (0,0,255),2)
showfig(carsample, None)
plt.show('validated')

def generate_seeds(centre, width, height):
    global imgwidth,imgheight
    minsize=int(min(width, height))
    seed=[None]*10
    for i in range(10):
        #random_integer1=np.random.randint(1000)
        #random_integer2=np.random.randint(1000)
        random_integer1 = (-width/2) + width *  np.random.rand(1,1)
        random_integer2 = (-height/2) + height *  np.random.rand(1,1)
        #seed[i]=(centre[0]+random_integer1%int(minsize/2)-int(minsize/2),centre[1]+random_integer2%int(minsize/2)-int(minsize/2))
        if(centre[0] + random_integer1 > imgwidth or centre[1]+random_integer2 > imgheight):
            if(centre[0] + random_integer1 > imgwidth and centre[1]+random_integer2 > imgheight):
                seed[i]=(centre[0]-random_integer1,centre[1]-random_integer2)
            elif(centre[0] + random_integer1 > imgwidth and centre[1]+random_integer2 < imgheight):
                seed[i]=(centre[0]-random_integer1,centre[1]+random_integer2)
            elif(centre[0] + random_integer1 < imgwidth and centre[1]+random_integer2 > imgheight):
                seed[i]=(centre[0]+random_integer1,centre[1]-random_integer2)
        elif(centre[0] + random_integer1 < 0 or centre[1]+random_integer2 < 0):
            if(centre[0] + random_integer1 <0 and centre[1]+random_integer2 <0):
                seed[i]=(centre[0]-random_integer1,centre[1]-random_integer2)
            elif(centre[0] + random_integer1 <0 and centre[1]+random_integer2 >0):
                seed[i]=(centre[0]-random_integer1,centre[1]+random_integer2)
            elif(centre[0] + random_integer1 >0 and centre[1]+random_integer2 <0):
                seed[i]=(centre[0]+random_integer1,centre[1]-random_integer2)
        else:
            seed[i]=(centre[0]+random_integer1,centre[1]+random_integer2)

    return seed

#masks are nothing but those floodfilled images per seed.
def generate_mask(image, seed_point):
    h=carsample.shape[0]
    w=carsample.shape[1]
    #print(seed_point)
    #OpenCV wants its mask to be exactly two pixels greater than the source image.
    mask=np.zeros((h+2, w+2), np.uint8)
    #We choose a color difference of (50,50,50). Thats a guess from my side.
    lodiff=70
    updiff=70
    connectivity=4
    newmaskval=255
    flags=connectivity+(newmaskval<<8)+cv2.FLOODFILL_FIXED_RANGE+cv2.FLOODFILL_MASK_ONLY
    
    _=cv2.floodFill(image, mask, seed_point, (255, 0, 0),
                (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

# we will need a fresh copy of the image so as to draw masks.
carsample_mask=cv2.imread(image_path)

# for viewing the different masks later
mask_list=[]

for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt) 
        centre=(int(rect[0][0]), int(rect[0][1]))
        angle = rect[2]
        if(angle == -180 or angle == 0):
            width=rect[1][0]
            height=rect[1][1]
        else:
            
            print("SWAPPED")
            width=rect[1][1]
            height=rect[1][0]
        seeds=generate_seeds(centre, width, height)
        
        #now for each seed, we generate a mask
        for seed in seeds:

            # plot a tiny circle at the present seed.
            cv2.circle(carsample, seed, 1, (0,0,255), -1)
            print("%s+Width%s+Height%s:%s"%(centre,int(width/2),int(height/2),seed))
           
            #if(seed[0] > -1 and seed[1] > -1 ):
            # generate mask corresponding to the current seed.
            mask=generate_mask(carsample_mask, seed)
            mask_list.append(mask)   

#We plot 1st ten masks here
plt.rcParams['figure.figsize'] = 15,4
fig = plt.figure()
plt.title('Masks!')
for mask_no in range(10):
    fig.add_subplot(2, 5, mask_no+1)
    showfig(mask_list[mask_no], plt.get_cmap('gray'))
plt.show()

validated_masklist=[]
for mask in mask_list:
    contour=np.argwhere(mask.transpose()==255)
    if validate(contour):
        validated_masklist.append(mask)

try:
    assert (len(validated_masklist)!=0)
except AssertionError:
    print("No valid masks could be generated")

# We check for repetation of masks here.
#from scipy import sum as
#import scipy.sum as scipy_sum
# This function quantifies the difference between two images in terms of RMS.
def rmsdiff(im1, im2):
    diff=im1-im2
    output=False
    if np.sum(abs(diff))/float(min(np.sum(im1), np.sum(im2)))<0.01:
        output=True
    return output

# final masklist will be the final list of masks we will be working on.
final_masklist=[]
index=[]
for i in range(len(validated_masklist)-1):
    for j in range(i+1, len(validated_masklist)):
        if rmsdiff(validated_masklist[i], validated_masklist[j]):
            index.append(j)
for mask_no in list(set(range(len(validated_masklist)))-set(index)):
    final_masklist.append(validated_masklist[mask_no])

cropped_images=[]
for mask in final_masklist:
    contour=np.argwhere(mask.transpose()==255)
    rect=cv2.minAreaRect(contour)
    width=int(rect[1][0])
    height=int(rect[1][1])
    centre=(int(rect[0][0]), int(rect[0][1]))
    box=cv2.boxPoints(rect) 
    box=np.int0(box)
    #check for 90 degrees rotation
    if ((width/float(height))>1):
        # crop a particular rectangle from the source image
        cropped_image=cv2.getRectSubPix(carsample_mask, (width, height), centre)
    else:
        # crop a particular rectangle from the source image
        cropped_image=cv2.getRectSubPix(carsample_mask, (height, width), centre)

    # convert into grayscale
    cropped_image=cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # equalize the histogram
    cropped_image=cv2.equalizeHist(cropped_image)
    # resize to 260 cols and 63 rows. (Just something I have set as standard here)
    cropped_image=cv2.resize(cropped_image, (260, 180))
    cropped_images.append(cropped_image)

_=plt.subplots_adjust(hspace=0.000)
number_of_subplots=len(cropped_images)
for i,v in enumerate(range(number_of_subplots)):
    v = v+1
    #ax1 = plt.subplot(number_of_subplots,1,v)
    showfig(cropped_images[i], plt.get_cmap('gray'))
    plt.show()
for img in cropped_images:
    data = pytesseract.image_to_data(Image.fromarray(img),output_type = 'dict')
    print(data)
