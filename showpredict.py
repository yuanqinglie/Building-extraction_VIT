
class showpredict():
  def __init__(self,labelpath,imagepath,NCLASSES,WIDTH,HEIGHT):
    self.imagepath=imagepath
    self.labelpath=labelpath
    self.NCLASSES=NCLASSES
    self.WIDTH=WIDTH
    self.HEIGHT=HEIGHT



    """
    self.img_label=np.moveaxis(rasterio.open(self.labelpath).read(),0,2)*255
    img=rasterio.open(self.imagepath)

    img=np.moveaxis(img.read()[0:3],0,2)
    self.old_img=cv2.cvtColor(img[:,:,0:3]*255, cv2.COLOR_BGR2RGB)

    """

    img_label =np.moveaxis(rasterio.open(self.labelpath).read(),0,2)
    self.img_label=(~(img_label.astype(bool))).astype(np.int32)*255
    img = np.array(Image.open(self.imagepath)) #/content/drive/MyDrive/test1/image/2_34.tif
    self.old_img=img
    img = img/255




    img = img.reshape(-1,self.HEIGHT,self.WIDTH,3)
    pr = model.predict(img,verbose=1)[0]
    self.seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))

    self.pr = pr.reshape((int(self.HEIGHT), int(self.WIDTH),self.NCLASSES)).argmax(axis=-1)

    self.colors = [[0,0,0],[255,255,255]]


    for c in range(self.NCLASSES):
        self.seg_img[:,:,0] += ((self.pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
        self.seg_img[:,:,1] += ((self.pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
        self.seg_img[:,:,2] += ((self.pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

    old_img = Image.fromarray(np.uint8(self.old_img))

    seg=np.zeros((512,512,3))

    seg+=(self.seg_img==255)*([255,0,0])

    seg = Image.fromarray(np.uint8(seg))
    self.image = Image.blend(old_img,seg,0.5)
    imagelabel=np.zeros((512,512,3))
    imagelabel+=(self.img_label==255)*([255,0,0])
    self.imagelabel = Image.blend(old_img,Image.fromarray(np.uint8(imagelabel)),0.5)
    #plt.imshow(self.imagelabel)




  #image.save("./img_out/"+jpg)
  def im_plot(self):


    c=np.uint8(self.seg_img)

    old_img1=np.uint8(self.old_img)

    #v =img_label-c
    #v[v>0]=255
    v=self.img_label

    v1=np.zeros((512,512,3))
    v1+=(v==255)*([0,255,0])

    v1 = np.uint8(v1)
    img2gray = cv2.cvtColor(v1,cv2.COLOR_BGR2GRAY)

    ret,mask = cv2.threshold(img2gray,0,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(old_img1,old_img1,mask=mask_inv)
    img2_fg = cv2.bitwise_and(v1,v1,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)

    pre_img = np.zeros((int(self.HEIGHT), int(self.WIDTH),3))

    FP=[]
    FN=[]
    for i in range(int(self.HEIGHT)):

       for j in range(int(self.WIDTH)):
          if self.seg_img[i,j,0]==self.img_label[i,j,0]==0: # TP(true positive,green)
             pre_img[i,j,:]+=dst[i,j,:]
          elif self.seg_img[i,j,0]==self.img_label[i,j,0]==255:
             pre_img[i,j,:]+=np.array(self.image)[i,j,:]
          elif (self.seg_img[i,j,0]==0 and self.img_label[i,j,0]==255):
              pre_img[i,j,:]+=[255,246,0]       #FN(false negative,yellow))
              FN.append([i,j])
          else:
              pre_img[i,j,:]+=[0,255,0] #FP(false positive,green)
              FP.append([i,j])
    pre_img=np.array(pre_img,dtype='uint8')


    return dst,pre_img,FN,FP,old_img1,self.img_label,self.image