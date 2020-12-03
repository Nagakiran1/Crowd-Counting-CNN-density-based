from keras.models import model_from_json
import numpy as np

from keras.models import model_from_json
import numpy as np

def Preoprocessimage(a, ModeofImage):
  #a[np.abs(a[:,:,0]-m[:,:,0])<10] = 255
  a[np.abs(a.sum(2) - ModeofImage.sum(2))<25] = 255
  kernel = np.ones((2,2),np.uint8)
  b = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
  b = cv2.GaussianBlur(b,(3,3),0)
  return b

def GetNoofPersons(points, row):
  coords = points[((points[:,0]>row['x']) & (points[:,0]<(row['x']+row['w']))) & ((points[:,1]>row['y']) & (points[:,1]<(row['y']+row['h'])))]
  return coords.shape[0]

def GenerateInsideData(activation_model, array, points, image, testing=False):
  # Buidling inbetween 6 layer model of trained model
  a = activation_model.predict(array)
  # print(int(model.predict(array)[0][0]), '  -  ', labels[rvalue])

  # Taking Fifthe layer Embeddings to represent the Convolution features of Images
  first_layer_activation = a[5]
  n = first_layer_activation[0, :, :, 0].copy()
  # Applying the Median Blur Image processing technique to reduce Noise of images
  n = cv2.medianBlur(n,3)
  # Applying the Median Blur Image processing technique to reduce Noise of images
  n[n<n.max()/3] = 0
  n = cv2.medianBlur(n,3)

  # Applyint Contours to detected crowd patterns
  df, im1 = ApplyContours(n.copy())

  # Adjusting the multiple contours and the area wise dilation of contours from Camera angle
  df, im2, ImSegments = AdjustContours(df, image.copy())

  # Assigning Number of persons associated with the each Cluster cropped
  df['NoofPersons'] = df.apply(lambda x: GetNoofPersons(points, x), 1)

  # Filter and zero padding the images to apply on the Crowd Counting model
  CrowdGroups  = np.concatenate([FilterImage(word)[np.newaxis] for word in ImSegments])
  if testing:
    return df, CrowdGroups,n,im2
  #df['Images'] = CrowdGroups
  return df, CrowdGroups


def PlotGroups(im1, df1, font=0.7):
  for ind, row in df1.iterrows():
      x,y,w,h = row[['x','y','w','h']].astype(int).values
      cv2.rectangle(im1,(x,y),(x+w,y+h),(0,255,0),2)
      cv2.putText(im1, 'group size - '+ str(int(row['NoofPersons'])), 
                  (x,y+20),  cv2.FONT_HERSHEY_SIMPLEX , font, (255, 0, 0), 2, cv2.LINE_AA) 
  return im1


# load json and create model
json_file = open('drive/My Drive/project/CrowdCounting/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
GroupModel = model_from_json(loaded_model_json)
# load weights into new model
GroupModel.load_weights("drive/My Drive/project/CrowdCounting/model1.h5")
print("Loaded model from disk")
 
ModeofImage = np.load('drive/My Drive/project/CrowdCounting/ModeOfImage.npy')



# load json and create model
json_file = open('drive/My Drive/project/CrowdCounting/PersonModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
PersonModel = model_from_json(loaded_model_json)
# load weights into new model
PersonModel.load_weights("drive/My Drive/project/CrowdCounting/PersonModel.h5")
print("Loaded model from disk")
 