

"""# Heatmap"""

tf.compat.v1.disable_eager_execution()

x=photo = np.reshape(old_img, [1, 512, 512, 3])

# This is the  prediction vector
output = model.output[0,65025,0]

#output = model.output[0,65536, 0]
# The is the output feature map
last_conv_layer = model.get_layer('conv2d_4') # activation_518 conv2d_223
#last_conv_layer = model.get_layer('activation_3')
# This is the gradient
grads = K.gradients(output, last_conv_layer.output)[0]

# This is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
#iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# These are the values of these two quantities, as Numpy arrays,

pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(100):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=2)

import matplotlib.pyplot as plt
heatmap = np.maximum(heatmap, 0)
#heatmap= - heatmap
heatmap =(heatmap-np.min(heatmap))/ (np.max(heatmap)-np.min(heatmap))

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (512, 512))
#heatmap=heatmap+heatmap1
# We convert the heatmap to RGB
heatmap = np.uint8(heatmap*255)
plt.axis('off')
plt.imshow(heatmap)

from PIL import Image
outputImg = Image.fromarray(heatmap)
outputImg = outputImg.convert('L')
outputImg.save('/content/drive/MyDrive/heatmap2.jpg')

plt.imshow(outputImg)

image=Image.open('/928443_sat.jpg') #247 185 146 206 0013 0075
old_img=np.array(image.resize((HEIGHT,WIDTH), Image.BILINEAR))

# We apply the heatmap to the original image
heatmap1 = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # COLORMAP_HOT COLORMAP_JET COLORMAP_OCEAN COLORMAP_PINK COLORMAP_SPRING
# alpha here is a heatmap intensity factor
alpha=0.5
superimposed_img = heatmap1*(1-alpha)+old_img*alpha
cv2_imshow(superimposed_img)