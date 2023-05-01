# Low-Light Image Restoration

## **Abstract**

- To enhance low-light images
- To restore the images to a normal light image condition
- To remove noise, adjust the brightness and contrast, and produce images that look like they were taken in normal light conditions.

## **Introduction**

Low light conditions can lead to poor image quality and limited visibility of important details in photographs. Traditional image enhancement techniques may not be effective in improving low-light images due to high noise levels and loss of information. This research aims to develop novel methods for enhancing low-light images by addressing these specific challenges, with the goal of improving the overall quality and visual clarity of photographs taken in low-light conditions.

## **Dataset**

The LOL dataset is a collection of 500 low-light and normal-light image pairs. It is divided into 485 training pairs and 15 testing pairs. The low-light images contain noise produced during the photo capture process.

## **Model Architecture**

The model architecture consists of converting the image to the auto-contrast form of the image and passing it through 6 convolutional blocks followed by 5 residual blocks. After this, the image is deconvolved and combined with the auto-contrast image to produce the final enhanced image.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*RRZb2RVq7DN0UjGlvbwX5Q.png)

## **Model Training**

The model is trained with a combination of perceptual loss calculate on 2 blocks of pre trained VGGnet and charbonnier loss calculated on the output of our model on the LOL dataset for 50 epochs

| ![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*uffFBFblsUsvZoragIJM6w.jpeg) |
| :--------------------------------------------------------------------------------------: |
|                                      PSNR vs Epochs                                      |

| ![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*vmf1CYpusTvcFJFzYJ5t5w.jpeg) |
| :--------------------------------------------------------------------------------------: |
|                                      Loss vs Epochs                                      |

## **Results**

Here are some example image results

![](https://miro.medium.com/v2/resize:fit:620/format:webp/1*urE3vJVJp9tSkIIG3Ld-MA.png)

## **Conclusions**

We found out that a custom loss function of perceptual loss and charbonnier loss could be used to produce the smoothest images.
With our custom model architecture, we were able to achieve low latency as well as a higher PSNR of around 20 as compared to similar models.

## **References**

- Syed Waqas Zamir, et. al “Learning Enriched Features for Real Image Restoration and Enhancement”

- Karen Simonyan, Andrew Zisserman “Very Deep Convolutional Networks for Large-Scale Image Recognition”
