# Text Guided Image Manipulation

  
Text-guided image manipulation is an image editing technique that manipulates a given image according to the natural language text descriptions. It is a rapidly growing technique in the field of NLP Natural Language Processing) and CV (Computer Vision). Recent advancements in Deep Learning have opened doors to various image manipulation applications. Despite remarkable advances in image generation methods, general domain high-fidelity image editing still needs to be improved. We propose a more approachable technique that can automatically edit a given image using natural language descriptions.


## Installation

To use this, you'll need to have Python 3 installed on your machine, along with several Python packages, including PyTorch, NumPy, and PIL. You can install these packages using pip by running the following command:

pip install -r requirements.txt


### Resources
- For the original fine-tuning, VRAM of 24 GB+ for 256x256 images are required.
- For the GPU-efficient fine-tuning, VRAM of 12 GB+ for 256x256 images and 24 GB+ for 512x512 images are required.
- For the inference, VRAM of 6 GB+ for 256x256 images and 9 GB+ for 512x512 images are required.


## Dataset
We have used CelebA-HQ dataset in our project. To precompute latents and fine-tune the Diffusion models, you need about 30+ images in the source domain. You can use both sampled images from the pretrained models or real source images from the pretraining dataset. If you want to use real source images,

for CelebA-HQ, you can use following code:
bash data_download.sh celeba_hq .

### Model Architecture
![architecture](https://raw.githubusercontent.com/agrawals1/IR_Project/main/Final_23April/Arch1.jpeg)
![architecture](https://raw.githubusercontent.com/agrawals1/IR_Project/main/Final_23April/Arch2.jpeg)


## Codebase
We have use the following techniques in our implementation:-

- Stable Diffusion (To Generate and Manipulate Image)
- Clip Loss (For Fine-tuning the diffusion process & for finding relevance corresponding to input)
- ID Loss (For Fine-tuning the diffusion process & for finding relevance corresponding to input)
- SSIM Loss (For finding relevance corresponding to input)
- LPIPS Loss  (For finding relevance corresponding to input)

## Usage
To use our, simply run the IR.py. Then you have to enter a source text (related to input image), the target text (that you want to happen to the image) and path of the input image. <br> For example:
>- python IR.py
>- Enter source text: human
>- Enter the target text: Curly hair
>- Enter image path: imgs/celeb1.png

## Contributions
This project is created by a team of M.Tech students at the Indraprastha Institute of Information Technology (IIIT) Delhi. Our team includes:
- Abhuday Tiwari (MT22005) 
-  Amrita Aash (MT22011) 
- Kirti Vashishtha (MT22035)
- Nikhilesh Verhwani (MT22114)
- Ritisha Gupta (MT22056)
- Shubham Agrawal (MT22124)  

## Contact
If you have any questions or feedback about the project, please contact us at any one of the following:
- [abhuday22005@iiitd.ac.in](mailto:abhuday22005@iiitd.ac.in)
- [amrita22008@iiitd.ac.in](mailto:amrita22008@iiitd.ac.in) 
- [kirti22035@iiitd.ac.in](mailto:kirti22035@iiitd.ac.in)
- [nikhilesh22114@iiitd.ac.in](mailto:nikhilesh22114@iiitd.ac.in) 
- [ritisha22056@iiitd.ac.in](mailto:ritisha22056@iiitd.ac.in)
- [shubham22124@iiitd.ac.in](mailto:shubham22124@iiitd.ac.in)

##  Acknowledgments
We'd like to thank the creators of the CLIP model and stable diffusion for their groundbreaking work in text-to-image generation/manipulation. We'd also like to acknowledge the many open-source libraries and tools that we've used in developing including PyTorch, NumPy, and PIL.

