
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.utils as tvu
import os   
import clip
import torchvision.transforms as transforms


imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


direction_loss_type='cosine'

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y) 


direction_loss = DirectionLoss(direction_loss_type)
model, clip_preprocess = clip.load('ViT-B/32', device='cuda')

preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])    

def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

def encode_text(tokens: list) -> torch.Tensor:
        return model.encode_text(tokens)

def get_text_features(class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to('cuda')

        text_features = encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features




def encode_images(images: torch.Tensor) -> torch.Tensor:        
        images = preprocess(images).to('cuda')
        return model.encode_image(images)

def get_image_features(img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

def compute_text_direction(source_class: str, target_class: str) -> torch.Tensor:
        source_features = get_text_features(source_class)
        target_features = get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction



def clip_directional_loss(src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

    target_direction = compute_text_direction(source_class, target_class)

    src_encoding    = get_image_features(src_img)
    target_encoding = get_image_features(target_img)
    edit_direction = (target_encoding - src_encoding)
    edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
    return direction_loss(edit_direction, target_direction).mean()    



if __name__ == "__main__":    
    x0s , xs = [], []
    sorted_org = sorted(os.listdir('MANI_'))
    print(sorted_org)
      
    x0 = Image.open(f'./Clip_imgs_F2M_org/0_orig.png').convert("RGB")
    # x0 = x0.resize((256,256))
    x0 = ToTensor()(x0)
    x0 = 2*x0-1
    # x0s.append(x0)
    edited_org = sorted(os.listdir('Clip_imgs_F2M_edited'))

    for img1 in edited_org:          
        x = Image.open(f'./Clip_imgs_F2M_edited/{img1}').convert("RGB")
        x = ToTensor()(x)
        x = 2*x-1
        # xs.append(x)
        
    x0_batch = torch.stack(x0)
    x_batch = torch.stack(xs)
    
    
    x0_batch = x0_batch.to('cuda')
    x_batch = x_batch.to('cuda')

    src_txt = 'Human'
    trg_txt = '3D render in the style of Pixar'

    loss_clip = (2 - clip_directional_loss(x0_batch, src_txt, x_batch, trg_txt)) / 2
    print("orginal")
    print(img)
    print("edited")
    print(img1)
    print(loss_clip)
    print("----------------------------------")