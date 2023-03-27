import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np
import random
import cv2
import os
from diffusion import DiffusionCLIP
from configs.paths_config import HYBRID_MODEL_PATHS
src_img = ""
src_txt = ""
tar_img = ""
tar_txt = ""
def parse_args_and_config():
    img_path = 'imgs/1.png'  
    align_face = True #param {type:"boolean"}
    edit_type = 'Pixar' #param ['Pixar', 'Neanderthal','Sketch', 'Painting by Gogh', 'Tanned',  'With makeup', 'Without makeup', 'Female → Male']
    degree_of_change = 1
    n_inv_step =  40#param{type: "integer"}
    n_test_step = 6 #param [6] 
    
    human_gdrive_ids = {'Pixar':               ["1IoT7kZhtaoKf1uvhYhvyqzyG2MOJsqLe", "human_pixar_t601.pth"],
                    'Neanderthal':             ["1Uo0VI5kbATrQtckhEBKUPyRFNOcgwwne", "human_neanderthal_t601.pth"],
                    'Painting by Gogh':        ["1NXOL8oKTGLtpTsU_Vh5h0DmMeH7WG8rQ", "human_gogh_t601.pth"],
                    'Tanned':                  ["1k6aDDOedRxhjFsJIA0dZLi2kKNvFkSYk", "human_tanned_t201.pth"],
                    'Female → Male':           ["1n1GMVjVGxSwaQuWxoUGQ2pjV8Fhh72eh", "human_male_t401.pth"],
                    'Sketch':                  ["1V9HDO8AEQzfWFypng72WQJRZTSQ272gb", "human_sketch_t601.pth"],
                    'With makeup':             ["1OL0mKK48wvaFaWGEs3GHsCwxxg7LexOh", "human_with_makeup_t301.pth"],
                    'Without makeup':          ["157pTJBkXPoziGQdjy3SwdyeSpAjQiGRp", "human_without_makeup_t301.pth"],
                    }
    
    # gid = human_gdrive_ids[edit_type][0]
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("precomputed", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    model_path = os.path.join('checkpoint', human_gdrive_ids[edit_type][1])
    # dl = GoogleDrive_Dowonloader(True)
    isExist = os.path.exists(model_path)
    if not isExist:
        raise ValueError
    # dl.ensure_file_exists(gid, model_path)

    t_0 = int(model_path.split('_t')[-1].replace('.pth',''))
    print(f'return step t_0: {t_0}')
    exp_dir = f"runs/MANI_{img_path.split('/')[-1]}_align{align_face}"
    os.makedirs(exp_dir, exist_ok=True)
    args_dic = {
    'edit_one_image' : True,
    'config': 'celeba.yml', 
    't_0': t_0, 
    'n_inv_step': int(n_inv_step), 
    'n_test_step': int(n_test_step),
    'sample_type': 'ddim', 
    'eta': 0.0,
    'bs_test': 1, 
    'model_path': model_path, 
    'img_path': img_path, 
    'deterministic_inv': 1, 
    'hybrid_noise': 0, 
    'n_iter': 1,  
    'align_face': align_face, 
    'image_folder': exp_dir,
    'model_ratio': degree_of_change,
    'edit_attr': None, 'src_txts': None, 'trg_txts': None,
    }
    
    args = dict2namespace(args_dic)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)

    # os.makedirs(args.exp, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def normalised(cliploss):
    if len(cliploss) > 0 :
        x = min(cliploss)
        if(x > 0):
            cliploss[1] = x-0.01
        return cliploss
    else:
        for thr in list(hybrid_config.keys()):
            if t.item() >= thr:
                et = 0
                logvar = 0
                for i, ratio in enumerate(hybrid_config[thr]):
                    ratio /= sum(hybrid_config[thr])
                    et_i = models[i+1](xt, t)
                    if learn_sigma:
                        et_i, logvar_learned = torch.split(et_i, et_i.shape[1] // 2, dim=1)
                        logvar_i = logvar_learned
                    else:
                        logvar_i = extract(logvars, t, xt.shape)
                    et += ratio * et_i
                    logvar += ratio * logvar_i
                break
        return cliploss

def Cliploss(src_img,src_txt,tar_img,tar_txt):
    x=0
    if(x==1):
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
    else:
         return random.uniform(0.4,1.0)

def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Config =")
    print("<" * 80)


    cliploss = []
    
    for i in range(1, 10+1):
        args.model_ratio = i
        runner = DiffusionCLIP(args, config)
        try:
            if args.edit_one_image:
                runner.edit_one_image()
            else:
                print('Choose one mode!')
                raise ValueError
        except Exception:
            logging.error(traceback.format_exc())

        cliploss.append(Cliploss(src_img,src_txt,tar_img,tar_txt))
    cliploss = normalised(cliploss)
    path = args.image_folder
    sorted_org = sorted(os.listdir(path))
    # print(sorted_org)
    img = cv2.imread(os.path.join(path,sorted_org[3]),cv2.IMREAD_ANYCOLOR)
    print(cliploss)
    cv2.imshow("result",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(cliploss)

    return 0


if __name__ == '__main__':
    sys.exit(main())
