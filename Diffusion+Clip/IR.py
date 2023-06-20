import os
import subprocess
import time
from losses import loss_directional as ld
import torch
from PIL import Image
path = os.path.join(os.curdir,'checkpoint')
dir_list = os.listdir(path)
model_path_list = []
src_text = input("Enter source text: ")
target_text = input("Enter the target text: ")
img_path = input("Enter image path: ")


for i in dir_list:
    trg_txt = i.split('_')[4][2:-2]
    t0 = i.split('_')[5]
    inv = i.split('_')[6][4:]
    gen = i.split('_')[7][4:]
    id_loss = i.split('_')[8][2:]
    l1_loss = i.split('_')[9][2:]
    if target_text == trg_txt:
        model_path_list.append(i)
    # print(trg_txt,t0,inv,gen,id_loss,l1_loss)
print(model_path_list)
if len(model_path_list) == 0:
    change = int(input("Power of change in range (1-15) integer: "))
    # to = 201, 301, 401,  500, 601 
    # l1 0.3, 0
    # id 0.3, 0
    l1 = 0.3
    t0 = 201
    id = 0.3
    if change == 1:
        t0 = 201
    elif change == 2:
        t0 = 301
    elif change == 3:
        t0 = 401
    elif change == 4:
        t0 = 500
    elif change == 5:
        t0 = 601
    elif change == 6:
        t0 = 201
        id = 0.0
    elif change == 7:
        t0 = 201
        l1 = 0.0
        id = 0.0
    elif change == 8:
        t0 = 301
        l1 = 0.3
        id = 0.0
    elif change == 9:
        t0 = 301
        l1 = 0.0
        id = 0.0
    elif change == 10:
        t0 = 401
        l1 = 0.3
        id = 0.0
        
    elif change == 11:
        t0 = 401
        l1 = 0.0
        id = 0.0
    elif change == 12:
        t0 = 500
        l1 = 0.3
        id = 0.0
    
    elif change == 13:
        t0 = 500
        l1 = 0.0
        id = 0.0
    
    elif change == 14:
        t0 = 601
        l1 = 0.3
        id = 0.0
    
    else :
        t0 = 601
        l1 = 0.0
        id = 0.0
    

    call = ["python", "/home2/bagler/Clip/DiffusionCLIP/main.py", "--clip_finetune",
        "--config", "celeba.yml",
        "--exp", "train",
        "--src_txts", "Human",
        "--trg_txts", target_text,
        "--do_train", "1",
        "--do_test", "0",
        "--n_train_img", "50",
        "--n_test_img", "10",
        "--n_iter", "5",
        "--t_0", f"{t0}",
        "--n_inv_step", "40",
        "--n_train_step", "6",
        "--n_test_step", "40",
        "--lr_clip_finetune", "4e-6",
        "--id_loss_w", f"{id}",
        "--l1_loss_w", f"{l1}"]
    print(call)
    subprocess.run(call)
    
    base_path = f'train_FT_CelebA_HQ_{[target_text]}_t{t0}_ninv40_ngen6_id{id}_l1{l1}_lr4e-06_'
    p = str(target_text).replace(" ","_")
    for i in range(0,5):
        model_path_list.append(base_path+f"{p}-{i}.pth")
    
    photos_dir = ""
    photos_list = []
    for i in model_path_list:
        current_time = int(time.time())
        print(f"Model -- {i}")
        trg_txt = i.split('_')[4][2:-2]
        t0 = i.split('_')[5][1:]
        inv = i.split('_')[6][4:]
        gen = i.split('_')[7][4:]
        id_loss = i.split('_')[8][2:]
        l1_loss = i.split('_')[9][2:]
        path = os.path.join('./checkpoint', i)
        l = ["python", "/home2/bagler/Clip/DiffusionCLIP/main.py", "--edit_one_image",
        "--config", "celeba.yml",
        "--exp", f"./runs/{current_time}/{target_text}/test",
        "--trg_txts", target_text,
        "--n_iter", "1",
        "--t_0", t0,
        "--n_inv_step", "200",
        "--n_test_step", "40",
        "--img_path", img_path,
        "--model_path", path]
        # print(l)
        subprocess.run(l)
        photos_dir = f"./runs/{current_time}/{target_text}/"
        all_items = os.listdir(photos_dir)

        directories = [item for item in all_items if os.path.isdir(os.path.join(photos_dir, item))]

        directory_name = directories[0]
        directory_path = os.path.join(photos_dir, directory_name)
        directory_path = os.path.join(directory_path,"image_samples")
        all_photos = os.listdir(directory_path)
        for file_name in all_photos:
            txt_img = os.path.join(directory_path,file_name)
            if file_name.startswith('3_gen'):
                # If it does, add the file to the list
                photos_list.append(txt_img)

        # print(photos_list)
    print('------------------------')
    # print(i)
    # res,idl,cl,ssim,lpips_Score=ld.getClipLoss(img_path,photos_list,src_text,target_text,torch.device('cuda'))   
    res=ld.getClipLoss(img_path,photos_list,src_text,target_text,torch.device('cuda'))   
    mini = 1
    for item in res:
        if(mini > item):
            mini = item
    for i in range(0,len(res)):
        if mini == res[i]:
            photo_img = photos_list[i]
            im = Image.open(photo_img)
            print(photo_img)
            im.show()
            break

else:
     
    photos_dir = ""
    photos_list = []
    src_image_path = []
    for i in model_path_list:
        current_time = int(time.time())
        print(f"Model -- {i}")
        trg_txt = i.split('_')[4][2:-2]
        t0 = i.split('_')[5][1:]
        inv = i.split('_')[6][4:]
        gen = i.split('_')[7][4:]
        id_loss = i.split('_')[8][2:]
        l1_loss = i.split('_')[9][2:]
        path = os.path.join('./checkpoint', i)
        l = ["python", "/home2/bagler/Clip/DiffusionCLIP/main.py", "--edit_one_image",
        "--config", "celeba.yml",
        "--exp", f"./runs/{current_time}/{target_text}/test",
        "--trg_txts", target_text,
        "--n_iter", "1",
        "--t_0", t0,
        "--n_inv_step", "200",
        "--n_test_step", "40",
        "--img_path", img_path,
        "--model_path", path]
        # print(l)
        subprocess.run(l)
        photos_dir = f"./runs/{current_time}/{target_text}/"
        all_items = os.listdir(photos_dir)

        directories = [item for item in all_items if os.path.isdir(os.path.join(photos_dir, item))]

        directory_name = directories[0]
        directory_path = os.path.join(photos_dir, directory_name)
        directory_path = os.path.join(directory_path,"image_samples")
        all_photos = os.listdir(directory_path)
        for file_name in all_photos:
            txt_img = os.path.join(directory_path,file_name)
            if file_name.startswith('3_gen'):
                # If it does, add the file to the list
                photos_list.append(txt_img)
        


    # for i in photos_list:
        # trt_path = os.path.join(".", )

    print('------------------------')
    # print(i)
    # res,idl,cl,ssim,lpips_Score=ld.getClipLoss(img_path,photos_list,src_text,target_text,torch.device('cuda'))   
    res=ld.getClipLoss(img_path,photos_list,src_text,target_text,torch.device('cuda'))   
    mini = 1
    for item in res:
        if(mini > item):
            mini = item
    for i in range(0,len(res)):
        if mini == res[i]:
            photo_img = photos_list[i]
            im = Image.open(photo_img)
            print(photo_img)
            im.show()
            break

    # cl=list(cl)

    # idl=list(idl)
    # print(cl)
    # print(idl)
    # print(ssim)
    # print(lpips_Score) 
    # print(photos_list)

    # print(photos_list)
    
