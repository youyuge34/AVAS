import os
import sys
sys.path.append("..")
sys.path.append("./../")

import numpy as np
import torch
from tqdm import tqdm
import cv2
from model.image.align_model_with_MKG import build_model
import util.torch_util as torch_util
from PIL import Image

def normalize_image(img, size=(224,224), channel_first=True, bgr2rgb=True, scale_normalize=True):
    try:
        img = Image.open(img).convert('RGB')
        img = np.array(img)
    except Exception as e:
        print(e, img)
        img = np.zeros((224,224,3))
    if img is None:
        img = np.zeros((224,224,3))
    if size is not None:
        img = cv2.resize(img, size)
    # if bgr2rgb:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if scale_normalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img / 255.0 - mean) / std
    if channel_first:
        img = np.transpose(img, (2, 0, 1))

    return img.astype(np.float32)

def main():
    # print()
    config = {"model_path":"./../weights/commodity_poi_mml_20211221_with_MKG_withMKG_iteration_200000_0.4215_0.407_0.018.pth", "bert_path": "./../weights/bert"}
    key_word = '普陀山'
    device = "cuda:0" if torch.cuda.is_available() else torch.device('cpu')
    opt = torch_util.Dict2Obj(config)
    model = build_model(opt)
    model = model.to(device)
    model.eval()

    # image = cv2.imread("./test_images/cherry.jpg")
    # image = torch.tensor([normalize_image(image, (224, 224))])
    # image = image.to(device)
    #
    #
    # image2 = cv2.imread('./test_images/gamble.png')
    # image2 = torch.tensor([normalize_image(image2, (224, 224))])
    #
    # images = torch.cat([image, image2], dim=0)
    # print(images.size())
    path_dir = './../src/images_test'
    img_paths = []
    for name in os.listdir(path_dir):
        if name.endswith('jpg') or name.endswith('png'):
            img_paths.append(os.path.abspath(os.path.join(path_dir, name)))

    tokenize = model.get_tokenizer()
    text_feature = tokenize([key_word], padding=True, truncation=True, return_tensors='pt', max_length=50)
    text = text_feature.to(device)

    csv_file = 'align_{}图片_with_MKG.csv'.format(key_word)
    csv_records = []
    batch_size = 8
    iteraionts = len(img_paths) // batch_size + 1
    # iteraionts = 1
    print('iterations=', iteraionts)
    with torch.no_grad():
        for i in tqdm(range(iteraionts)):
            imgs = img_paths[i*batch_size:i*batch_size+batch_size]
            if len(imgs) == 0:
                continue
            imgs = [normalize_image(x) for x in imgs]
            imgs = [torch.tensor([x]) for x in imgs]
            imgs = torch.cat(imgs, dim=0)
            image_embeddings = model.encode_image(imgs)
            text_embeddings = model.encode_text(text)
            # loss = model(image, text)
            # print(loss)
            # import pdb; pdb.set_trace()
            res = image_embeddings @ text_embeddings.t()
            res = res.detach().cpu().numpy().squeeze()
            print(res)
            for j in range(res.shape[0]):
                csv_records.append([img_paths[i*batch_size+j], res[j]])
    with open(csv_file, 'w') as f:
        for line in csv_records:
            # print(line)
            f.write(','.join([str(x) for x in line]) + '\n')

        # sims = model(image, text)
        # print(sims)
        # py_probs = sims.softmax(dim=-1).cpu().numpy()
        # print(py_probs)

        # 计算损失
        # labels = torch.arange(b, device = device)
        # loss = (F.cross_entropy(sims, labels) + F.cross_entropy(sims.t(), labels)) / 2
        # return loss.mean()

    # model.image_model.eval()
    # torch.save(model.image_model.state_dict(), "./efficientnet-b3-align-tc-fixed.pth")

    # torch.save({
    #         'state_dict': model.state_dict()
    #     }, "./align_new.pth")
    # torch.save(model.state_dict(), "./align_new.pth")

if __name__ == '__main__':
    main()
