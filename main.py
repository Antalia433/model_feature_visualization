#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


class AllScatter:
    '''
    为了多组坐标点，专门做的一个类，把单独的查值和找点变成组内查找、
    '''

    def __init__(self):
        self.sc = []
        self.sc_size = []
        self.ind_sc = None

    def initialize(self, obj, size):
        self.sc.append(obj)
        self.sc_size.append(size)

    def contains(self, input_event):
        # cont, ind = scatter.contains(event) cont代表鼠标event是否在sca上，ind是sca中数据的索引（最开始输入sca的点的顺序，从0开始
        # 索引是单类索引、需要算上过往ind的counts、
        all_ind = 0
        for idx, i in enumerate(self.sc):
            # print(idx,i)
            cont, ind = i.contains(input_event)
            if cont:
                self.ind_sc = i
                all_ind += ind['ind'][0]
                return cont, ind, all_ind
            all_ind += self.sc_size[idx]
        return cont, ind, all_ind

    def get_offsets(self, ind_info):
        # pos = scatter.get_offsets()[ind["ind"][0]] pos将返回对应ind索引的坐标点(x,y)
        pos = self.ind_sc.get_offsets()[ind_info]
        return pos


def make_all_info(fe_path, info_path, model_path, root_path):
    import torch
    from Model import Model_close
    import torchvision.transforms as transforms
    from Dataset import main as data_loader
    def hook(module, inp, out):
        global features
        features = out

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Model_close(base_model='E4', out_dim=5).to(DEVICE)
    ck = torch.load(model_path, map_location=DEVICE)
    net.load_state_dict(ck['model'])
    net.eval()
    myhook = net.backbone._fe.register_forward_hook(hook)

    # 简单测试一下hook情况：
    path = 'e:/4.jpg'
    imgTrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((448, 448)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = imgTrans(Image.open(path)).unsqueeze(dim=0)
    pred = net(img)
    print(pred, '\n', features, features.shape)

    # 抽数据集数据
    test_loader = data_loader(root_path, info_path)
    embeddings = []
    labels = []
    preds = []
    imgs_path = []
    k = 0
    with torch.no_grad():
        for images, target, path in test_loader:
            # if k > 5:
            #     break
            _ = net(images.to(DEVICE))
            pred = torch.argmax(_)
            embeddings.append([float(features.data[0][0]), float(features.data[0][1])])
            labels.append(int(target.numpy()))
            preds.append(int(pred))
            imgs_path.append(path[0])
            k += len(images)
    with open(fe_path, 'w') as f:
        json.dump([embeddings, labels, imgs_path, preds], f, ensure_ascii=False)

    return embeddings, labels, imgs_path, preds


def make_fig(colors, embeddings, imgs_path, img_types, img_classes, mode):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sca = AllScatter()
    embeddings = np.array(embeddings)
    fig, ax = plt.subplots()
    if mode == 'label':
        for i in range(len(img_classes)):
            # 取某一个label的数据形成一组点，传到ax上，sca做各组点的统筹
            inds = np.where(np.array(img_types) == i)[0]
            scatter = ax.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
            sca.initialize(scatter, len(inds))
            print(len(inds))
    elif mode == 'pred':
        # 预测结果的排序跟标签对不上、要重新列一组path
        change_imgs_path = []
        for i in range(len(img_classes)):
            inds = np.where(np.array(img_types) == i)[0]
            for idx in inds:
                change_imgs_path.append(imgs_path[idx])
            scatter = ax.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
            sca.initialize(scatter, len(inds))
            print(len(inds))
        imgs_path = change_imgs_path

    im = OffsetImage(Image.open(imgs_path[0]))
    annot = AnnotationBbox(im, (0, 0), xycoords='data',
                           # box_alignment=(0, 0),# 注释框与数据点的距离参照点、默认为中心点，(0,0)/(1,1)表示左下角/右上角
                           boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
    ax.add_artist(annot)
    annot.set_visible(False)

    def hover(event):
        is_ind, ind, all_idx = sca.contains(event)
        # 鼠标悬浮在数据点上时、出注释
        if is_ind:
            # 指定注释的位置、pos即当前数据点的位置
            pos = sca.get_offsets(ind["ind"][0])
            annot.xy = pos
            out_path = imgs_path[all_idx]
            print(pos, out_path, all_idx)
            # 保持注释图box的长边是350
            ratio = 350
            img = Image.open(out_path)
            if img.size[0] > img.size[1]:
                max_size = img.size[0]
                max_idx = 0
            else:
                max_size = img.size[1]
                max_idx = 1
            ratio = ratio / max_size
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)))
            # 根据event(鼠标)的位置、控制注释box位置
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            annot.xybox = (img.size[max_idx] / 2 * ws, img.size[max_idx] / 2 * hs)
            annot.set_visible(True)
            # 把img送进注释box
            im.set_data(img)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    ax.legend(img_classes)
    plt.subplots_adjust(left=0.07, right=0.95, top=0.94, bottom=0.08)


def main():
    root_path = ''
    info_path = ''
    model_path = ''
    fe_path = 'all_re.json'
    # 检测/加载特征值、
    if os.path.exists(fe_path):
        with open(fe_path, 'r') as f:
            embeddings, labels, imgs_path, preds = json.load(f)
    else:
        embeddings, labels, imgs_path, preds = make_all_info(fe_path, info_path, model_path, root_path)
    # 2 加载动态图像
    img_classes = ['label1', 'label2', 'l3', 'l4', 'l5']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    make_fig(colors, embeddings, imgs_path, labels, img_classes, 'label')
    make_fig(colors, embeddings, imgs_path, preds, img_classes, 'pred')
    plt.show()

if __name__ == '__main__':
    main()
