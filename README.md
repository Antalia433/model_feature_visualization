对模型结果与数据情况的可视化：

需要改模型源码，在fc层前加了一组可以生成二维向量的fe，可参照efficientnet_torch.model.EfficientNet

## 实际效果

【gif】

## 需要配置的内容

Model.py，按照自己的模型改

Dataset.py，默认的数据数据文件格式是csv，columns=['name','label']，name里放相对位置

几个参数要按需调：

```python
MyDataset.img_size = 448 #设成模型训练时的默认大小，否则结果不对
MyDataset.new_crop = T/F # 是否等比例裁剪，如果训练时没有等比例裁剪，不要开
```

main.py里的main:

```python	
root_path = '' # 数据文件里name相对位置的root path
info_path = '' # 可读的数据文件path、
model_path = '' #可读的模型path
fe_path = 'all_re.json' # 存信息的path
img_classes = ['label1','label2','l3','l4','l5'] # 可以放中文，顺序跟model识别的label一致
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # 自定义颜色
```



会按label情况和预测结果开两个fig、鼠标悬浮能直接显示原数据、可以直观的审核数据

有fe的模型在训练时，会扛不住预处理，acc很难涨、

测试时关闭预处理，acc还行，只比不加fe层的模型少1%左右、能看不好用……



参考链接：[模型可视化](https://github.com/adambielski/siamese-triplet)/[plt动态图像注释](https://zhuanlan.zhihu.com/p/459638677)/[EfficientNet ](https://github.com/lukemelas/EfficientNet-PyTorch)