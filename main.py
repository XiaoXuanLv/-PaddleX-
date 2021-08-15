# 解压数据集（解压一次即可，请勿重复解压）
# 该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**
!unzip -oq /home/aistudio/data/data103743/objDataset.zip


# 查看数据集文件结构
!tree objDataset -L 2


# 安装PaddleX
!pip install paddlex



# 划分数据集
!paddlex --split_dataset --format VOC --dataset_dir objDataset/facemask --val_value 0.2 --test_value 0.1



# 数据预处理
from paddlex.det import transforms

train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(target_size=608,interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Resize(target_size=608,interp='CUBIC'),
    transforms.Normalize(),
])



# 读取PascalVOC格式的检测数据集，并对样本进行相应的处理
import paddlex as pdx

train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/train_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/facemask',
    file_list='objDataset/facemask/val_list.txt',
    label_list='objDataset/facemask/labels.txt',
    transforms=eval_transforms)


# 初始化模型
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')



# 模型训练
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')



# 模型预测

image_name = 'objDataset/facemask/JPEGImages/maksssksksss104.png'
result = model.predict(image_name)
pdx.det.visualize(image_name,result,threshold=0.5,save_dir='./output/yolov3_mobilenetv1')

