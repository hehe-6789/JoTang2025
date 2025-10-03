import torch
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
import os
root_dir='../../picture/custom_image_dataset'
is_test=False

#定义管道
transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

#定义数据库
class data_base(Dataset):
    def __init__(self,root_dir=root_dir,transform=transform,is_test=is_test):
        super().__init__()
        self.root_dir=root_dir
        self.transform=transform
        self.is_test=is_test
        """记录关键数据"""
        self.labels=[]
        self.dir_path=[]

        """分为训练集与测试集讨论"""
        if not is_test:
            for class_name in sorted(os.listdir(root_dir)):
                root_second_dir=os.path.join(root_dir,class_name)
                for class_name2 in os.listdir(root_second_dir):
                    self.dir_path.append(os.path.join(root_second_dir,class_name2))
                    self.labels.append(int(class_name))
        else:
            for class_name in sorted(os.listdir(root_dir),key=lambda x:int(x[4:-4])):
                self.dir_path.append(os.path.join(root_dir,class_name))
                self.labels.append(class_name)

    def __len__(self):
        return len(self.dir_path)            


    def __getitem__(self,index):
        if not is_test:
            image=Image.open(self.dir_path[index]).convert('RGB')
            image=self.transform(image)
            return image,self.labels[index]
        else:
            image=Image.open(self.dir_path[index]).convert('RGB')
            image=self.transform(image) 
            return image,self.labels[index]

#定义获取加载器加载器
def get_loader(root_dir=root_dir,is_test=is_test,transform=transform,batch_size=64,shuffle=True,numworkers=os.cpu_count()//2):
    """"""
    database=data_base(root_dir=root_dir,transform=transform,is_test=is_test)
    database=DataLoader(database,batch_size=batch_size,shuffle=shuffle,num_workers=numworkers)
    return database

