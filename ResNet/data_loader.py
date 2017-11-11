import os
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.
    
    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """
    def __init__(self, root,ratio=0.2,method='train', transform=None):
        """Initializes image paths and preprocessing module."""
        #self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        self.root=root
        self.ratio=ratio
        self.method=method
        self.read_files()

    def read_files(self):
         health=[]
         sick=[]
         health_label=[]
         sick_label=[]
         for d in os.listdir(self.root):
            for f in os.listdir(os.path.join(path,d)):
               if d=='health':
                  d0=d+'/'
                  health.append(os.path.join(path,d0,f))
                  health_label.append(1)
               if d=='sick':
                  d0=d+'/'
                  sick.append(os.path.join(path,d0,f))
                  sick_label.append(0)
         

         image_list = np.hstack((health, sick))
         label_list = np.hstack((health_label, sick_label))

         self.image_paths = np.array([image_list, label_list])
         self.image_paths = self.image_paths.transpose()
         np.random.shuffle(self.image_paths)


         n_train=int(len(self.image_paths)*(1-self.ratio))
         tra_image=self.image_paths[0:n_train]
         val_image=self.image_paths[n_train:]

         if self.method=='train':
             self.image_path=tra_image
         if self.method=='val':
             self.image_path=val_image
         
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image= self.image_path[:,0][index]
        label=int(self.image_path[:,1][index])   # int() function is very important here,or it will wrong!!!
        img = Image.open(image).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return (img,label)
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_path)


transform = transforms.Compose([
                    transforms.Scale([1080,1080]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
def get_loader(image_path, batch_size,transform=transform,method='val',num_workers=2):
    """Builds and returns Dataloader."""
    
#    transform = transforms.Compose([
#                    transforms.Scale(image_size),
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = ImageFolder(image_path, method=method,transform=transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


path="/home/ye/user/yejg/database/eye_jpg/train/"

imagefolder=ImageFolder(root=path)
loader=get_loader(path,20)

i=0
for (img,label) in loader:
    if i>1:
       break
    else:
      print(img.size(),label.size())
    i+=1

