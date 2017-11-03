import os
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset compatible with prebuilt DataLoader.
    
    This is just for tutorial. You can use the prebuilt torchvision.datasets.ImageFolder.
    """
    def __init__(self, root, transform=None):
        """Initializes image paths and preprocessing module."""
        #self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform
        self.root=root
        self.read_files()

    def read_files(self):
         health=[]
         sick=[]
         health_label=[]
         sick_label=[]
         for d in os.listdir(self.root):
            for f in os.listdir(os.path.join(path,d)):
               if d=='gen1_health':
                  d0=d+'/'
                  health.append(os.path.join(path,d0,f))
                  health_label.append(1)
               if d=='gen1_sick':
                  d0=d+'/'
                  sick.append(os.path.join(path,d0,f))
                  sick_label.append(0)
         

         image_list = np.hstack((health, sick))
         label_list = np.hstack((health_label, sick_label))

         self.image_paths = np.array([image_list, label_list])
         self.image_paths = self.image_paths.transpose()
         np.random.shuffle(self.image_paths)
         
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[:,0][index]
        label=int(self.image_paths[:,1][index])   # int() function is very important here,or it will wrong!!!
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (image,label)
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

    
def get_loader(image_path, image_size, batch_size,num_workers=2):
    """Builds and returns Dataloader."""
    
    transform = transforms.Compose([
                    transforms.Scale(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


path="/home/ye/user/yejg/database/eye_jpg/train/"

imagefolder=ImageFolder(root=path)
print(imagefolder[10])
image=imagefolder[10]
loader=get_loader(path,(1080,1080),20)
i=0
for (img,label) in loader:
    if i>1:
       break
    else:
      print(img.size(),label.size())
    i+=1
