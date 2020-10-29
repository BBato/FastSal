import model.fastSal as fastsal
from utils import load_weight
from dataset.utils import resize_interpolate, pytorch_normalze
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from generate_img import post_process_png, post_process_probability2

def convert_vgg_img(src, target_size):
    vgg_img = src
    original_size = vgg_img.size
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[1] or target_size[1] != original_size[0]):
            vgg_img = vgg_img.resize((target_size[1], target_size[0]), Image.ANTIALIAS)
        elif isinstance(target_size, int):
            vgg_img = vgg_img.resize((int(original_size[0]/target_size), int(original_size[2]/target_size))
                                     , Image.ANTIALIAS)
    vgg_img = np.asarray(vgg_img, dtype=np.float32)
    #print(vgg_img.shape,'???')
    vgg_img = pytorch_normalze(torch.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return vgg_img, np.asarray(original_size)

class img_dataset(Dataset):
    def __init__(self, frame):
        self.frame = frame
    def __getitem__(self, item):
        vgg_img, original_size = convert_vgg_img(self.frame, (192, 256))
        return vgg_img, original_size
    def __len__(self):
        return 1
    
if __name__ == '__main__':


    model = fastsal.fastsal(pretrain_mode=False, model_type='A')
    state_dict, opt_state = load_weight('weights/{}_{}.pth'.format('salicon', 'A'), remove_decoder=False)
    model.load_state_dict(state_dict)

    # define a video capture object 
    vid = cv2.VideoCapture(0) 

    while(True): 
        
        # Capture the video frame by frame 
        ret, frame = vid.read() 
        
        # Resize to standard size
        # frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        print("Video ready.")

        # Convert to PIL object
        cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        print("PIL ready.")

        # Generate Dataset object 
        ds = img_dataset(pil_im)
        print("Dataset ready.")

        # Load Dataset object
        vgg_data = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        print("Data loader ready.")

        for x, original_size_list in vgg_data:

            # Run model
            y = model(x)
            print("Model ready.")

            # Prepare for display
            y = nn.Sigmoid()(y)
            print("Sigmoid ready.")
            y = y.detach().numpy()
            for i, prediction in enumerate(y[:, 0, :, :]):
                original_size = original_size_list[i].numpy()
                img_data = post_process_png(prediction, original_size)
                cv2.imshow('img_output_path', img_data)
                cv2.imshow('original', frame)
             

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 