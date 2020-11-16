import model.fastSal as fastsal
from utils import load_weight
import torch
import time

if __name__ == '__main__':
    coco_c = 'weights/coco_C.pth'  # coco_C
    coco_a = 'weights/coco_A.pth'  # coco_A
    salicon_c = 'weights/salicon_C.pth'  # salicon_C
    salicon_a = 'weights/salicon_A.pth'  # coco_A

    with torch.no_grad():

        x = torch.zeros((1, 3, 192, 256))

        model = fastsal.fastsal(pretrain_mode=False, model_type='A')
        state_dict, opt_state = load_weight(coco_a, remove_decoder=False)
        model.load_state_dict(state_dict)

        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )

        t1 = time.time()
        for i in range(0,3):
            y = model(x)
            print("x")
        t2 = time.time()
        interval = (t2-t1)/3
        fps = 1/interval
        print("Average FPS for coco_A was "+str(fps))
        print("Average time for coco_A was "+str(interval))

        model = fastsal.fastsal(pretrain_mode=False, model_type='A')
        state_dict, opt_state = load_weight(salicon_a, remove_decoder=False)
        model.load_state_dict(state_dict)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )
        t1 = time.time()
        for i in range(0,3):
            y = model(x)
            print("x")
        t2 = time.time()
        interval = (t2-t1)/3
        fps = 1/interval
        print("Average FPS for salicon_A was "+str(fps))
        print("Average time for salicon_A was "+str(interval))

        model = fastsal.fastsal(pretrain_mode=False, model_type='C')
        state_dict, opt_state = load_weight(coco_c, remove_decoder=False)
        model.load_state_dict(state_dict)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )
        t1 = time.time()
        for i in range(0,3):
            y = model(x)
            print("x")
        t2 = time.time()
        interval = (t2-t1)/3
        fps = 1/interval
        print("Average FPS for coco_C was "+str(fps))
        print("Average time for coco_C was "+str(interval))

        model = fastsal.fastsal(pretrain_mode=False, model_type='C')
        state_dict, opt_state = load_weight(salicon_c, remove_decoder=False)
        model.load_state_dict(state_dict)
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
        )
        t1 = time.time()
        for i in range(0,3):
            y = model(x)
            print("x")
        t2 = time.time()
        interval = (t2-t1)/3
        fps = 1/interval
        print("Average FPS for salicon_C was "+str(fps))
        print("Average time for salicon_C was "+str(interval))

    print('All model loaded and tested')