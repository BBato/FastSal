import model.fastSal as fastsal
from utils import load_weight
from dataset.utils import resize_interpolate, pytorch_normalze
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from generate_img import post_process_png, post_process_probability2
import time
import onnxruntime

def convert_vgg_img(src, target_size):
    vgg_img = src
    original_size = vgg_img.size
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if target_size[0] != original_size[1] or target_size[1] != original_size[0]:
            vgg_img = vgg_img.resize((target_size[1], target_size[0]), Image.ANTIALIAS)
        elif isinstance(target_size, int):
            vgg_img = vgg_img.resize(
                (
                    int(original_size[0] / target_size),
                    int(original_size[2] / target_size),
                ),
                Image.ANTIALIAS,
            )
    vgg_img = np.asarray(vgg_img, dtype=np.float32)
    vgg_img = pytorch_normalze(torch.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return vgg_img, np.asarray(original_size)


if __name__ == "__main__":

    # model = fastsal.fastsal(pretrain_mode=False, model_type='A')
    # state_dict, opt_state = load_weight('weights/{}_{}.pth'.format('salicon', 'A'), remove_decoder=False)
    # model.load_state_dict(state_dict)
    # model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    # )

    ort_session = onnxruntime.InferenceSession("exported.onnx")

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        t1 = time.process_time()

        # Capture the video frame by frame
        #ret, frame = vid.read()

        # Resize to standard size
        #frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)

        # Convert to PIL object
        #cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #pil_im = Image.fromarray(cv2_im)
        a = torch.zeros((3,240,320))
        
        
        #b = a.numpy()
        pil_im = torchvision.transforms.ToPILImage()(a)
        #print(cv2_im.shape)
        #exit()
        pil_im = pil_im.convert("RGB")
        x, camera_image_size = convert_vgg_img(pil_im, (192, 256))
        x = x[np.newaxis, :, :, :]

        # Run model
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        y = ort_outs[0]

        # Prepare for display
        y = torch.nn.Sigmoid()(torch.tensor(y))

        y = y.detach().numpy()
        for i, prediction in enumerate(y[:, 0, :, :]):
            img_data = post_process_png(prediction, camera_image_size)
            cv2.imshow("img_output_path", img_data)
            #cv2.imshow("original", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        t2 = time.process_time()
        interval = t2 - t1
        fps = 1 / interval
        print("FPS: " + str(fps))

    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()
