import model.fastSal as fastsal
from utils import load_weight
import torch
import time
import torch.onnx
import onnxruntime
from PIL import Image
import numpy as np

if __name__ == "__main__":
    coco_c = "weights/coco_C.pth"  # coco_C
    coco_a = "weights/coco_A.pth"  # coco_A
    salicon_c = "weights/salicon_C.pth"  # salicon_C
    salicon_a = "weights/salicon_A.pth"  # coco_A

    with torch.no_grad():

        x = torch.zeros((1, 3, 192, 256))

        model = fastsal.fastsal(pretrain_mode=False, model_type="A")
        state_dict, opt_state = load_weight(coco_a, remove_decoder=False)
        model.load_state_dict(state_dict)
        model.eval()

        # print("Generating sample...")
        # torch_out = model(x)

        # torch.onnx.export(
        #     model,  # model being run
        #     x,  # model input (or a tuple for multiple inputs)
        #     "exported.onnx",  # where to save the model (can be a file or file-like object)
        #     export_params=True,  # store the trained parameter weights inside the model file
        #     opset_version=10,  # the ONNX version to export the model to
        #     do_constant_folding=True,  # whether to execute constant folding for optimization
        #     input_names=["input"],  # the model's input names
        #     output_names=["output"],  # the model's output names
        # )

        ort_session = onnxruntime.InferenceSession("exported.onnx")

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        t1 = time.time()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        y = ort_outs[0]

        # Prepare for display
        y = torch.nn.Sigmoid()(torch.tensor(y))

        y = y.detach().numpy()
        for i, prediction in enumerate(y[:, 0, :, :]):
            img_data = post_process_png(prediction, camera_image_size)
            cv2.imshow('img_output_path', img_data)
            cv2.imshow('original', frame)

        # print(img_out_y.shape)
        # img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')


        # # get the output image follow post-processing step from PyTorch implementation
        # final_img = Image.merge(
        #     "YCbCr", [
        #         img_out_y,
        #         img_cb.resize(img_out_y.size, Image.BICUBIC),
        #         img_cr.resize(img_out_y.size, Image.BICUBIC),
        #     ]).convert("RGB")

        # # Save the image, we will compare this with the output image from mobile device
        # final_img.save("./_static/img/cat_superres_with_ort.jpg")

        t2 = time.time()
        interval = (t2 - t1)
        fps = 1 / interval
        print("Average FPS was " + str(fps))
        print("Average time was " + str(interval))

        # t1 = time.time()
        # for i in range(0, 3):
        #     y = model(x)
        #     print("x")
        # t2 = time.time()
        # interval = (t2 - t1) / 3
        # fps = 1 / interval
        # print("Average FPS for coco_A was " + str(fps))
        # print("Average time for coco_A was " + str(interval))

    #     model = fastsal.fastsal(pretrain_mode=False, model_type="A")
    #     state_dict, opt_state = load_weight(salicon_a, remove_decoder=False)
    #     model.load_state_dict(state_dict)
    #     model = torch.quantization.quantize_dynamic(
    #         model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    #     )
    #     t1 = time.time()
    #     for i in range(0, 3):
    #         y = model(x)
    #         print("x")
    #     t2 = time.time()
    #     interval = (t2 - t1) / 3
    #     fps = 1 / interval
    #     print("Average FPS for salicon_A was " + str(fps))
    #     print("Average time for salicon_A was " + str(interval))

    #     model = fastsal.fastsal(pretrain_mode=False, model_type="C")
    #     state_dict, opt_state = load_weight(coco_c, remove_decoder=False)
    #     model.load_state_dict(state_dict)
    #     model = torch.quantization.quantize_dynamic(
    #         model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    #     )
    #     t1 = time.time()
    #     for i in range(0, 3):
    #         y = model(x)
    #         print("x")
    #     t2 = time.time()
    #     interval = (t2 - t1) / 3
    #     fps = 1 / interval
    #     print("Average FPS for coco_C was " + str(fps))
    #     print("Average time for coco_C was " + str(interval))

    #     model = fastsal.fastsal(pretrain_mode=False, model_type="C")
    #     state_dict, opt_state = load_weight(salicon_c, remove_decoder=False)
    #     model.load_state_dict(state_dict)
    #     model = torch.quantization.quantize_dynamic(
    #         model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    #     )
    #     t1 = time.time()
    #     for i in range(0, 3):
    #         y = model(x)
    #         print("x")
    #     t2 = time.time()
    #     interval = (t2 - t1) / 3
    #     fps = 1 / interval
    #     print("Average FPS for salicon_C was " + str(fps))
    #     print("Average time for salicon_C was " + str(interval))

    # print("All model loaded and tested")
