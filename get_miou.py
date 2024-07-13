import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from nets.deeplabv3_plus import DeepLab
from utils.utils_metrics import compute_mIoU, show_results
from utils.utils import cvtColor

def get_miou(miou_mode=0, num_classes=2, name_classes=["_background_", "cracks"], VOCdevkit_path='VOCdevkit'):
    '''
    进行指标评估需要注意以下几点：
    1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常
    2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
    '''
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeepLab(num_classes=num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16)

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        deeplab = deeplab.to(device)

        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)

            # Convert image to RGB
            image = cvtColor(image)

            # Convert image to tensor and move it to the same device as the model
            preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

            with torch.no_grad():
                output = deeplab(image_tensor)
                output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Get prediction and convert to numpy

            image_result = Image.fromarray(output.astype('uint8'))
            image_result.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == "__main__":
    get_miou()
