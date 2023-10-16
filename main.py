import time
import tkinter as tk
from get_resource import resource_path
from tkinter import *
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageTk
from torchvision import transforms as T

root = Tk()

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # x1 = F.pad(x1,[torch.div(diffX, 2, rounding_mode='trunc'), torch.div(diffX - diffX, 2, rounding_mode='trunc'),
        #                 torch.div(diffY, 2, rounding_mode='trunc'), torch.div(diffY - diffY, 2, rounding_mode='trunc')])
        # torch.div(a, b, rounding_mode='trunc')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 16)
        # self.down1 = Down(16, 32)
        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        # self.down4 = Down(128, 128)
        # self.up1 = Up(256, 64, bilinear)
        # self.up2 = Up(128, 32, bilinear)
        # self.up3 = Up(64, 16, bilinear)
        # self.up4 = Up(32, 16, bilinear)
        # self.outc = OutConv(16, n_classes)

        self.inc = DoubleConv(n_channels, 16 * 4)
        self.down1 = Down(16 * 4, 32 * 4)
        self.down2 = Down(32 * 4, 64 * 4)
        self.down3 = Down(64 * 4, 128 * 4)
        self.down4 = Down(128 * 4, 128 * 4)
        self.up1 = Up(256 * 4, 64 * 4, bilinear)
        self.up2 = Up(128 * 4, 32 * 4, bilinear)
        self.up3 = Up(64 * 4, 16 * 4, bilinear)
        self.up4 = Up(32 * 4, 16 * 4, bilinear)
        self.outc = OutConv(16 * 4, n_classes)

        # self.inc = DoubleConv(n_channels, 16*8)
        # self.down1 = Down(16*8, 32*8)
        # self.down2 = Down(32*8, 64*8)
        # self.down3 = Down(64*8, 128*8)
        # self.down4 = Down(128*8, 128*8)
        # self.up1 = Up(256*8, 64*8, bilinear)
        # self.up2 = Up(128*8, 32*8, bilinear)
        # self.up3 = Up(64*8, 16*8, bilinear)
        # self.up4 = Up(32*8, 16*8, bilinear)
        # self.outc = OutConv(16*8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def form_2D_label(mask, class_map):
    # mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2], dtype=np.uint8)

    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i

    return label


'''This converts predicted map to RGB labels'''


def map_this(y_pred, class_map):
    y_pred_rgb = np.zeros((y_pred.shape[1], y_pred.shape[2], 3))
    for j in range(y_pred.shape[1]):
        for k in range(y_pred.shape[2]):
            y_pred_rgb[j, k, :] = class_map[y_pred[0][j][k]]
    return y_pred_rgb


def plot_result(img, title):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.imshow(img)
    plt.show()





def classify(img):
    height, width, _ = img.shape
    # print(height,width)
    img = cv2.resize(img, (width - 256, height - 256))
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gt1 = gray.flatten()
    mpn_class = ['Benign', 'Essential Thrombocythemia','Polycythemia vera', 'Myelofibrosis']
    mpn_grey_value = [76,29,179,226]
    class_pixels = []
    for value in mpn_grey_value:
        class_pixels.append(np.sum(gt1 == value))
    print(class_pixels)
    index = np.argmax(class_pixels)
    # print(index)
    print(mpn_class[index])
    return mpn_class[index]

def predict(image):
    global time_taken_min
    global time_taken_sec
    class_map = [[0., 0., 0.], [0., 255., 0.], [255., 0., 0.], [0., 0., 255.], [0., 255., 255.], [255., 255., 0.]]
    model = UNet(n_channels=3, n_classes=6)
    file_path = resource_path('6_class_Unet(16_4)_with_aug.pth')
    state = torch.load(file_path, map_location='cpu')
    model.load_state_dict(state["model_state_dict"])
    start = time.time()
    height, width, _ = image.shape
    flag, temp_flag = 0, 0
    final_mask_wo = np.zeros((height - 256, width - 256, 3), dtype=np.uint8)
    t_height, t_width, _ = final_mask_wo.shape
    temp_w, temp_h = 0, 0
    mask_h1, mask_h2, mask_w1, mask_w2 = 0, 256, 0, 256
    while temp_h + 512 < height or temp_w + 512 < width:
        if temp_w + 512 > width:
            w1 = width - 512
            w2 = width
            h1 = temp_h
            h2 = temp_h + 512
            temp_h += 256
            flag += 1
            temp_w = 0
        elif temp_h + 512 > height:
            h1 = height - 512
            h2 = height
            w1 = temp_w
            w2 = temp_w + 512
            temp_w += 256
        else:
            w1 = temp_w
            w2 = temp_w + 512
            h1 = temp_h
            h2 = temp_h + 512
            temp_w += 256
        temp_img = image[h1:h2, w1:w2]
        # print('H1 --- ', h1, 'H2 --- ', h2, 'W1 --- ', w1, 'W2 --- ', w2)
        # print('H1 --- ', mask_h1, 'H2 --- ', mask_h2, 'W1 --- ', mask_w1, 'W2 --- ', mask_w2)
        # print('--------------')
        # plot_result(temp_img,"Input")
        t = T.Compose([T.ToTensor()])
        temp_img = t(temp_img)
        temp_img = temp_img[None, :]
        temp_img = temp_img.to('cpu', dtype=torch.float)
        output = model(temp_img)
        output_soft = F.softmax(output, dim=1)
        output_num = output_soft.detach().numpy()
        pred_mask = np.argmax(output_num, axis=1)
        y_pred_rgb = map_this(pred_mask, class_map)
        y_pred_rgb = np.array(y_pred_rgb, dtype=np.uint8)
        if mask_w2 > t_width:
            mask_w1 = t_width - 256
            mask_w2 = t_width
        if mask_h2 > t_height:
            mask_h1 = t_height - 256
            mask_h2 = t_height
        final_mask_wo[mask_h1:mask_h2, mask_w1:mask_w2] = y_pred_rgb[128:384, 128:384]
        mask_w1 += 256
        mask_w2 += 256
        if flag != temp_flag:
            mask_h1 += 256
            mask_h2 += 256
            mask_w1 = 0
            mask_w2 = 256
            temp_flag = flag
    # final_mask[h1+128:h2-128,w1+128:w2-128] = y_pred_rgb[128:384,128:384]
    # y_pred_rgb = cv2.cvtColor(y_pred_rgb, cv2.COLOR_BGR2RGB)
    # plot_result(y_pred_rgb,"output")
    # cv2.imwrite('test_msk.png', cv2.cvtColor(final_mask_wo, cv2.COLOR_RGB2BGR))
    end = time.time()
    time_taken_min = (end - start) // 60.0
    time_taken_sec = round((end - start) % 60.0,2)
    print('Total Time : ', time_taken_min, 'mins , ', time_taken_sec, 'secs')
    return final_mask_wo
    # return cv2.cvtColor(final_mask_wo, cv2.COLOR_RGB2BGR)
    # plot_result(image, "Input")
    # plot_result(mask, "Mask")
    # plot_result(final_mask_wo, "Final output")


def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_mask = predict(image)
        mpn_class = classify(pred_mask)
        text = " \nTime taken : " + str(time_taken_min) + "min," + str(time_taken_sec) + "secs"
        textArea.insert('1.0', mpn_class)
        textArea.insert('2.0', text)
        scale_percent = 25  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # pred_mask = cv2.resize(pred_mask, dim, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
        pred_mask = cv2.resize(pred_mask, (600, 600), interpolation=cv2.INTER_AREA)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels

        # convert the images to PIL format...
        image = Image.fromarray(image)
        mask = Image.fromarray(pred_mask)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        mask = ImageTk.PhotoImage(mask)

        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=mask)
            panelB.image = mask
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=mask)
            panelA.image = image
            panelB.image = mask



panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
textArea = tk.Text(root, height=15, width=20, wrap=tk.WORD)
# textArea.insert('1.0', 'Click the above button')
# text = "Click the above button to predict \n\nTime taken : " + str(time_taken_min) + "mins," + str(time_taken_sec) + "sec"
# text = "Click the above button to predict \nTime taken : {} mins, {} secs.\n".format(time_taken_min,time_taken_sec)
# textArea.insert(tk.END, text)
textArea.pack(side="bottom")

btn = Button(root, text="Select an image", bg='black', fg='Red', command=select_image)
btn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
# btn.pack(side="bottom")
# kick off the GUI
root.mainloop()
