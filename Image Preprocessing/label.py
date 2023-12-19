import os
import cv2
import cv2, json
import numpy as np
import requests as req
from PIL import Image
from io import BytesIO

input_file = open ('./export-2022-12-18T17_26_53.729Z.json')
json_array = json.load(input_file)

for item in json_array:
    src1 = (item["Label"]["objects"][0]["instanceURI"])
    # src2 = (item["Label"]["objects"][1]["instanceURI"])
    # src3 = (item["Label"]["objects"][2]["instanceURI"])
    response1 = req.get(src1)
    
    # response2 = req.get(src2)
    # response3 = req.get(src3)
    img1 = np.array(Image.open(BytesIO(response1.content)))
    # img2 = np.array(Image.open(BytesIO(response2.content)))
    # img3 = np.array(Image.open(BytesIO(response3.content)))
    # print(img1)
    # white = np.array([[255]*256 for _ in range(256)])

    # for i in range(256):
    #     for j in range(256):
    #         if img1[i, j][0] == 255:
    #             white[i, j] = 0
    # cv2.imshow('My Image', img1)

    # # 按下任意鍵則關閉所有視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("./bonelabel/" + item["External ID"], img1)

    img1=np.array(img1,dtype='uint8')
    img1=Image.fromarray(img1)
    img1.save("./bonelabel/" + item["External ID"][:-4] + ".bmp")