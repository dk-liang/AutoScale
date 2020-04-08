# import cv2
# import numpy as np
# import scipy.io
# import random
# import torch
# Whilte = [255,255,255]
#
# def areaCal(contour):
#
#     area = 0
#     for i in range(len(contour)):
#         area += cv2.contourArea(contour)
#     return area
#
# def find_max_and_second_large_num(list):
#     first = list[0]
#     second = 0
#     third = 0
#     for i in range(1,len(list)):
#         if list[i] > first:
#             second =first
#             first = list[i]
#             index_first = i
#         elif list[i] > second:
#             second = list[i]
#             index_second = i
#
#         elif list[i] > third:
#             third = list[i]
#             index_third = i
#     return index_first, index_second, index_third
#
# def get_overlap_region(patch_array,original_size):
#     h = original_size[2]
#     w = original_size[3]
#     overlap_map = np.zeros((h,w))
#     # print(patch_array)
#     for i in range(len(patch_array)):
#         box = patch_array[i]
#         overlap_map[box[1]:(box[1] + box[3]),box[0]:(box[0] + box[2])] += 1
#     return overlap_map
#
# def  bboverlap(bbox0, bbox1):
#
#     x0, y0, w0, h0 = bbox0
#     x1, y1, w1, h1 = bbox1
#     #print(x0, y0, w0, h0 , x1, y1, w1, h1)
#     if x0 > x1 + w1 or x1 > x0 + w0 or y0 > y1 + h1 or y1 > y0 + h0:
#         return False
#
#     else:
#         return True
#
# def findmaxcontours(distance_map, find_max, fname,rate,target):
#     #distance_map = distance_map.astype(np.uint8)
#     target = target.data.numpy()
#     target = 255*target/np.max(target)
#     target = target[0].astype(np.uint8)
#     distance_map = 255*distance_map/np.max(distance_map)
#     distance_map = distance_map[0][0]
#     target = cv2.resize(target, (distance_map.shape[1],distance_map.shape[0]))
#     img = distance_map.astype(np.uint8)
#     #cv2.imwrite('./middle_process/input_img.jpg', distance_map)
#     #img = cv2.imread('./middle_process/input_img.jpg')
#     Img = img
#     #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2GRAY)
#     gray =img
#     #cv2.imwrite("./middle_process/gray.jpg", gray)
#
#     Thresh = 5.0/6 * 255.0
#     #print(Thresh)
#     ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)
#     cv2.imwrite("./middle_process/binary_only_roi.jpg", binary)
#     #cv2.imshow("binary", binary)
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #print(len(contours))
#     list_index = []
#     for i in range(len(contours)):
#         list_index.append(areaCal(contours[i]))
#     list_index.sort(reverse = True)
#
#     if len(list_index)<3:
#         print("findContours error")
#         list_index.append(list_index[0])
#         list_index.append(list_index[0])
#
#     if len(list_index) >= 5:
#         list_index = list_index[0:5]
#         index_new = random.sample(list_index, 2)
#         index_new.sort(reverse = True)
#         first = index_new[0]
#         sceond = index_new[1]
#     else:
#         first = list_index[0]
#         sceond = list_index[1]
#
#     if find_max==True:
#         first = list_index[0]
#         sceond = list_index[1]
#         third  = list_index[2]
#
#     first_index = 0
#     sceond_index = 0
#     third_index = 0
#     for i in range(len(contours)):
#         if areaCal(contours[i]) == first:
#             first_index = i
#
#         if areaCal(contours[i]) == sceond:
#             sceond_index = i
#
#         if areaCal(contours[i]) == third:
#             third_index = i
#     cor_array = []
#     cv2.drawContours(img, contours[first_index], -1, (0, 0, 255), 3)
#     x, y, w, h = cv2.boundingRect(contours[first_index])
#     cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 0, 0), 2)
#     # cv2.rectangle(Img, (x+10,y+10) , (int(x+50),int(y+10)),
#     #                               (255, 0, 0), thickness=-1)
#     coordinate_first = [x, y, w, h]
#     cor_array.append([x,y])
#     cv2.drawContours(img, contours[sceond_index], -1, (0, 255, 0), 3)
#     x, y, w, h = cv2.boundingRect(contours[sceond_index])
#     cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     coordinate_second = [x, y, w, h]
#     cor_array.append([x,y])
#     cv2.drawContours(img, contours[third_index], -1, (0, 255, 0), 3)
#     x, y, w, h = cv2.boundingRect(contours[third_index])
#     cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     coordinate_third = [x, y, w, h]
#    # Img = cv2.applyColorMap(Img, cv2.COLORMAP_JET)
#     # if bboverlap(coordinate_first, coordinate_second)==True:
#     #target = cv2.applyColorMap(target, cv2.COLORMAP_JET)
#     #Img = np.hstack((Img,target))
#     #save_path = "./middle_process/contours_result_xu/"+fname[0]
#     # # #print(save_path)
#    # if len(rate) !=0:
#     #    for i in range(2):
#      #       cv2.putText(Img, 'r: '+str(rate[i].cpu().detach().numpy()[0]), (cor_array[i][0]+30,cor_array[i][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
#       #      cv2.imwrite(save_path, Img)
#     #
#     # # cv2.imshow('show', Img)
#     # # cv2.imshow("img", img)
#     return coordinate_first, coordinate_second, coordinate_third
#
# # def main():
# #     img = cv2.imread("./input_img.jpg")
# #     result = findmaxcontours(img)
# #     print(result)
# #
# #
# #
# # if __name__ == '__main__':
# #     main()

import cv2
import numpy as np
import scipy.io
import random
import torch
Whilte = [255,255,255]

def areaCal(contour):

    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour)
    return area

def find_max_and_second_large_num(list):
    first = list[0]
    second = 0
    third = 0
    for i in range(1,len(list)):
        if list[i] > first:
            second =first
            first = list[i]
            index_first = i
        elif list[i] > second:
            second = list[i]
            index_second = i

        elif list[i] > third:
            third = list[i]
            index_third = i
    return index_first, index_second, index_third

def get_overlap_region(patch_array,original_size):
    h = original_size[2]
    w = original_size[3]
    overlap_map = np.zeros((h,w))
    # print(patch_array)
    for i in range(len(patch_array)):
        box = patch_array[i]
        overlap_map[box[1]:(box[1] + box[3]),box[0]:(box[0] + box[2])] += 1
    return overlap_map

def  bboverlap(bbox0, bbox1):

    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1
    #print(x0, y0, w0, h0 , x1, y1, w1, h1)
    if x0 > x1 + w1 or x1 > x0 + w0 or y0 > y1 + h1 or y1 > y0 + h0:
        return False

    else:
        return True

def findmaxcontours(distance_map, find_max, fname):
    #distance_map = distance_map.astype(np.uint8)
    # target = target.data.numpy()
    # target = 255*target/np.max(target)
    # target = target[0].astype(np.uint8)
    distance_map = 255*distance_map/np.max(distance_map)
    distance_map = distance_map[0][0]
    #target = cv2.resize(target, (distance_map.shape[1],distance_map.shape[0]))
    img = distance_map.astype(np.uint8)
    #cv2.imwrite('./middle_process/input_img.jpg', distance_map)
    #img = cv2.imread('./middle_process/input_img.jpg')
    Img = img
    #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2GRAY)
    gray =img
    #cv2.imwrite("./middle_process/gray.jpg", gray)

    Thresh = 8.0/11.0 * 255.0
    #print(Thresh)
    ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)
    #cv2.imwrite("./middle_process/binary_only_roi.jpg", binary)
    #cv2.imshow("binary", binary)
    binary, contours, hierarchy= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    list_index = []
    for i in range(len(contours)):
        list_index.append(areaCal(contours[i]))
    list_index.sort(reverse = True)

    if len(list_index) ==0:
        return [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]
    if len(list_index)<3:
        #print("findContours error",fname)
        list_index.append(list_index[0])
        list_index.append(list_index[0])

    if len(list_index) >= 5:
        list_index = list_index[0:5]
        index_new = random.sample(list_index, 2)
        index_new.sort(reverse = True)
        first = index_new[0]
        sceond = index_new[1]
    else:
        first = list_index[0]
        sceond = list_index[1]

    if find_max==True:
        first = list_index[0]
        sceond = list_index[1]
        third  = list_index[2]

    first_index = 0
    sceond_index = 0
    third_index = 0
    for i in range(len(contours)):
        if areaCal(contours[i]) == first:
            first_index = i

        if areaCal(contours[i]) == sceond:
            sceond_index = i

        if areaCal(contours[i]) == third:
            third_index = i
    cor_array = []
    cv2.drawContours(img, contours[first_index], -1, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(contours[first_index])
    cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # cv2.rectangle(Img, (x+10,y+10) , (int(x+50),int(y+10)),
    #                               (255, 0, 0), thickness=-1)
    coordinate_first = [x, y, w, h]
    cor_array.append([x,y])
    #cv2.drawContours(img, contours[sceond_index], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(contours[sceond_index])
    #cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    coordinate_second = [x, y, w, h]
    cor_array.append([x,y])
    #cv2.drawContours(img, contours[third_index], -1, (0, 255, 0), 3)
    x, y, w, h = cv2.boundingRect(contours[third_index])
    #cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    coordinate_third = [x, y, w, h]
   # Img = cv2.applyColorMap(Img, cv2.COLORMAP_JET)
    # if bboverlap(coordinate_first, coordinate_second)==True:
    #target = cv2.applyColorMap(target, cv2.COLORMAP_JET)
    #Img = np.hstack((Img,target))
    save_path = "./middle_process/contours_result_8/"+fname[0]
    # # #print(save_path)
   # if len(rate) !=0:
    #    for i in range(2):
     #       cv2.putText(Img, 'r: '+str(rate[i].cpu().detach().numpy()[0]), (cor_array[i][0]+30,cor_array[i][1]+30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    #print(save_path)
    cv2.imwrite(save_path.replace('h5','jpg'), Img)
    #
    # # cv2.imshow('show', Img)
    # # cv2.imshow("img", img)
    return coordinate_first, coordinate_second, coordinate_third, contours

# def main():
#     img = cv2.imread("./input_img.jpg")
#     result = findmaxcontours(img)
#     print(result)
#
#
#
# if __name__ == '__main__':
#     main()
