import os
import numpy as np
import SimpleITK as sitk

predict_output_path = "../test_prediction"
bound_output_path = "../test_nicer"

seg_tail = r"SegRAIROIResampling.mha"

def _get_file_name(full_path):
    return os.path.basename(full_path)
seg = []
def create_bound(inputs,output_path):
    for f in inputs:
        #print(f)
        img = sitk.ReadImage(f)
        np_float = sitk.GetArrayFromImage(img)

        np_int = np.array(np_float,dtype=np.int32)
        img_int = sitk.GetImageFromArray(np_int)
        distmap = sitk.SignedMaurerDistanceMap(img_int)
        np_dist_map = sitk.GetArrayFromImage(distmap)
        zero_dist = np.logical_and(np_dist_map < 0.01,np_dist_map > -0.01)
        zero_dist = np.array(zero_dist,dtype=np.float32)
        seg.append(zero_dist)
    # define the range of label 1-3,
    result = seg[0] + seg[1]*2
    #print(np.shape(result[0]))
    opt_image = sitk.GetImageFromArray(result)
    print(f)
    sitk.WriteImage(opt_image,os.path.join(output_path,_get_file_name(f)))

def drow_gt_bound(gt_path, out_path):
    files = os.listdir(gt_path)
    file_list = []
    for f in files:
        if not -1 == f.find(seg_tail):
            file_list.append(os.path.join(gt_path,f))

    create_bound(file_list,out_path)

def drow_bound(target_path,out_path):
    files = os.listdir(target_path)
    file_list = []
    for f in files:
        file_list.append(os.path.join(target_path,f))

    create_bound(file_list,out_path)

if __name__ == "__main__":
    files = os.listdir(predict_output_path)
    #lst = [r"D:\chromeDownload\CT MRI\CT+MRI\test_opt.mha",]
    lst = []
    for f in files:
        lst.append(os.path.join(predict_output_path,f))
    create_bound(lst,bound_output_path)
    #open it while use.
    # drow_gt_bound("/home/gf/guofeng/all_120/all_resized_124","../test_image")
    # drow_bound(r"D:\cuitjobs\newBegin\unet\provider\predict", r"D:\cuitjobs\newBegin\unet\bound")
    # drow_bound(r"D:\cuitjobs\newBegin\densedilatedaspp\provider\predict", r"D:\cuitjobs\newBegin\densedilatedaspp\bound")
    # drow_bound(r"D:\cuitjobs\newBegin\deeplab_like\provider\predict",r"D:\cuitjobs\newBegin\deeplab_like\bound")
    pass


