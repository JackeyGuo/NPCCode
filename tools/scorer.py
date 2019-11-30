import os
import config.config as cfg

#cfg.name = 'resnet_aspp_5fold'
#cfg.name = 'xception_aspp'
#name = 'vnet_sqrt_new'
name = 'compared_result/deeplab_jacc'
#name = 'cnn_v2_softmax'
threshold = 0.6
truth_path = cfg.data_path
# softmax or sigmoid
predict_op = 'softmax'
score_bin_dir = r"/home/gf/guofeng/ImageValidationISLES/bin"
seg_tail = r"T1SegRAIROIResampling.mha"

if predict_op == 'sigmoid':
    output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/comp_result/%s/score_%.1f"%(name,threshold)
    predict_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/comp_result/%s/predict_%.1f"%(name,threshold)
else:
    output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/score" % (name)
    predict_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/%s/predict" % (name)

def score(t_p_o_list):
    os.chdir(score_bin_dir)
    for i in t_p_o_list:
        truth = i[0]
        predict = i[1]
        output = i[2]
        os.system("./ImageValidationISLES " + truth + " " + predict + " " + output)
        # if sys.platform.find("win"):
        #     print(__file__, ":using windows scorer")
        #     os.system("ImageValidationISLES.exe " + truth + " " + predict + " " + output)
        # else:
        #     print(__file__, ":using linux scorer")
        #     os.system("ImageValidationISLES " + truth + " " + predict + " " + output)

def _create_t_p_o_list():
    t_p_o_list = []
    pre_list = os.listdir(predict_path)
    for f in pre_list:
        name = f.split('.')[0]
        t = os.path.join(truth_path,name + seg_tail)
        p = os.path.join(predict_path,f)
        o = os.path.join(output_path,name + ".json")
        t_p_o_list.append((t,p,o))
    return t_p_o_list

def create_t_p_o_list(truth_path, predict_path, output_path):
    t_p_o_list = []
    pre_list = os.listdir(predict_path)
    for f in pre_list:
        name = f.split('.')[0]
        t = os.path.join(truth_path, name + seg_tail)
        p = os.path.join(predict_path, f)
        o = os.path.join(output_path, name + ".json")
        t_p_o_list.append((t, p, o))
    return t_p_o_list


if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tpo = _create_t_p_o_list()
    score(tpo)