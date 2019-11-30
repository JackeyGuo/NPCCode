import os
import sys
import json
import re
import argparse
#import config.config as cfg
#model_name = 'resnet_aspp_5fold_weights'
#model_name = 'mynet'
#model_name = 'vnet_new'
model_name = 'xception_aspp_sqrt'
threshold = 0.6
# softmax or sigmoid
predict_op = 'softmax'
#model_name = 'unet'
#model_name = 'deform_xception_aspp'
if predict_op == 'sigmoid':
    output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/comp_result/%s/score_%.1f"%(model_name,threshold)
    result_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/comp_result/%s/score_%.1f.txt"%(model_name,threshold)
else:
    output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/compared_result/%s/score" % (model_name)
    result_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/result/compared_result/%s/score.txt" % (model_name)
#output_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/two_seg_new/score"
#result_path = r"/home/data_new/guofeng/projects/Segmentation/NPCCode/two_seg_new/score.txt"
json_file = open('/home/data_new/guofeng/projects/Segmentation/NPCCode/result/compared_result/%s/%s.txt'% (model_name,model_name),'w')
assd_file = open('/home/data_new/guofeng/projects/Segmentation/NPCCode/result/compared_result/%s/assd_%s.txt'% (model_name,model_name),'w')
#f1_file = open('/home/data_new/guofeng/projects/Segmentation/NPCCode/result/compared_result/%s/f1_%s.txt'% (model_name,model_name),'w')
def count(files, threshold=0.55):
    counts = {}
    error_file = 0
    result_file = open(result_path,'w')

    for file in files:
        #print(file.split('/')[-1])
        #if file.split('/')[-1] != 'LiuQuanXing.json' and file.split('/')[-1] != '_C11-002u_DICOM_PA5_ST0_SE7.json':
        with open(file) as f:
            try:
                jsinfo = json.load(f)
                filename = file.split('/')[-1]
                json_file.write(filename + ': ' + str(jsinfo[0]['DICE']) + '\n')
                assd_file.write(filename + ': ' + str(jsinfo[0]['ASSD']) + '\n')
                for k in jsinfo[0]:
                    if not counts.get(k):
                        counts[k]=0
                    counts[k] += jsinfo[0][k]
                #print(jsinfo[0]['DICE'])
                # print file name
                if jsinfo[0]['DICE'] <= threshold:
                    error_file += 1
                    filename = file.split('/')[-1]
                    result_file.write(filename + ': ' + str(jsinfo[0]['DICE']) + '\n')
                    print(filename)
            except:
                print('wrong file :', file)
    print('error file number : ', error_file)

    precision = counts['PRECISION'] / len(files)
    recall = counts['RECALL'] / len(files)
    f1_score = round(2 * precision * recall / (precision + recall), 4)
    print('f1_score ', f1_score)
    for k in counts:
        counts[k] = counts[k] / len(files)
        print (k,counts[k])
        result_file.write(str(k) + ' '+ str(round(counts[k],4)) + '\n')
    result_file.write('F1_SCORE' + ' ' + str(f1_score) + '\n')


def runforfile(pat,dir,func):
    repa = re.compile(pat)
    countfiles = []
    for path,dirs,files in os.walk(dir):
        for file in files:
            if (repa.match(file)):
                countfiles.append(os.path.join(path, file))

    if len(countfiles) > 0:
        func(countfiles)


if __name__ == "__main__":
    AP = argparse.ArgumentParser(description='this script count the ASSD,DICE,and some thing others.')
    AP.add_argument('-target',
                    default=output_path)
    AP.add_argument('-pat',default='.*\.json')
    pa = AP.parse_args()
    runforfile(pa.pat,pa.target,count)
