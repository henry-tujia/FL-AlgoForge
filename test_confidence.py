import torch
import os
import json
import sys
sys.path.append("/mnt/data/th")
import pandas
from FedML.fedml_api.data_preprocessing.covid.data_loader import get_client_idxes_dict, get_client_dataloader
from sklearn.metrics import classification_report
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','retro'])
import wandb
from models.resnet import resnet8


def load_ori_model(pt_path,KD =False):
    model  = resnet8(4,KD=KD)

    paras_old = model.state_dict()
    # print(len(.keys()))
    paras = torch.load(pt_path)

    for key,value in zip(list(paras_old.keys()),list(paras.values())):
        paras_old[key] = value

    model.load_state_dict(paras_old)
    # device = torch.device("cuda")
    model.to(device)

    return model

def find_pt(root_path):

    algs_to_pt = {}

    for d in os.listdir(root_path):
        print(d)
        if os.path.isdir(d):
            if 'config.txt' in os.listdir(os.path.join(root_path,d)):
                with open(os.path.join(root_path,d,'config.txt'),'r') as f:
                    contens = json.load(f)
                    if "covid" == contens['dataset'] and contens['partition_alpha']==0.3:
                        if contens['method'] in algs_to_pt.keys():
                            algs_to_pt[contens['method']].append(os.path.join(root_path,d))
                        else:
                            algs_to_pt.update({contens['method']:[os.path.join(root_path,d)]})

    return algs_to_pt


def eval_model_confidence(model,KD = False):
    model.eval()

    preds = None
    labels = None
    confidence = None

    acc = None
    confidences = None
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            x = x.to(device)
            target = target.to(device)

            if KD:
                hs,_ = model(x)
                ws = model.fc.weight

                pred = hs.mm(ws.transpose(0, 1))

            else:
                pred = model(x)
            probs = torch.softmax(pred,-1)

            # loss = criterion(pred, target)
            _, predicted = torch.max(probs, 1)

            if preds is None:
                preds = predicted.cpu()
                labels = target.cpu()
                confidence = (_**2).cpu()
            else:
                preds = torch.concat((preds,predicted.cpu()),dim=0)
                labels = torch.concat((labels,target.cpu()),dim=0)
                confidence = torch.concat((confidence,(_**2).cpu()),dim=0)
    data_temmp = pandas.DataFrame()
    ECE = 0
    for i in range(10):
        start =i*0.1
        end = start+0.1
        confidence_bin_rate_mean = ((confidence<=end)*(start<confidence)*confidence).mean()
        confidence_bin_rate = ((confidence<=end)*(start<confidence)).float().sum()
        confidence_bin_right = ((labels == preds)*(confidence<=end)*(start<confidence)).float().sum()
        confidence_bin_true_mean = confidence_bin_right/max(confidence_bin_rate,1)
        # confidence_bin_false = ((labels == 8)*(confidence<=end)*(start<confidence)*(preds!=2)).float().sum()
        df_temp = pandas.Series([confidence_bin_rate_mean.item(),confidence_bin_true_mean.item()]).to_frame(str(i))
        data_temmp = pandas.concat([data_temmp,df_temp],axis=1)

        ECE  += torch.abs(confidence_bin_rate_mean-confidence_bin_true_mean)*confidence_bin_rate

    confusion_matrix = classification_report(labels,preds)

    return data_temmp,confusion_matrix,ECE#/labels.shape[0]


def download_model():
    for i in [9,10,11]:#range(13):
        print(i)
        os.environ["HTTPS_PROXY"] = "http://10.21.0.15:7890"
        run = wandb.init()
        artifact = run.use_artifact('henrytujia/FedTH/model:v'+str(i), type='model')
        os.makedirs(os.path.join("./covid_pts/",str(i)))
        artifact_dir = artifact.download(os.path.join("./covid_pts/",str(i)))



if __name__ == "__main__":

    device = torch.device("cuda")

    test_dl = get_client_dataloader("/mnt/data/th/FedTH/data/dataset/covid", 64, {}, client_idx=None, train=False)
    # download_model()
    logs_path = "./covid_pts"
    confidence_bins = {}
    ECE_dict = {}

    # for i in [9,10,11]:
    # for d in os.listdir(os.path.join(logs_path)):
    #     if os.path.isdir(os.path.join(logs_path,d)):
    #         alg = d.split('.')[0]


    
    dict_algs_pts = find_pt(logs_path)

    # print(dict_algs_pts)
    for key,values in dict_algs_pts:
        for value in values:

            if "fedrs" in value :
                KD = True
            else:
                KD = False
            model = load_ori_model(os.path.join(value),KD)

            condifence_bin,confusion_matrix,ECE = eval_model_confidence(model,KD)
            print(value,confusion_matrix)
            # confidence_bins.update({alg:condifence_bin})
            # ECE_dict.update({alg:ECE})
            # print(alg,ECE)



    # for name,ds in confidence_bins.items() :
    #     ds.index = ["Outputs","Acc"]
    #     ds.T.plot(kind="bar")
    #     plt.xlabel('\#Confidence Bin')
    #     plt.ylabel('\% of Data Samples')
    #     plt.ylim((0,ds.loc["All"].sum() ))
    #     bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    #     plt.text(-0.07, ds.loc["All"].sum()-0.2, "ECE:{:.2f}".format(ECE_dict[name]),fontdict={"size":6,"weight":"bold"},bbox =bbox_props)
    #     plt.xticks(rotation=0) # 旋转90度
    #     # plt.title(name)
    #     # sns.barplot(data = df_confidence)
    #     plt.legend(loc="upper left",ncol= 2,fontsize=6)
    #     plt.savefig(name+"_ECE.pdf",dpi = 300)
    #     plt.show()