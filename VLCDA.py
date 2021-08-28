import sys
sys.path.append('./Data_Process')
path_result = "./Latent_representation/"
from Models import *
from Metrics import *
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
from sklearn.externals import joblib
import warnings
import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA
import random
import time
import math
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from sklearn.ensemble import GradientBoostingClassifier 
import pandas as pd
import numpy as np  
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import IsolationForest
import pylab as plt

import lightgbm as lgb
circRNAnumbercode = np.loadtxt(r'.\Datasets\circRNA number ID.txt',dtype=bytes).astype(str)
diseasenumbercode = np.genfromtxt(r'.\Datasets\disease number ID.txt',dtype=str,delimiter='\t')




warnings.filterwarnings('ignore')
######################################################### Setting #####################################################
Dataset = 'features'
Classification = False
Clustering = False
t_SNE = False
########################################## hyper-parameters##############################################################
Epoch_Num = 200
Learning_Rate = 5*1e-4

Hidden_Layer_1 = 1318
Hidden_Layer_2 = 128
################################### Load dataset   ######################################################################
load_data = Load_Data(Dataset)
Features, Labels = load_data.CPU()
Features = torch.Tensor(Features)
################################### Calculate the adjacency matrix #########################################################
if('Adjacency_Matrix' in vars()):
    print('Adjacency matrix is raw')
    pass
else:
    print('Adjacency matrix is caculated by KNN')
    graph = Graph_Construction(Features)
    Adjacency_Matrix = graph.KNN()

################################################## convolution_kernel ##############################################
convolution_kernel = Convolution_Kernel(Adjacency_Matrix)
Adjacency_Convolution = convolution_kernel.Adjacency_Convolution()
############################################ Results  Initialization ###################################################
ACC_VGAE_total = []
NMI_VGAE_total = []
PUR_VGAE_total = []

ACC_VGAE_total_STD = []
NMI_VGAE_total_STD = []
PUR_VGAE_total_STD = []

F1_score = []
#######################################################  Loss Function #################################################
def Loss_Function(Graph_Reconstruction, Graph_Raw, H_2_mean, H_2_std):
    bce_loss = torch.nn.BCELoss(size_average=False)
    Reconstruction_Loss = bce_loss(Graph_Reconstruction.view(-1), Graph_Raw.view(-1))
    KL_Divergence = -0.5 / Adjacency_Matrix.size(0) * (1 + 2 * H_2_std - H_2_mean ** 2 - torch.exp(H_2_std) ** 2).sum(1).mean()
    return Reconstruction_Loss, KL_Divergence

############################################## Model ###################################################################
model_VGAE = myVGAE(Features.shape[1], Hidden_Layer_1, Hidden_Layer_2)
optimzer = torch.optim.Adam(model_VGAE.parameters(), lr=Learning_Rate)
start_time = time.time()
#################################################### Train ###################################################
for epoch in range(Epoch_Num):
    Latent_Representation, Graph_Reconstruction, H_2_mean, H_2_std = model_VGAE(Adjacency_Convolution, Features)
    Reconstruction_Loss, KL_Divergence = Loss_Function(Graph_Reconstruction, Adjacency_Matrix, H_2_mean, H_2_std)
    loss = Reconstruction_Loss + KL_Divergence

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Latent_Representation.cpu().detach().numpy()

end_time = time.time()
print(Latent_Representation.shape)

nc = 604 #number of circRNAs
nd = 88 # number of diseases
na = 659 # number of circRNA-disease associations 
r = 0.5 # Decising the size of feature subset
nn = 604*88-659 # number of unknown samples
# adjacency matrix
A = np.zeros((nc,nd),dtype=float)
ConnectDate = np.loadtxt(r'.\Datasets\known disease-circRNA association number ID.txt',dtype=int)-1 
for i in range(na):
    A[ConnectDate[i,0], ConnectDate[i,1]] = 1 # the element is 1 if the miRNA-disease pair has association

dataset_n = np.argwhere(A == 0)
Trainset_p = np.argwhere(A == 1)

predict_0 =np.zeros((dataset_n.shape[0]+Trainset_p.shape[0]))
random.seed(1)
Trainset_n = dataset_n[random.sample(list(range(nn)),na)]
print("Trainset_n",np.array(Trainset_n).shape)
print("Trainset_p",np.array(Trainset_p).shape)

Trainset= np.vstack((Trainset_n,Trainset_p))  
print("Trainset_n",np.array(Trainset).shape)
Y_value=[]
for i in range(659):
    Y_value.append(0.0)
for i in range(659,1318):
    Y_value.append(1.0)
X_train = Latent_Representation
X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, Y_value, test_size=0.8, random_state=0)
print(Latent_Representation.shape)

gbm = lgb.LGBMClassifier(objective='binary', num_leaves=25,
                                learning_rate=0.1, n_estimators=383, max_depth=7,
                                bagging_fraction=0.7, feature_fraction=0.9, reg_lambda=0.2)
gbm.fit(X1_train, y1_train,eval_set=[(X1_test, y1_test)],early_stopping_rounds=5)


# scores = cross_val_score(model, X_test_new, y1_test, cv=5)
predict_0 = gbm.predict_proba(X1_test)[:,1]
predict_0scoreranknumber =np.argsort(-predict_0)
predict_0scorerank = predict_0[predict_0scoreranknumber]
diseaserankname_pos = Trainset[predict_0scoreranknumber,1]
diseaserankname = diseasenumbercode[diseaserankname_pos,1]
circRNArankname_pos = Trainset[predict_0scoreranknumber,0]
circRNArankname = circRNAnumbercode[circRNArankname_pos,1]
predict_0scorerank_pd=pd.Series(predict_0scorerank)
diseaserankname_pd=pd.Series(diseaserankname)
circRNArankname_pd=pd.Series(circRNArankname)
prediction_0_out = pd.concat([diseaserankname_pd,circRNArankname_pd,predict_0scorerank_pd],axis=1)
prediction_0_out.columns=['Disease','circRNA','Score']
prediction_0_out.to_excel(r'prediction results for all unknown samples2.xlsx', sheet_name='Sheet1',index=False)
