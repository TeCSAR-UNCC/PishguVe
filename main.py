# # This file is based on the following git repository: 
# 1) https://github.com/TeCSAR-UNCC/Pishgu
# and
# 2) https://github.com/TeCSAR-UNCC/CARPe_Posterum/blob/master/utils/network.py

# The paper can be cited as follows:
# @article{noghre2022pishgu,
#   title={Pishgu: Universal Path Prediction Architecture through Graph Isomorphism and Attentive Convolution},
#   author={Noghre, Ghazal Alinezhad and Katariya, Vinit and Pazho, Armin Danesh and Neff, Christopher and Tabkhi, Hamed},
#   journal={arXiv preprint arXiv:2210.08057},
#   year={2022}
# }


import torch
import time
import os
from statistics import mean
from thop import profile
from torch_geometric.data.batch import Batch as tgb
import numpy as np

import utils.loader as dl
import utils.network as net
import utils.util as ut
from tqdm import tqdm

data_files = ["Carolinas_eyelevel", "NGSIM", "Carolinas_highangle"]

input_data_sec = 3
output_horizon_sec = 5
data_frame_rate = 5


# # For input of 3 seconds we need 15 frames input 
OBS_STEP = input_data_sec*data_frame_rate 
# 25 frames output
PRED_STEP = output_horizon_sec*data_frame_rate

NUM_POINTS_PER_POS = 2 #(x,y)
SAVE_MODEL=True
TRAIN=False

output_size = NUM_POINTS_PER_POS*PRED_STEP
num_features = NUM_POINTS_PER_POS*OBS_STEP

try_num_veh = 0

folder_dir = "./Trajectory_Datasets/"

def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all//n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames*i

        if i == n_horiz-1:
            en_id = n_all-1
        else:
            en_id = n_frames*i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id+1])

    return avg_res



if TRAIN==True:
    saveFolder = 'train0' # folder name where models will be saved
    lr = 0.01
    EPOCHS=300
    print("Train file: " + data_files[0])
    print("Training:")   


    for test_file in tqdm(data_files):
        
        if SAVE_MODEL==True:
            if not os.path.exists("models/"+str(saveFolder)+"/"):
                    os.makedirs("models/"+str(saveFolder)+"/")

        # GPU is set here
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv(num_features, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        data_dir = folder_dir+test_file+'/'
        

        data_folders = ['test','train', 'val']
    
        _, train_loader = dl.data_loader(data_dir+data_folders[1])
        _, val_loader = dl.data_loader(data_dir+data_folders[2])
        

        best_info = [1000.0, 1000.0, 0]

       
        for epoch in tqdm(range(0, EPOCHS)):
            
            # Training
            losses = ut.train(model, train_loader, optimizer, device, obs_step=OBS_STEP)
            
            # Validation
            ade, fde = ut.test(model, val_loader, device)
            if ade < best_info[0]:
                best_info[0] = ade
                best_info[1] = fde
                best_info[2] = epoch
                print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
                if (SAVE_MODEL==True):
                 
                    model_path = "./models/"+str(saveFolder)+"/"+ test_file+"_PishguVe"+  ".pt"
                    torch.save(model.state_dict(), model_path)
                # print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
        print(test_file + "   Best ADE: " + str(best_info[0]) + "   FDE: " + str(best_info[1]) + "   Epoch: " + str(best_info[2]) + "   lr: " + str(lr) + "\n\n")


else:
    print("Testing")
    all_ops = []
    times = []

    epoch_no = 60

    for test_file in tqdm(data_files):
        saveFolder = 'train0'
        total_traj = 0
        
        device = torch.device('cpu')
        model = net.NetGINConv(num_features, output_size).to(device)

        # model path
        model_folder = './models/'
        
        model.load_state_dict(torch.load(model_folder+str(saveFolder)+"/" + test_file + "_PishguVe" + ".pt", map_location='cpu'))
        
        data_dir = folder_dir+test_file+'/'

        data_folders = ['test','train', 'val']

        _, test_loader = dl.data_loader(data_dir+data_folders[0], batch_size=1)

        ade_batches, fde_batches = [], []
        try_rmse_batch = torch.full((25,1), 0.0)
        try_rmse_batch = try_rmse_batch.squeeze(dim=1)

        try_i = 0

        model.eval()
        for batch in test_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end, frame_id) = batch
            total_traj += pred_traj_gt.size(0)

            data_list = ut.getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
            graph_batch = tgb.from_data_list(data_list)

            start = time.time()
            pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
            end = time.time()
            times.append(end-start)

            # pred_traj = pred_traj.reshape(pred_traj.shape[0],12,2).detach()
            pred_traj = pred_traj.reshape(pred_traj.shape[0],25,2).detach()

            x_try = pred_traj.shape
            try_num_veh += pred_traj.shape[0]

            pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

            ade_batches.append(torch.sum(ut.displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
            fde_batches.append(torch.sum(ut.final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())

            try_rmse_batch += (ut.rmse(pred_traj_real, pred_traj_gt, mode='raw'))#.detach().item()
            try_i+=1

            ops, params = 0,0
            all_ops.append(ops)

        
        rmse = try_rmse_batch/try_i



        ade = sum(ade_batches) / (total_traj * 25)
        fde = sum(fde_batches) / (total_traj)
        print(test_file + "  ADE: " + str(ade) + "  FDE: " + str(fde))

        print("RMSE: 1s,2s,3s,4s,5s")
        pred_fde_horiz = horiz_eval(rmse, 5)
        print(pred_fde_horiz)

    total_params = sum(p.numel() for p in model.parameters())
    print("New Total number of parameters: ", total_params)

    


