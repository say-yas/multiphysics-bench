import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_E_flow
import os
from scipy import io as sio
import numpy as np
import tqdm
import json
import pandas as pd

device = 'cuda'

if __name__ == '__main__':
    save_dir = './eval_results/E_flow'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    range_allec_V_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/ec_V/range_allec_V.mat"
    range_allec_V = sio.loadmat(range_allec_V_paths)['range_allec_V']

    max_ec_V = range_allec_V[0, 1]
    min_ec_V = range_allec_V[0, 0]

    # load max_u_flow min_u_flow
    range_allu_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/u_flow/range_allu_flow.mat"
    range_allu_flow = sio.loadmat(range_allu_flow_paths)['range_allu_flow']

    max_u_flow = range_allu_flow[0, 1]
    min_u_flow = range_allu_flow[0, 0]

    # load max_v_flow min_v_flow
    range_allv_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/v_flow/range_allv_flow.mat"
    range_allv_flow = sio.loadmat(range_allv_flow_paths)['range_allv_flow']

    max_v_flow = range_allv_flow[0, 1]
    min_v_flow = range_allv_flow[0, 0]

    # Let's load the TE_heat dataset.
    train_loader, test_loaders, data_processor = load_E_flow(
            n_train=10000, batch_size=16,
            test_resolutions=[128], n_tests=[1000],
            test_batch_sizes=[64],
    )
    data_processor = data_processor.to(device)
    data_processor.eval()

    model = FNO(n_modes=(12, 12),
                 in_channels=1,
                 out_channels=3,
                 hidden_channels=128,
                 projection_channel_ratio=2)
    model = model.to(device)

    model.max_ec_V = max_ec_V
    model.min_ec_V = min_ec_V
    model.max_u_flow = max_u_flow
    model.min_u_flow = min_u_flow
    model.max_v_flow = max_v_flow
    model.min_v_flow = min_v_flow

    model.load_state_dict(torch.load("/data/bailichen/PDE/PDE/FNO/examples/models/checkpoints/E_Flow/1/model_epoch_49_state_dict.pt", weights_only=False))
    print("Model weights loaded from model_weights.pt")

    # Set the model to evaluation mode
    model.eval()

    u_metric_total, v_metric_total, T_metric_total, sample_total = 0, 0, 0, 0
    res_dict = {
        'RMSE': {'u': 0, 'v': 0, 'T': 0},
        'nRMSE': {'u': 0, 'v': 0, 'T': 0},
        'MaxError': {'u': 0, 'v': 0, 'T': 0},
        'fRMSE': {},
        'bRMSE': {'u': 0, 'v': 0, 'T': 0},
    }

    def get_nRMSE():
        u,v,T = pred
        u_metric = torch.norm(u - outputs[:, 0, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 0, :, :], 2, dim=(1, 2))
        v_metric = torch.norm(v - outputs[:, 1, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 1, :, :], 2, dim=(1, 2))
        T_metric = torch.norm(T - outputs[:, 2, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 2, :, :], 2, dim=(1, 2))

        res_dict['nRMSE']['u'] += u_metric.sum()
        res_dict['nRMSE']['v'] += v_metric.sum()
        res_dict['nRMSE']['T'] += T_metric.sum()

    def get_RMSE():
        u, v, T = pred  # pred contains the model predictions (B, C, H, W)
        # Compute RMSE for each channel (averaged over spatial dimensions, per sample)

        u_metric = torch.sqrt(torch.mean((u - outputs[:, 0, :, :]) ** 2, dim=(1, 2)))
        v_metric = torch.sqrt(torch.mean((v - outputs[:, 1, :, :]) ** 2, dim=(1, 2)))
        T_metric = torch.sqrt(torch.mean((T - outputs[:, 2, :, :]) ** 2, dim=(1, 2)))

        # Accumulate into the results dictionary
        res_dict['RMSE']['u'] += u_metric.sum()
        res_dict['RMSE']['v'] += v_metric.sum()
        res_dict['RMSE']['T'] += T_metric.sum()

    def get_MaxError():
        u, v, T = pred
        # Compute the maximum absolute error for each channel (along the spatial dimensions)
        u_metric = torch.abs(u - outputs[:, 0, :, :]).flatten(1).max(dim=1)[0]  # flatten first, then take max
        v_metric = torch.abs(v - outputs[:, 1, :, :]).flatten(1).max(dim=1)[0]  # flatten first, then take max
        T_metric = torch.abs(T - outputs[:, 2, :, :]).flatten(1).max(dim=1)[0]  # flatten first, then take max
        # Accumulate results
        res_dict['MaxError']['u'] += u_metric.sum()
        res_dict['MaxError']['v'] += v_metric.sum()
        res_dict['MaxError']['T'] += T_metric.sum()

    def get_bRMSE():
        u, v, T = pred
        # Extract boundary pixels (1 pixel wide on each of the top, bottom, left, and right edges)
        boundary_mask = torch.zeros_like(outputs[:, 0, :, :], dtype=bool)
        boundary_mask[:, 0, :] = True  # top boundary
        boundary_mask[:, -1, :] = True  # bottom boundary
        boundary_mask[:, :, 0] = True  # left boundary
        boundary_mask[:, :, -1] = True  # right boundary

        # Compute boundary RMSE
        u_boundary_pred = u[boundary_mask].view(u.shape[0], -1)
        u_boundary_true = outputs[:, 0, :, :][boundary_mask].view(u.shape[0], -1)
        u_metric = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['u'] += u_metric.sum()

        v_boundary_pred = v[boundary_mask].view(v.shape[0], -1)
        v_boundary_true = outputs[:, 1, :, :][boundary_mask].view(v.shape[0], -1)
        v_metric = torch.sqrt(torch.mean((v_boundary_pred - v_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['v'] += v_metric.sum()

        T_boundary_pred = T[boundary_mask].view(T.shape[0], -1)
        T_boundary_true = outputs[:, 2, :, :][boundary_mask].view(T.shape[0], -1)
        T_metric = torch.sqrt(torch.mean((T_boundary_pred - T_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['T'] += T_metric.sum()

    def get_fRMSE():
        u, v, T = pred  # pred shape: (Batch, Channel, Height, Width)

        # Initialize result storage
        for freq_band in ['low', 'middle', 'high']:
            res_dict['fRMSE'][f'u_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'v_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'T_{freq_band}'] = 0.0

        # Define frequency band ranges (based on the paper's settings)
        freq_bands = {
            'low': (0, 4),  # k_min=0, k_max=4
            'middle': (5, 12),  # k_min=5, k_max=12
            # k_min=13, k_max=∞ (use the Nyquist frequency in practice)
        }

        def compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W):
            """Compute the fRMSE for the specified frequency band"""
            # Build a mask selecting frequencies within the band
            kx = torch.arange(H, device=pred_fft.device)
            ky = torch.arange(W, device=pred_fft.device)
            kx, ky = torch.meshgrid(kx, ky, indexing='ij')

            # Compute radial wavenumbers (avoid duplicate counting of 0 and Nyquist frequencies)
            r = torch.sqrt(kx ** 2 + ky ** 2)
            if k_max is None:
                mask = (r >= k_min)
                k_max = max(H // 2, W // 2) #nyquist
            else:
                mask = (r >= k_min) & (r <= k_max)

            # Compute the error
            diff_fft = torch.abs(pred_fft - true_fft) ** 2
            band_error = diff_fft[:, mask].sum(dim=1)  # sum over spatial dimensions
            band_error = torch.sqrt(band_error) / (k_max - k_min + 1)
            return band_error

        # Compute fRMSE for each channel
        for channel_idx, (pred_ch, true_ch, name) in enumerate([
            (u, outputs[:, 0, :, :], 'u'),
            (v, outputs[:, 1, :, :], 'v'),
            (T, outputs[:, 2, :, :], 'T')
        ]):
            # Fourier transform (after shift, low frequencies are centered)
            pred_fft = torch.fft.fft2(pred_ch)
            true_fft = torch.fft.fft2(true_ch)
            H, W = pred_ch.shape[-2], pred_ch.shape[-1]

            # Compute each frequency band
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{name}_{band}'] += error.sum()

    # Start testing
    for idx, sample in enumerate(tqdm.tqdm(test_loaders[128])):
        with torch.no_grad():
            print(sample['y'].mean())
            sample = data_processor.preprocess(sample)
            print(sample['y'].mean())
            inputs,outputs = sample['x'].to(device),sample['y'].to(device).squeeze()
            pred_outputs = model(inputs)
            pred_outputs, _ = data_processor.postprocess(pred_outputs)
            pred_outputs = pred_outputs.squeeze()

            # # GT de-normalization
            # outputs[:, 0, :, :] = ((outputs[:, 0, :, :] + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(
            #     torch.float64)
            # outputs[:, 1, :, :] = ((outputs[:, 1, :, :] + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(
            #     torch.float64)
            # outputs[:, 2, :, :] = ((outputs[:, 2, :, :] + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(
            #     torch.float64)
            #
            # #Pred de-normalization
            # u_u_N = ((pred_outputs[:, 0, :, :] + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(
            #     torch.float64)
            # u_v_N = ((pred_outputs[:, 1, :, :] + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(
            #     torch.float64)
            # T_N = ((pred_outputs[:,2,:,:] + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(
            #     torch.float64)
            # pred = (u_u_N,u_v_N,T_N)

            pred = (pred_outputs[:, 0, :, :], pred_outputs[:, 1, :, :], pred_outputs[:, 2, :, :])

            get_RMSE()
            get_nRMSE()
            get_MaxError()
            get_bRMSE()
            get_fRMSE()

            sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            res_dict[metric][var] /= sample_total
            res_dict[metric][var] = res_dict[metric][var].item()


    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')
    # TODO: save log
    with open(os.path.join(save_dir, f'log_final.json'), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False)


    data = res_dict
    res = []
    for metric in data:
        for var in data[metric]:
            res.append(data[metric][var])

    output_file = os.path.join(save_dir, './exp.csv')
    frmse_df = pd.DataFrame(res)
    frmse_df.to_csv(output_file, index=False, encoding="utf-8", float_format="%.16f")
