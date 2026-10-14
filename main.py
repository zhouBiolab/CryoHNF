import os
import sys
import time
from datetime import datetime as dt
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_search.pose_encoder.cryoai import CryoAI
from pose_search.pose_find import lie_tools
from pose_search.pose_find.mask import get_circular_mask
from pose_search.pose_find.mask import CircularMask
from torch.utils.data import DataLoader
from core.pose import PoseTracker
from pose_search.pose_find import pose_search2
from pose_search.pose_find import pose_search
from data import dataset
from data import ctf
from core.lattice import Lattice
from models.models_SCUNet3D2 import HetOnlyVAE
from core import utils
from data import mrc
from core import fft
import argparse
from data import cryoio
from pathlib import Path

torch.backends.cudnn.deterministic = True

log = utils.log
vlog = utils.vlog


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def add_args(parser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--zdim", type=int, required=True, help="Dimension of latent variable, ####"
    )
    parser.add_argument(
        "--poses", type=os.path.abspath, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--mask", metavar="mrc", type=os.path.abspath, help="Optional mask (.mrc, default: sphere mask)"
    )
    parser.add_argument(
        "--split",
        choices=("1", "2"),
        help="Split dataset for computing GSFSC"
    )
    parser.add_argument(
        "--load", metavar="WEIGHTS.PKL", help="Initialize training from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1,
        help="Checkpointing interval in N_EPOCHS (default: %(default)s)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval in N_IMGS (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, 100000), help="Random seed"
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--uninvert-data",
        dest="invert_data",
        action="store_false",
        help="Do not invert data sign",
    )

    group.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="Turn off real space windowing of dataset",
    )
    group.add_argument(
        "--window-r",
        type=float,
        default=0.85,
        help="Windowing radius (default: %(default)s)",
    )
    group.add_argument(
        "--ind", type=os.path.abspath, help="Filter particle stack by these indices"
    )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )

    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Minibatch size (default: %(default)s)",
    )
    group.add_argument(
        "--wd",
        type=float,
        default=0,
        help="Weight decay in Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate in AdamW optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--norm",
        type=float,
        nargs=2,
        default=[0, 1],
        help="Data normalization as shift, 1/scale (default: 0, 1)",
    )
    group.add_argument(
        "--prior-weight",
        type=float,
        default=1e-3,
        help="Weight for the normalized prior-consistency loss (default: %(default)s)",
    )
    group.add_argument(
        "--beta",
        default=0,
        help="KLD loss weight",
    )

    group = parser.add_argument_group("Network Architecture")
    group.add_argument(
        "--dec-layers",
        dest="players",
        type=int,
        default=2,
        help="Number of decoder hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--dec-dim",
        dest="pdim",
        type=int,
        default=256,
        help="Number of nodes in decoder hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--enc-layers",
        dest="qlayers",
        type=int,
        default=3,
        help="Number of encoder hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--enc-dim",
        dest="qdim",
        type=int,
        default=256,
        help="Number of nodes in encoder hidden layers (default: %(default)s)",
    )

    group.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--l-extent",
        type=float,
        default=0.5,
        help="Coordinate lattice size (if not using positional encoding) (default: %(default)s)",
    )
    group.add_argument(
        "--pe-type",
        choices=(
            "hash_grid",
            "geom_ft",
            "geom_full",
            "geom_lowf",
            "geom_nohighf",
            "linear_lowf",
            "gaussian",
            "none",
        ),
        default="hash_grid",
        help="Type of positional encoding (default: %(default)s)",
    )
    group.add_argument(
        "--pe-dim",
        type=int,
        default=32,
        help="Num sinusoid features in positional encoding (default: 32)",
    )
    group.add_argument(
        "--activation",
        choices=("relu", "leaky_relu"),
        default="relu",
        help="Activation (default: %(default)s)",
    )
    group.add_argument(
        "--pose_estimation",
        choices=("gt", "abinit_hps", "abinit_encoder", "refine"),
        default="gt",
        help="pose_estimation in hartley space",
    )
    group.add_argument(
        "--domain",
        choices=("real", "hartley", "all"),
        default="real",
        help="reconstruct space",
    )
    group.add_argument(
        "--refine_net",
        choices=("scunet",),
        default="scunet",
        help="refine_net type",
    )

    return parser


def calc_fsc(vol1, vol2):
    """
    Helper function to calculate the FSC between two (assumed masked) volumes
    vol1 and vol2 should be maps of the same box size, structured as numpy arrays with ndim=3, i.e. by loading with
    cryodrgn.mrc.parse_mrc
    """
    # load masked volumes in fourier space

    vol1_ft = np.fft.fftshift(np.fft.fftn(vol1))
    vol2_ft = np.fft.fftshift(np.fft.fftn(vol2))

    # define fourier grid and label into shells
    Dx, Dy, Dz = vol1.shape
    x = np.arange(-Dx // 2, Dx // 2)
    y = np.arange(-Dy // 2, Dy // 2)
    z = np.arange(-Dz // 2, Dz // 2)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    r_max = max(Dx, Dy, Dz) // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max, r_step)
    bin_labels = np.searchsorted(bins, r, side="right")

    # calculate the FSC via labeled shells
    num = ndimage.sum(
        np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins + 1
    )
    den1 = ndimage.sum(np.abs(vol1_ft) ** 2, labels=bin_labels, index=bins + 1)
    den2 = ndimage.sum(np.abs(vol2_ft) ** 2, labels=bin_labels, index=bins + 1)
    fsc = num / np.sqrt(den1 * den2)

    x = bins / Dx  # x axis should be spatial frequency in 1/px
    return x, fsc


def eval_vol(
        model,
        lattice,
        mask,
        norm,
        optim,
        epoch,
        pose_search_step,
        Apix=1.,
        flip=False,
        out_weights=None,
        outdir=None,
        out_vol=None,
):
    assert not model.training

    D = lattice.D
    if mask is None:
        mask = lattice.get_sphere_mask(D // 2)
    y_recon_3d, z_mu, z_logvar, similarity_per_sample = model(coords=lattice.coords[None], mask=mask[None])
    vol_recon = y_recon_3d['recon']
    vol_recon_mlp = y_recon_3d['recon_mlp']
    print("vol_recon", vol_recon.shape)
    vol_recon = vol_recon * norm[1] + norm[0]
    vol_recon_mlp = vol_recon_mlp * norm[1] + norm[0]
    if out_vol is not None and vol_recon is not None:
        mrc.write(out_vol['recon'], vol_recon.squeeze(0).detach().cpu().numpy(), Apix=Apix, is_vol=True)
        mrc.write(out_vol['recon_mlp'], vol_recon_mlp.squeeze(0).detach().cpu().numpy(), Apix=Apix, is_vol=True)
    save_checkpoint(model, optim, epoch, z=None, pose_search_step=pose_search_step, outdir=outdir)


def get_rot_trans(model, pose_model, y, y_HT, ctf_gen, lattice, ps_params, ind, pose_search_step,
                  rot_gt, trans_gt, epoch, args, transtoreal=1, dict=None):
    rot = rot_gt if rot_gt is not None else None
    trans = trans_gt if trans_gt is not None else None
    if trans_gt is None or rot_gt is None:
        if args.pose_estimation == "abinit_hps":
            if pose_search_step["is_in_pose_search_step"]:
                if epoch >= 0:  # and not pose_search_step["training_schedule"][epoch]:
                    with torch.no_grad():
                        model.decoder.eval()
                        if args.zdim > 0:
                            z_mu, z_logvar = model.encode(img=y, ind=ind)
                            z = model.reparameterize(z_mu, z_logvar)
                        else:
                            z = None
                        t0 = time.time()
                        if 0 <= epoch <= transtoreal:
                            rot, trans, _ = pose_search.opt_theta_trans(
                                model, y_HT, lattice, ps_params['ht'], z=z, ctf_i=ctf_gen['hartley'],
                                gt_trans=trans_gt if trans_gt is None else trans_gt.flip(-1)
                            )
                            t1 = time.time()
                            # print("pose_search_time", t1-t0)
                            pose_search_step["predicted_rots"][ind] = rot.clone().detach().cpu()
                            pose_search_step["predicted_trans"][ind] = trans.clone().detach().cpu()
                            model.decoder.train()
                        elif epoch > transtoreal:
                            dict['transtoreal'] = True
                            rot, trans, _ = pose_search2.opt_theta_trans(
                                model, y, lattice, ps_params['real'], z=z, ctf_i=ctf_gen['hartley'],
                                gt_trans=trans_gt if trans_gt is None else trans_gt.flip(-1)
                            )
                            t1 = time.time()
                            # print("pose_search_time", t1-t0)
                            pose_search_step["predicted_rots"][ind] = rot.clone().detach().cpu()
                            pose_search_step["predicted_trans"][ind] = trans.clone().detach().cpu()
                            model.decoder.train()
                    '''pose_dict = {
                        'R': rot,
                        'index': in_dict['index']
                    }
                    if not self.no_trans:
                        pose_dict['t'] = trans'''
                    # model.decoder.eval()
            elif pose_search_step["use_point_estimates"]:
                if pose_search_step["first_switch_to_point_estimates"]:
                    pose_search_step["first_switch_to_point_estimates"] = False
                    log("Initializing pose table from"
                        "hierarchical pose search")
                    model.pose_table.initialize(pose_search_step["predicted_rots"],
                                                pose_search_step["predicted_trans"])
                pose_dict = model.pose_table(ind)
                rot = pose_dict['R'].to(lattice.device)
                trans = pose_dict['t'].to(lattice.device)
            if pose_search_step["is_in_pretrain"]:  # or pose_search_step["training_schedule"][epoch]:
                rot = pose_search_step["predicted_rots"][ind].to(lattice.device)
                trans = pose_search_step["predicted_trans"][ind].to(lattice.device) if trans_gt is None else trans_gt
        elif args.pose_estimation == "abinit_encoder":
            '''if pose_model_train["in_pretrain"]:
                model.train()
                pose_model.train()
                rot = pose_model_train["predicted_rots"][ind].to(lattice.device)
                trans = pose_model_train["predicted_trans"][ind].to(lattice.device)
            if pose_model_train["training_schedule"][epoch]:
                pose_model.eval()
                model.train()
                rot, trans = pose_model(y_HT)
            elif not pose_model_train["training_schedule"][epoch]:
                model.eval()
                pose_model.train()'''
            rot, trans = pose_model(y_HT)

            print("pose_model rot", rot.shape, rot)
            print("pose_model trans", trans)
    return rot, trans


def calcu_coord(y, pose_search_step, trans, rot, lattice, B, D, trans_gt, epoch, args, transtoreal=1):
    coords = {}
    if pose_search_step["is_in_pose_search_step"] or pose_search_step["use_point_estimates"] or pose_search_step[
        "is_in_pretrain"]:
        # coords = (lattice.coords.repeat(B, 1, 1) / lattice.extent / 2) @ rot
        # y_gt_processed = lattice.translate_rt(y, trans.unsqueeze(1)).reshape(B, D, D)
        # trans = trans.flip(-1)
        if 0 <= epoch <= transtoreal:
            rot[..., [0, 1], :] = rot[..., [1, 0], :]
        trans = trans.flip(-1)
        coords['real'] = (lattice.coords.repeat(B, 1, 1) / lattice.extent / 2 + F.pad(trans, [0, 1]).unsqueeze(
            1) / D) @ rot if trans_gt is None else ((lattice.coords.repeat(B, 1, 1) / lattice.extent / 2 + F.pad(
            trans_gt, [0, 1]).unsqueeze(1) / D) @ rot)
        if 0 <= epoch <= transtoreal:
            rot[..., [0, 1], :] = rot[..., [1, 0], :]
        trans = trans.flip(-1)
        coords['hartley'] = lattice.coords_ht @ rot if args.domain != 'real' else None
        '''lattice.coords.repeat(B, 1, 1) / lattice.extent / 2  @ rot'''
        y_gt_processed = y
    else:
        with torch.no_grad():
            coords['real'] = ((lattice.coords.repeat(rot.shape[0], 1, 1) / lattice.extent / 2 + F.pad(trans,
                                                                                                      [0, 1]).unsqueeze(
                1).repeat(rot.shape[0] // B, 1, 1) / D) @ rot) if trans_gt is None else ((lattice.coords.repeat(
                rot.shape[0], 1, 1) / lattice.extent / 2 + F.pad(trans_gt, [0, 1]).unsqueeze(1).repeat(
                rot.shape[0] // B, 1,
                1) / D) @ rot)
            rot[..., [0, 1], :] = rot[..., [1, 0], :]
            coords['hartley'] = lattice.coords_ht @ rot if args.domain != 'real' else None
            rot[..., [0, 1], :] = rot[..., [1, 0], :]
            y_gt_processed = y
    # print("rot", rot.shape)
    return coords, y_gt_processed


def eval_volume_method(model, lattice, z_dim, norm, zval=None, radius=None):
    """
    hypervolume: HyperVolume
    lattice: Lattice
    z_dim: int
    norm: (mean, std)
    zval: [z_dim]
    radius: int
    """
    coords = lattice.coords_ht
    extent = lattice.extent
    resolution = lattice.D + 1
    if radius is None:
        radius = resolution // 2
    radius_normalized = extent * 2 * radius / resolution
    z = None
    if zval is not None:
        z = torch.tensor(zval, dtype=torch.float32, device=coords.device).reshape(1, z_dim)

    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    model.eval()
    assert not model.training
    with torch.no_grad():
        for i, dz in enumerate(np.linspace(-extent, extent, resolution, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            x = x.reshape(1, -1, 3)
            y = model.decoder(x, in_posesearch=True, domain='hartley', coords_ht=x)
            slice_radius = int(np.sqrt(max(radius_normalized ** 2 - dz ** 2, 0.)) * resolution)
            slice_mask = CircularMask(lattice, slice_radius).binary_mask
            y['recon_hartley'][0, ~slice_mask] = 0.0
            y['recon_hartley'] = y['recon_hartley'].view(resolution, resolution).detach().cpu().numpy()
            volume[i] = y['recon_hartley']
        volume = volume * norm[1] + norm[0]
        volume_real = fft.ihtn_center(volume[0:-1, 0:-1, 0:-1])  # remove last +k freq for inverse FFT
    model.train()
    return volume_real


def train_batch(
        model,
        pose_model,
        lattice,
        mask,
        ctf_gen,
        y,
        y_HT,
        rot_gt,
        trans_gt,
        ind,
        epoch,
        pose_search_step,
        pose_model_train,
        ps_params,
        optimizers,
        scaler,
        args,
        gradient_accumulation,
        pretrain=False,
        outdir=None,
):
    if gradient_accumulation['global_step'] % gradient_accumulation['accum_steps'] == 0:
        for key in optimizers:
            optimizers[key].zero_grad()
    model.train()
    # Cast operations to mixed precision if using torch.cuda.amp.GradScaler()
    B = y.size(0)
    D = lattice.D
    transtoreal = 0
    gen_loss = 0
    mygo = {'transtoreal': False}
    # with torch.no_grad():
    # rot_lossl2 = 0
    # print("trans_gt", trans_gt)
    # print("rot_gt", rot_gt)
    # print("y.dtype", type(y), y.dtype)
    # print("y_HT.dtype", type(y_HT), y_HT.dtype)
    rot, trans = get_rot_trans(model, pose_model, y, y_HT, ctf_gen, lattice, ps_params, ind, pose_search_step,
                               rot_gt, trans_gt, epoch, args, transtoreal=transtoreal, dict=mygo)
    '''rot = torch.tensor([[[0.38609147, 0.82618785, 0.41030124],
                         [0.90462804, -0.4261471, 0.00684476],
                         [0.18050376, 0.3685273, -0.9119244]]], device=lattice.device)
    trans = torch.tensor([[-0.02577796*D, 0.0705817*D]], device=lattice.device)
    print("🤡🤡rotrot", to_numpy(rot))
    print("🤡🤡transtrans", to_numpy(trans))'''

    # trans *= -1
    # print("transtranstrans", F.pad(trans, [0, 1]).unsqueeze(1) / D)
    coords, y_gt_processed = calcu_coord(y, pose_search_step, trans, rot, lattice, B, D, trans_gt,
                                         epoch, args, transtoreal=transtoreal)
    # coords = (lattice.coords.repeat(B, 1, 1) / lattice.extent / 2) @ rot
    # y_gt_processed = lattice.translate_rt(y, trans.unsqueeze(1)).reshape(B, D, D)

    '''print("🤡🤡transtranstrans", F.pad(trans, [0, 1]).unsqueeze(1) / D)
    print("🤡🤡rotrotrot", rot)
    print("🤡🤡coords", coords[0, 0])
    flog(f"🤡🤡{ind} trans is :{trans}")
    flog(f"🤡🤡{ind} rot is :{rot}")
    print(f"🤡🤡Rotation angles (in radians):🤡🤡Alpha: {torch.atan2(rot[0, ..., 2, 1], rot[0, ..., 2, 2]).item()}, "
          f"Beta: {torch.atan2(-rot[0, ..., 2, 0], torch.sqrt(rot[0, ..., 0, 0] ** 2 + rot[0, ..., 1, 0] ** 2)).item()}, "
          f"Gamma: {torch.atan2(rot[0, ..., 1, 0], rot[0, ..., 0, 0]).item()}")'''

    # Read rotation matrices
    '''file_path = '/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/recon/9076/y_recon_3d/pose_search_xyz.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("\nThis is a new file.")  # optionally write initial content
    with open(file_path, 'r') as file:
        original_content = file.read()
    with open(file_path, 'w') as file:
        file.write(f"\nind {ind} : Alpha: {torch.atan2(rot[0, ..., 2, 1], rot[0, ..., 2, 2]).item()}, "
                   f"Beta: {torch.atan2(-rot[0, ..., 2, 0], torch.sqrt(rot[0, ..., 0, 0] ** 2 + rot[0, ..., 1, 0] ** 2)).item()},"
                   f"Gamma:{torch.atan2(rot[0, ..., 1, 0], rot[0, ..., 0, 0]).item()}")
        file.write(original_content)'''

    with torch.no_grad():
        if mask is not None:
            mask = F.grid_sample(mask.permute([2, 1, 0]).repeat(B, 1, 1, 1)[:, None], coords.view(B, D, D, D, 3) * 2,
                                 mode='nearest', align_corners=True).reshape(B, D, D, D)
            mask = (mask > 0).bool()
        else:
            mask = lattice.get_sphere_mask(D // 2).repeat(B, 1, 1, 1)  # D//2,D // 3

    # model.train()
    # if pose_search_step["training_schedule"][epoch]:
    with torch.cuda.amp.autocast():
        print("pretrain :", pose_search_step["is_in_pretrain"] or
              pose_search_step["is_in_pose_search_step"] or
              args.pose_estimation == "abinit_encoder")
        y_recon_3d, z_mu, z_logvar, similarity_per_sample = model(y.to(lattice.device), ind=ind,
                                                                  coords=coords['real'], mask=mask,
                                                                  pretrain=epoch <= -1 or
                                                                           pose_search_step[
                                                                               "is_in_pretrain"] or
                                                                           pose_search_step[
                                                                               "is_in_pose_search_step"] or
                                                                           args.pose_estimation == "abinit_encoder",
                                                                  img_ht=y_HT.to(lattice.device),
                                                                  domain=args.domain,
                                                                  coords_ht=coords['hartley'],
                                                                  dict=mygo)
        y_recon = y_recon_3d['recon'].mean(-1)
        y_psf = torch.fft.fftshift(torch.fft.irfft2(
            torch.fft.rfft2(torch.fft.ifftshift(y_recon.float())) * torch.fft.fftshift(ctf_gen['real'])
            .repeat_interleave(y_recon.shape[0] // ctf_gen['real'].shape[0], dim=0)[..., :D // 2 + 1]))
        if args.domain != 'real' and mygo['transtoreal'] is False:
            trans_ht = trans[..., None, :].reshape(-1, 1, 2) if trans_gt is \
                                                                None else trans[..., None, :].reshape(-1, 1, 2).flip(-1)
            y_gt_processed = lattice.translate_ht(y_HT.reshape(
                -1, (lattice.D + 1) ** 2), trans_ht, freqs2d=model.freqs2d).reshape(
                B, -1)
            # y_gt = fft.ihtn_center(y_gt_processed.reshape(B, D+1, D+1)[..., 0:-1, 0:-1])
            mask_2d = get_circular_mask(lattice, (D + 1) // 2)
            '''print("mask_2d", mask_2d.shape)
            print("(mask_2d>0).sum().shape", (mask_2d > 0).sum())
            print("ctf_gen hartley", ctf_gen['hartley'].shape)
            print("recon_hartley", y_recon_3d['recon_hartley'].shape)'''
            recon_hartley = y_recon_3d['recon_hartley'].view(B, -1)[:, mask_2d]
            recon_hartley *= ctf_gen['hartley'].view(B, -1)[:, mask_2d]
            gen_loss_ht = F.mse_loss(recon_hartley, y_gt_processed[..., mask_2d])
            gen_loss += gen_loss_ht
        else:
            gen_loss_ht = 0
        '''import pickle
        # print("y_recon", y_recon.shape)
        with open('/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/num0_ctfs.pkl', 'wb') as file:
            pickle.dump(ctf_gen.detach().cpu().numpy(), file)'''

        '''mrc.write(
            "/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/recon/9076/posesearch/" + 'y_recon{}{}.mrc'.format(
                ind, time.time()),
            y_recon.detach().cpu().numpy(), 1, is_vol=False)
        mrc.write(
            "/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/recon/9076/posesearch/" + 'y_recon_ctf{}{}.mrc'.format(
                ind, time.time()),
            y_psf.detach().cpu().numpy(), 1, is_vol=False)'''
        # latent loss
        kld = torch.mean(
            -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1), dim=0
        ) if args.zdim > 0 else 0
        # print("y_psf", y_psf.shape)
        # print("y_gt_processed", y_gt_processed.shape)

        if args.pose_estimation == "abinit_encoder":
            # if pose_search_step["is_in_pretrain"] or pose_search_step["is_in_pose_search_step"] or args.pose_estimation == "abinit_encoder":
            # y_psf = fft.ht2_center(y_psf)
            # y_psf = fft.symmetrize_ht(y_psf)
            # y_gt_processed = fft.ht2_center(y_gt_processed)
            # rot_loss = F.mse_loss(lie_tools.rotmat_to_s2s2(pose_model_train['predicted_rots'][ind]).to(y_psf.device),
            #                       lie_tools.rotmat_to_s2s2(rot))
            # gen_loss = F.mse_loss(y_psf, y_gt_processed.repeat_interleave(y_psf.shape[0]//y_gt_processed.shape[0],dim=0).to(y_psf.device))#  + rot_loss + rot_lossl2
            # print("gen_loss", gen_loss.size())
            batch_head_loss = ((torch.abs(
                y_psf - y_gt_processed.repeat_interleave(y_psf.shape[0] // y_gt_processed.shape[0], dim=0).to(
                    y_psf.device))).sum((-1, -2)))  # B x M
            print("batch_head_loss", batch_head_loss.shape)
            min_argmin = torch.min(batch_head_loss.reshape(B, -1), 1)
            print("min_argmin", min_argmin)
            batch_min_head_loss = min_argmin[0]  # B
            gen_loss += batch_min_head_loss.mean()
        elif epoch >= 0 or args.domain == "real":
            '''if y_recon_3d['ij'][0] and y_recon_3d['ij'][1] is not None:
                gen_loss += F.mse_loss(y_psf[:, y_recon_3d['ij'][0] + 6:y_recon_3d['ij'][0] + 30,
                                       y_recon_3d['ij'][1] + 6:y_recon_3d['ij'][1] + 30, ...],
                                       y.to(y_psf.device)[:, y_recon_3d['ij'][0] + 6:y_recon_3d['ij'][0] + 30,
                                       y_recon_3d['ij'][1] + 6:y_recon_3d['ij'][1] + 30, ...])
            else:'''
            gen_loss += F.mse_loss(y_psf, y.to(y_psf.device))

        l1_loss_noise = torch.sum(torch.abs(y_recon_3d['recon_mlp']))
        unet_l1_loss_noise = y_recon_3d['unet_pred']
        l1_loss_Negative = torch.sum(
            torch.abs(torch.minimum(y_recon_3d['recon_mlp'], torch.zeros_like(y_recon_3d['recon_mlp']))))
        num_negative_voxels = ((y_recon_3d['recon_mlp'] < 0).clone().detach()).sum() + 1
        print("negative_num", num_negative_voxels, torch.max(y_recon_3d['recon']))
        # print("y_recon_3d", y_recon_3d['recon'].shape)
        # print("y have Nan", torch.isnan(y).any())
        # print("y_HT have Nan", torch.isnan(y_HT).any())
        # print("y_recon_3d have Nan", torch.isnan(y_recon_3d).any())
        if gradient_accumulation['global_step'] % 300 == 0:
            if args.domain != 'real':
                volume_real = eval_volume_method(model, lattice, 0, [0, 2902.8], zval=None, radius=None)
                mrc.write(
                    outdir + 'y_recon_3d_ht/' + 'y_recon_3d_ht_[{}]___{}.mrc'.format(
                        ind[0, ...], D), volume_real.astype(np.float32, copy=False), 3, is_vol=True)
            mrc.write(
                outdir + 'y_recon_3d/' + 'y_recon_3d_[{}]___{}.mrc'.format(
                    ind[0, ...], D), y_recon_3d['recon'][0, ...].clone().detach().cpu().numpy(), 1.3, is_vol=True)
            mrc.write(
                outdir + 'y_recon.mrc',
                y_recon.clone().detach().cpu().numpy(), 3, is_vol=False)
            mrc.write(
                outdir + 'y_recon_ctf.mrc',
                y_psf.clone().detach().cpu().numpy(), 3, is_vol=False)
            mrc.write(
                outdir + 'y_gt.mrc',
                y.clone().detach().cpu().numpy(), 3, is_vol=False)

        loss = gen_loss + kld / (mask > 0).sum() * B * args.beta + + args.prior_weight * similarity_per_sample \
            # + (l1_loss_Negative / num_negative_voxels * B / 10)
        # + (unet_l1_loss_noise / (mask > 0).sum() * B / 1000) \
        # kld / (mask > 0).sum() * B * args.beta +
        # similarity_per_sample / 1000

    scaler.scale(loss / gradient_accumulation['accum_steps']).backward()
    gradient_accumulation['global_step'] += 1
    if pose_model is not None:
        for name, param in pose_model.named_parameters():
            if param.grad is None:
                print(f"No gradient for {name}")
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"Gradient for {name} contains NaN values")
                    # flog(f"Gradient for {name} contains NaN values")
                if param.grad.abs().mean() > 1000:
                    print(f"Gradient for {name}: {param.grad.abs().mean()}")
                    # flog(f"Gradient for {name}: {param.grad.abs().mean()}")
    for name, param in model.named_parameters():
        '''if param.grad is None:
            print(f"No gradient for {name}")'''
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Gradient for {name} contains NaN values")
                # flog(f"Gradient for {name} contains NaN values")
            if param.grad.abs().mean() > 1000:
                print(f"Gradient for {name}: {param.grad.abs().mean()}")
                # flog(f"Gradient for {name}: {param.grad.abs().mean()}")
    if gradient_accumulation['global_step'] % gradient_accumulation['accum_steps'] == 0:
        for key in optimizers:
            scaler.step(optimizers[key])
        scaler.update()
    '''else:
        loss = 0
        gen_loss = 0
        kld = 0
        l1_loss_noise = 0
        l1_loss_Negative = 0
        num_negative_voxels = 0
        similarity_per_sample = 0
        gen_loss_ht = 0'''
    return (to_numpy(loss), to_numpy(gen_loss), to_numpy(kld), to_numpy(l1_loss_noise) / to_numpy(
        mask > 0).sum() * B, to_numpy(l1_loss_Negative) / to_numpy(num_negative_voxels) * B,
            to_numpy(similarity_per_sample), to_numpy(gen_loss_ht))


def eval_z(
        model,
        data,
        batch_size,
        device,
        args=None
):
    assert not model.training
    batch_it = 0
    log_interval = args.log_interval * (batch_size // args.batch_size)
    z_mu_all = []
    z_logvar_all = []
    ind_all = []
    data_generator = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for minibatch in data_generator:
        ind = minibatch['index']
        y = minibatch['y'].to(device)
        y_HT = minibatch['y_HT'].to(device)
        batch_it += len(ind)
        z_mu, z_logvar = model(y, ind=ind, encode_only=True, img_ht=y_HT)
        z_mu_all.append(z_mu.detach().cpu().numpy())
        z_logvar_all.append(z_logvar.detach().cpu().numpy())
        ind_all.append(ind.detach().cpu().numpy())
        if batch_it % log_interval == 0:
            log(
                "# [Generation of latent space] {}/{} images".format(batch_it, len(data))
            )
    z_mu_all = np.vstack(z_mu_all)
    z_logvar_all = np.vstack(z_logvar_all)
    ind_all = np.concatenate(ind_all)
    return {'z_mu': z_mu_all, 'z_logvar': z_logvar_all, 'ind': ind_all}


def save_checkpoint(model, optimizers, epoch, z, pose_search_step=None, outdir=None):
    """Save model weights, latent encoding z, and decoder volumes"""
    # save model weights
    optimizers_state_dict = {}
    for key in optimizers.keys():
        optimizers_state_dict[key] = optimizers[key].state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizers_state_dict,
            "pose_search_step": pose_search_step if pose_search_step else None,
        },
        outdir + 'weights.{}.pkl'.format(epoch),
    )
    # save z
    if z is not None:
        utils.save_pkl(z, outdir + 'z.{}.pkl'.format(epoch))


def train(data, ctf_params, posetracker, mask, device, ckpt_file=None, outdir=None, args=None):
    t1 = dt.now()
    os.makedirs(outdir + 'y_recon_3d', exist_ok=True)
    os.makedirs(outdir + 'y_recon_3d_ht', exist_ok=True)
    start_epoch = 0
    batch_size = args.batch_size
    num_epochs = args.num_epochs if (
            args.zdim > 0 or args.pose_estimation != "gt" or args.pe_type != 'hash_grid' or args.num_epochs < 3) else 3
    checkpoint = args.checkpoint
    log_interval = args.log_interval * batch_size
    Nimg = len(data)
    D = data.D
    '''if args.pose_estimation != 'gt':
        args.domain = 'all'''
    Dz = D
    extent = args.l_extent
    lattice = Lattice(D, Dz, extent, device=device, endpoint=False)
    freq_mask = lattice.get_circular_mask(D // 2)

    in_dim = (D + 1) ** 2
    activation = nn.ReLU if args.activation == 'relu' else nn.LeakyReLU
    qlayers = args.qlayers
    qdim = args.qdim
    players = args.players
    pdim = args.pdim

    ps_params = None
    if args.pose_estimation == "abinit_hps":
        ps_params = {'real': {
            'l_min': D,  # 32,  # 12,
            'l_max': D,  # 72,  # 32,
            't_extent': 20.0,
            't_n_grid': 7,
            'niter': 4,
            'nkeptposes': 6,
            'base_healpy': 2,
            't_xshift': 0,
            't_yshift': 0,
            'no_trans_search_at_pose_search': False,
            'n_tilts_pose_search': None,
            'tilting_func': None,
            'average_over_tilts': False},
            'ht': {
                'l_min': 12,  # 32,  # 12,
                'l_max': 32,  # 72,  # 32,
                't_extent': 20.0,
                't_n_grid': 7,
                'niter': 4,
                'nkeptposes': 8,
                'base_healpy': 2,
                't_xshift': 0,
                't_yshift': 0,
                'no_trans_search_at_pose_search': False,
                'n_tilts_pose_search': None,
                'tilting_func': None,
                'average_over_tilts': False}}

    model = HetOnlyVAE(lattice, qlayers, qdim, players, pdim, in_dim, Nimg, zdim=args.zdim, enc_type=args.pe_type,
                       enc_dim=args.pe_dim, activation=activation, domain=args.domain,
                       ps_params=ps_params, pose_ues=args.pose_estimation, refin_type=args.refine_net)

    if args.refine_net == "scunet":
        CKPT_NAME = "model.pth"
        CKPT_PATH = Path(__file__).resolve().with_name(CKPT_NAME)
        ckpt = torch.load(str(CKPT_PATH), map_location='cpu')
        model.decoder.refin_net.load_state_dict(ckpt)

    for param in model.decoder.refin_net.parameters():
        param.requires_grad = False

    model.to(device)
    print("args.domain", args.domain)
    first_switch_to_point_estimates = True
    ###

    pose_model = None
    pose_model_train = dict()
    if args.pose_estimation == "abinit_encoder":
        pose_model = CryoAI(device)
        pose_model.to(device)
        pose_model_train['predicted_rots'] = lie_tools.random_rotmat(np.prod(Nimg), device='cpu')
        pose_model_train["predicted_trans"] = torch.zeros((Nimg, 2), device='cpu')
        pose_model_train['in_pretrain'] = True
        pose_model_train["training_schedule"] = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        start_epoch = -1

    pose_search_step = dict()
    if args.pose_estimation == "abinit_hps":
        pose_search_step["predicted_rots"] = lie_tools.random_rotmat(np.prod(Nimg), device='cpu')
        # pose_search_step["predicted_rots"] = np.eye(3).reshape(1, 3, 3).repeat(Nimg, axis=0)
        pose_search_step["predicted_trans"] = torch.zeros((Nimg, 2), device='cpu')
        pose_search_step["first_switch_to_point_estimates"] = first_switch_to_point_estimates
        pose_search_step["is_in_pretrain"] = True
        pose_search_step["training_schedule"] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        start_epoch = -1
    elif args.pose_estimation == "refine":
        pose_search_step["first_switch_to_point_estimates"] = False
        pose_search_step["is_in_pose_search_step"] = False
        pose_search_step["use_point_estimates"] = True
        pose_search_step["is_in_pretrain"] = False
    else:
        pose_search_step["is_in_pose_search_step"] = False
        pose_search_step["is_in_pretrain"] = False
        pose_search_step["use_point_estimates"] = False

    if posetracker is not None and args.pose_estimation != "gt":
        log("Initializing pose table from ground truth")
        rotmat_gt, trans_gt = posetracker.get_pose(abinit=True, device="cpu")
        model.pose_table.initialize(rotmat_gt, trans_gt)
        pose_search_step["first_switch_to_point_estimates"] = False
    elif posetracker is None and args.pose_estimation == "gt":
        raise RuntimeError("No Pose input and pose estimation is set to 'gt'.")
    # epochs_pose_search = max(2, 500000 // Nimg + 1)
    epochs_pose_search = max(2, 5)

    if ckpt_file:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        if args.pose_estimation == "abinit_hps":
            pass
            # pose_search_step["predicted_rots"] = ckpt["pose_search_step"]["predicted_rots"]
            # pose_search_step["predicted_trans"] = ckpt["pose_search_step"]["predicted_trans"]
        if start_epoch >= epochs_pose_search:
            pose_search_step["first_switch_to_point_estimates"] = False

    data_generator = DataLoader(
        data, batch_size=batch_size, num_workers=12, pin_memory=True,
        shuffle=True,
    )

    lr = args.lr
    wd = args.wd

    optimizer_params = []
    optimizers = dict()
    '''optimizers['model'] = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)'''
    scunet_params = list(model.decoder.decoder[1].parameters())
    mlp_params = list(model.decoder.decoder[0].parameters())

    optimizer_params.append({'params': scunet_params, 'lr': lr, 'weight_decay': wd})
    optimizer_params.append({'params': mlp_params, 'lr': lr, 'weight_decay': wd})

    if args.pose_estimation == "abinit_hps":
        optimizers['pose_encoder'] = torch.optim.AdamW(model.pose_table.parameters(), lr=1.0e-3, weight_decay=wd)
        # pose_encoder_params = list(pose_model.parameters())
        # optimizer_params.append({'params': pose_encoder_params, 'lr': lr, 'weight_decay': wd})

    if args.pe_type == "hash_grid" and hasattr(model.decoder, 'hash_grid_encoding'):
        # For zdim>0 HetOnlyVAE overrides enc_type to geom_ft, so the decoder has no hash_grid_encoding
        hash_grid_params = list(model.decoder.hash_grid_encoding.parameters())  # get hash_grid_encoding parameters
        optimizer_params.append({'params': hash_grid_params, 'lr': lr, 'weight_decay': wd})

    optimizers['volume'] = torch.optim.AdamW(optimizer_params)
    # optim_volume = torch.optim.Adam(model.decoder.parameters(), lr=lr, weight_decay=wd)
    if args.zdim > 0:
        optimizers['conf'] = torch.optim.AdamW(model.encoder.parameters(),
                                               lr=1.0e-2 if model.encode_mode == 'latent_table' else lr,
                                               weight_decay=wd)

    if ckpt_file:
        for key in optimizers:
            print(optimizers[key])
            # print(ckpt['optimizer_state_dict'][key]['param_groups'])
            optimizers[key].load_state_dict(ckpt['optimizer_state_dict'][key])
        # optim.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.cuda.amp.GradScaler()

    flog(model)
    flog(
        "{} parameters in model".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    '''for epoch in range(0, 1):
        t2 = dt.now()
        loss_accum = 0
        genloss_accum = 0
        kld_accum = 0
        batch_it = 0
        torch.cuda.empty_cache()
        for minibatch in data_generator:
            ind = minibatch[-1]
            y = minibatch[0].to(device)

            B = len(ind)
            batch_it += B

            rot, trans = posetracker.get_pose(ind, device=device)
            freqs = lattice.coords_2d.unsqueeze(0).expand(
                B, *lattice.coords_2d.shape
            )/ ctf_params[ind, 0].view(B, 1, 1)
            ctf_gen=ctf.compute_ctf(freqs, *torch.split(ctf_params[ind, 1:], 1, 1)).view(B,D,D)
            ctf_gen *= freq_mask.view(1,D,D)
            loss, genloss, kld = train_batch(
                model,
                lattice,
                mask,
                ctf_gen,
                y,
                rot,
                trans,
                optim,
                scaler,
                args,
                pretrain=True,
            )
            loss_accum += loss * B
            genloss_accum += genloss * B
            kld_accum += kld * B
            if batch_it % 10 == 0:
                log(
                    "# [Train Epoch: {}/{}] [{}/{} images] loss={:.6f}, genloss={:.6f}, kld={:.6f}".format(
                        epoch + 1, start_epoch+num_epochs, batch_it, Nimg, loss, genloss, kld
                    )
                )
        flog(
            "# =====> Epoch: {} Average loss = {:.6f}, Averge genloss = {:.6f}, Average kld = {:.6f}; Finished in {}".format(
                epoch + 1,
                loss_accum / Nimg,
                genloss_accum / Nimg,
                kld_accum / Nimg,
                dt.now() - t2,
            )
        )
        if (epoch+1) % checkpoint == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.eval()
                z = eval_z(
                    model,
                    data,
                    batch_size * 128,
                    device,
                    args,
                )
                save_checkpoint(model, optim, epoch, z, outdir)

    td = dt.now() - t1
    flog("Finished in {} ({} per epoch)".format(td, td / (num_epochs)))'''

    '''pose_search_step = dict()
    if args.pose_estimation == "abinit_hps":
        pose_search_step["predicted_rots"] = np.eye(3).reshape(1, 3, 3).repeat(
            Nimg, axis=0)
        pose_search_step["predicted_trans"] = np.zeros((Nimg, 2))
        pose_search_step["first_switch_to_point_estimates"] = first_switch_to_point_estimates
        pose_search_step["is_in_pretrain"] = True
    elif args.pose_estimation == "refine":
        pose_search_step["first_switch_to_point_estimates"] = False
        pose_search_step["is_in_pose_search_step"] = False
        pose_search_step["use_point_estimates"] = True
    else:
        pose_search_step["is_in_pose_search_step"] = False
    '''
    gradient_accumulation = dict()
    gradient_accumulation['accum_steps'] = 1
    gradient_accumulation['global_step'] = 0

    batch_it = 0
    release = -1
    if ckpt_file:
        batch_it += start_epoch * Nimg
        if args.pe_type == 'hash_grid':
            if release == -1 and batch_it > min(30000, Nimg // 2):
                release = 0
                for param in model.decoder.refin_net.parameters():
                    param.requires_grad = True
        else:
            if release == -1 and batch_it > 30000:
                release = 0
                for param in model.decoder.refin_net.parameters():
                    param.requires_grad = True

    for epoch in range(start_epoch, start_epoch + num_epochs):
        t2 = dt.now()
        loss_accum = 0
        genloss_accum = 0
        kld_accum = 0

        torch.cuda.empty_cache()

        if args.pose_estimation == "abinit_hps":
            pose_search_step["is_in_pretrain"] = epoch < 0
            pose_search_step["is_in_pose_search_step"] = (0 <= epoch < epochs_pose_search)
            pose_search_step["use_point_estimates"] = (epoch >= max(0, epochs_pose_search))

        if args.pose_estimation == "abinit_encoder":
            pose_model_train['in_pretrain'] = epoch < 0

        for minibatch in data_generator:
            ind = minibatch['index']
            y = minibatch['y']
            y_HT = minibatch['y_HT']

            '''ind = minibatch[-1]
            y = minibatch[0].to(device)'''

            B = len(ind)
            batch_it += B

            rot, trans = posetracker.get_pose(ind, device=device) if args.pose_estimation == "gt" else (None, None)

            ctf_gen = {}
            freqs = lattice.coords_2d.unsqueeze(0).expand(
                B, *lattice.coords_2d.shape
            ) / ctf_params[ind, 0].view(B, 1, 1)
            ctf_gen_real = ctf.compute_ctf(freqs, *torch.split(ctf_params[ind, 1:], 1, 1)).view(B, D, D)
            ctf_gen_real *= freq_mask.view(1, D, D)
            ctf_gen['real'] = ctf_gen_real

            freqs_ht = lattice.freqs2d.unsqueeze(0).expand(
                B, *lattice.freqs2d.shape
            ) / ctf_params[ind, 0].view(B, 1, 1)
            ctf_gen_ht = ctf.compute_ctf(freqs_ht, *torch.split(ctf_params[ind, 1:], 1, 1)).view(B, D + 1, D + 1)
            ctf_gen['hartley'] = ctf_gen_ht

            if args.pose_estimation == "abinit_encoder" or args.pose_estimation == "abinit_hps":
                rot = None
                # pose_search_step["predicted_rots"][ind], _ = posetracker.get_pose(ind, device='cpu')

            loss, genloss, kld, noise_loss, Negative_loss, similarity_per_sample, gen_loss_ht = train_batch(
                model,
                pose_model,
                lattice,
                mask,
                ctf_gen,
                y,
                y_HT,
                rot,
                trans,
                ind,
                epoch,
                pose_search_step,
                pose_model_train,
                ps_params,
                optimizers,
                scaler,
                args,
                gradient_accumulation,
                outdir=outdir,
            )
            loss_accum += loss * B
            genloss_accum += genloss * B
            kld_accum += kld * B
            if B > batch_it % 10 >= 0:
                log(
                    "# [Train Epoch: {}/{}] [{}/{} images] loss={:.6f}, genloss={:.6f}, kld={:.6f}, l1_noise_loss={:.6f}, l1_Negative_loss={:.6f}, similarity_per_sample={:.6f}, gen_hartley_loss={:.6f}".format(
                        epoch + 1, start_epoch + num_epochs, batch_it, Nimg, loss, genloss, kld, noise_loss,
                        Negative_loss, similarity_per_sample, gen_loss_ht
                    )
                )

            '''if batch_it >= 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    model.eval()
                    z = eval_z(
                        model,
                        data,
                        batch_size * 128,
                        device,
                        args,
                    )
                    save_checkpoint(model, optimizers, epoch, z, outdir)'''

            save_size = 5000
            if B > batch_it % save_size >= 0 >= args.zdim and batch_it >= save_size:
                out_weights = outdir + "weights.{}.pkl".format(epoch)
                out_vol = {}
                out_vol['recon'] = outdir + "epoch{}.batch_it{:03d}.{}.mrc".format(epoch, batch_it, D)
                out_vol['recon_mlp'] = outdir + "epoch{}.batch_it_MLP{:03d}.{}.mrc".format(epoch, batch_it, D)
                with torch.no_grad():
                    model.eval()
                    eval_vol(
                        model,
                        lattice,
                        mask[:-1, :-1, :-1].flatten() if mask is not None else None,
                        data.norm,
                        optimizers,
                        epoch,
                        pose_search_step,
                        Apix=data.apix,
                        out_weights=out_weights,
                        outdir=outdir,
                        out_vol=out_vol,
                    )
            if args.pe_type == 'hash_grid':
                if release == -1 and batch_it > min(30000, Nimg // 2):
                    release = 0
                    for param in model.decoder.refin_net.parameters():
                        param.requires_grad = True
            else:
                if release == -1 and batch_it > 30000:
                    release = 0
                    for param in model.decoder.refin_net.parameters():
                        param.requires_grad = True

        flog(
            "# =====> Epoch: {} Average loss = {:.6f}, Averge genloss = {:.6f}, Average kld = {:.6f}; Finished in {}".format(
                epoch + 1,
                loss_accum / Nimg,
                genloss_accum / Nimg,
                kld_accum / Nimg,
                dt.now() - t2,
            )
        )
        if (epoch + 1) % checkpoint == 0:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                if args.zdim > 0:
                    z = eval_z(
                        model,
                        data,
                        batch_size * 128,
                        device,
                        args,
                    )
                else:
                    z = None
                    out_vol = {
                        "recon": outdir + f"vol_epoch{epoch:03d}.recon.{D}.mrc",
                        "recon_mlp": outdir + f"vol_epoch{epoch:03d}.recon_mlp.{D}.mrc",
                    }
                    eval_vol(
                        model,
                        lattice,
                        mask[:-1, :-1, :-1].flatten() if mask is not None else None,
                        data.norm,
                        optimizers,
                        epoch,
                        pose_search_step,
                        Apix=data.apix,
                        out_weights=None,
                        outdir=outdir,
                        out_vol=out_vol,
                    )
                save_checkpoint(model, optimizers, epoch, z, pose_search_step, outdir)

    td = dt.now() - t1
    flog("Finished in {} ({} per epoch)".format(td, td / num_epochs))


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.ind is not None:
        ind = utils.load_pkl(args.ind)
    else:
        ind = None

    particles = args.particles
    flog(f"Loading dataset from {particles}")
    if args.datadir is not None:
        datadir = args.datadir
    elif particles.endswith('.cs'):
        datadir = os.path.dirname(os.path.dirname(os.path.abspath(particles)))
    else:
        datadir = None

    if args.split is not None:
        assert ind is None, "ERROR: Split dataset with filtered indices"
        if particles.endswith(".cs"):
            raw = np.load(args.particles)
            assert 1 in raw['alignments3D/split'], "ERROR: Particles file without random subset flag"
            ind = np.argwhere(raw['alignments3D/split'] == (int(args.split) - 1))[:, 0].tolist()
        elif particles.endswith(".star"):
            s = cryoio.Starfile.load(args.particles)
            assert 'rlnRandomSubset' in s.df, "ERROR: Particles file without random subset flag"
            ind = np.argwhere(s.df['rlnRandomSubset'] == int(args.split))[:, 0].tolist()
        else:
            raise RuntimeError("Particle file doesn't contain split indices")
        flog(f"Split dataset randomly for GSFSC")

    if args.lazy:
        data = dataset.LazyMRCData(
            particles,
            datadir=datadir,
            norm=args.norm,
            window=args.window,
            invert_data=args.invert_data,
            window_r=args.window_r,
            ind=ind,
            flog=flog,
            domain=args.domain,
            zdim=args.zdim,
        )
    else:
        data = dataset.MRCData(
            particles,
            datadir=datadir,
            norm=args.norm,
            window=args.window,
            invert_data=args.invert_data,
            window_r=args.window_r,
            ind=ind,
            flog=flog,
            domain=args.domain,
        )
    Nimg = data.N
    D = data.D

    if args.ctf is not None:
        ctf_params = ctf.load_ctf_for_training(D, args.ctf)
        flog(f"Loading ctf from {args.ctf}")
    else:
        if particles.endswith(".cs"):  # Parse ctf params from a cryoSPARC .cs metafile
            ctf_params = cryoio.parse_ctf_csparc(args.particles, D=D, Apix=data.apix)
        elif particles.endswith(".star"):  # Parse ctf params from a Relion .star metafile
            ctf_params = cryoio.parse_ctf_relion(args.particles, D=D, Apix=data.apix)
        else:
            raise RuntimeError("Must provide CTF parameters")
        # Slice out the first column (D)
        ctf_params = ctf_params[:, 1:]

    if ind is not None:
        ctf_params = ctf_params[ind]
    ctf_params = torch.tensor(ctf_params, dtype=torch.float32, device=device)  # star/cs parse yields float64, which poisons autograd dtype

    # if args.pose_estimation == "abinit":
    posetracker = None
    if args.poses is not None:
        pose = args.poses
        posetracker = PoseTracker.load(
            pose,
            Nimg,
            D,
            ind=ind,
        )
    elif args.pose_estimation != "abinit_hps" and args.pose_estimation != "abinit_encoder":
        if particles.endswith(".cs"):  # Parse image poses from a cryoSPARC .cs metafile
            rot, trans = cryoio.parse_pose_csparc(args.particles)
        elif particles.endswith(".star"):  # Parse image poses from a Relion .star metafile
            rot, trans = cryoio.parse_pose_relion(args.particles, Apix=data.apix)
        else:
            raise RuntimeError("Must provide estimated poses")
        if ind is not None:
            rot, trans = rot[ind], trans[ind]
        posetracker = PoseTracker(rot, trans, D)

    if args.mask is not None:
        mask, _ = mrc.parse_mrc(args.mask, is_vol=True)
        mask = torch.tensor(mask.copy(), device=device)
        mask = fft.symmetrize_3DGPU(mask).squeeze()
    else:
        mask = None

    if args.load is not None:
        ckpt = args.load
    else:
        ckpt = None

    train(data, ctf_params, posetracker, mask, device=device, ckpt_file=ckpt, outdir=outdir, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    outdir = args.outdir + '/'
    if args.split == '1':
        outdir += 'halfA/'
    elif args.split == '2':
        outdir += 'halfB/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    LOG = f"{outdir}/run.log"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, LOG)


    flog("Use cuda {}".format(use_cuda))
    if not use_cuda:
        log("WARNING: No GPUs detected")
    flog(" ".join(sys.argv))

    main(args)
