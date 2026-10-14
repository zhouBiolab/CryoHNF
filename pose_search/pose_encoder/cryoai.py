"""cryoAI class"""

import torch
import torch.fft
from torch import nn
from .ml_modules import FCBlock, CNNEncoderVGG16
from .filter_utils import GaussianPyramid
from .lie_tools import rotation_6d_to_matrix

import os
import time


class CryoAI(nn.Module):
    def __init__(self, device, num_rotations=7):
        """
        Initialization of a cryoAI model.

        Parameters
        ----------
        config: namespace
        """
        super(CryoAI, self).__init__()
        # self.config = config
        self.sidelen_input = 128
        self.device = device
        self.num_rotations = num_rotations
        '''# Schedulers
        self.schedulers = {}
        if config.use_frequency_marcher:
            self.frequency_marcher = FrequencyMarcher(self, config.side_len, config.test_convergence_every)
            self.schedulers['frequency_marcher'] = self.frequency_marcher
        else:
            self.frequency_marcher = None

        # Volume representation
        if config.volume_representation == 'imp-fourier':
            params_implicit = {"type": config.implicit_representation_type, "force_symmetry": config.force_symmetry}
            self.pred_map = ImplicitFourierVolume(
                img_sz=config.map_shape[0],
                params_implicit=params_implicit,
                frequency_marcher=self.frequency_marcher
            )
        else:
            raise NotImplementedError'''

        # Pre-filtering
        self.gaussian_filters = GaussianPyramid(
            kernel_size=11,
            kernel_variance=0.01,
            num_octaves=4,
            octave_scaling=10
        )
        num_additional_channels = 0


        # Encoder
        self.cnn_encoder = CNNEncoderVGG16(
            in_channels=1+num_additional_channels,
            batch_norm=False,
            pretrained=False,
            flip_images=False,
            high_res=(self.sidelen_input > 128)
        )
        map_height = self.sidelen_input # 128
        map_width = self.sidelen_input
        cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(map_height, map_width)
        print("cnn_encoder_out_shape", cnn_encoder_out_shape)
        latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))
        print("latent_code_size", latent_code_size)


        # Orientation regressors

        self.orientation_dims = 6
        self.last_nonlinearity = None
        self.latent_to_rot3d_fn = rotation_6d_to_matrix
        '''
        self.orientation_dims = 4
        self.last_nonlinearity = None
        self.latent_to_rot3d_fn = quat_to_rotmat
        '''
        # We split the regressor in 2 to have access to the latent code
        self.orientation_encoder = FCBlock(
            in_features=latent_code_size,
            out_features=256,
            features=[512],
            nonlinearity='relu',
            last_nonlinearity='relu',
            batch_norm=False
        )
        self.orientation_regressor = FCBlock(
            in_features=256,
            out_features=self.orientation_dims,
            features=[],
            nonlinearity='relu',
            last_nonlinearity=self.last_nonlinearity, # None
            batch_norm=False
        )
        '''self.orientation_regressor = nn.ModuleList()
        for _ in range(self.num_rotations):
            # We split the regressor in 2 to have access to the latent code
            self.orientation_regressor.append(FCBlock(
                in_features=latent_code_size,
                out_features=self.orientation_dims,
                features=[512, 256],
                nonlinearity='relu',
                last_nonlinearity=None,
                batch_norm=False)
            )'''

        # Shift regressor
        self.shift_encoder = FCBlock(
            in_features=latent_code_size,
            out_features=2,
            features=[512, 256],
            nonlinearity='relu',
            last_nonlinearity=None
        )

        # Shift model
        '''self.shift = Shift(
            num_particles=config.num_particles,
            size=config.ctf_size,
            resolution=config.resolution,
            requires_grad=False,
            flip_images=config.flip_images
        )'''


    def warm_start(self, path_to_warm_start):
        """
        Warm start encoder + decoder.

        Parameters
        ----------
        path_to_warm_start: str
        """
        print("Full warm start.")
        self.load_state_dict(torch.load(path_to_warm_start))

    def warm_start_volume(self, path_to_warm_start_volume):
        """
        Warm start decoder.

        Parameters
        ----------
        path_to_warm_start_volume: str
        """
        print("Warm start volume.")
        self.pred_map.load_state_dict(torch.load(path_to_warm_start_volume))

    def save(self, checkpoints_dir, epoch):
        """
        Saves the model.

        Parameters
        ----------
        checkpoints_dir: abspath
        epoch: int
        """
        torch.save(self.state_dict(),
                   os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
        torch.save(self.pred_map.state_dict(),
                   os.path.join(checkpoints_dir, 'pred_map_epoch_%04d.pth' % epoch))

    def forward(self, y):
        """
        Forward pass.

        Parameters
        ----------
        in_dict: dictionary

        Returns
        -------
        output_dict: dictionary
        """
        start_time = time.time()
        proj = y.unsqueeze(1).to(self.device)

        # Schedulers
        '''for name, scheduler in self.schedulers.items():
            scheduler.update()'''

        latent_code = None
        # print("proj", proj.shape)
        # proj = self.gaussian_filters(proj)
        # print("proj", proj.shape)
        # print("self.cnn_encoder(proj)", self.cnn_encoder(proj).shape, self.cnn_encoder(proj)[0,...] == self.cnn_encoder(proj)[1,...])

        latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
        # print("latent_code", latent_code[0,...] == latent_code[1,...])
        latent_code_pose = self.orientation_encoder(latent_code)
        # print("latent_code_pose", latent_code_pose[0,...] == latent_code_pose[1,...])
        latent_code_prerot = self.orientation_regressor(latent_code_pose)
        # Interpret the latent code as a rotation
        print("latent_code_prerot", latent_code_prerot.shape, latent_code_prerot)
        pred_rotmat = self.latent_to_rot3d_fn(latent_code_prerot)
        print("pred_rotmat", pred_rotmat.shape)




        encoder_time = time.time()

        # Shift
        shift_params = self.shift_encoder(latent_code)
        # print("shift_params[..., 0]", shift_params)
        tans = shift_params
        '''



        ctf_shift_prep_time = time.time()

        # Query the volume (slicing / projection)
        pred_fproj_prectf = self.pred_map(pred_rotmat)

        decoder_time = time.time()

        # Apply the remaining step of the forward model: CTF + Shift
        pred_fproj = self.ctf(
            pred_fproj_prectf,
            in_dict['idx'],
            pred_ctf_params,
            mode=self.config.use_ctf,
            frequency_marcher=self.frequency_marcher
        )
        pred_fproj = self.shift(
            pred_fproj,
            in_dict['idx'],
            pred_shift_params,
            mode=self.config.use_shift,
            frequency_marcher=self.frequency_marcher
        )

        end_time = time.time()

        times = {'start': start_time,
                 'encoder': encoder_time,
                 'ctf_shift_prep': ctf_shift_prep_time,
                 'decoder': decoder_time,
                 'end': end_time}
        
        # Fill the output dictionary
        output_dict = {'rotmat': pred_rotmat,
                       'latent_code': latent_code,
                       'latent_code_prerot': latent_code_prerot,
                       'fproj': pred_fproj,
                       'fproj_prectf': pred_fproj_prectf,
                       'pred_ctf_params': pred_ctf_params,
                       'pred_shift_params': pred_shift_params,
                       'times': times}

        # Make sure we are in sync by bringing back the proj to the primal domain
        if self.config.data_loss_domain == 'primal' or self.config.compute_proj:
            pred_proj = fourier_to_primal_2D(pred_fproj)
            output_dict['proj'] = pred_proj.real

        fproj_gt = in_dict['fproj']  # B, 1, S, S
        if self.frequency_marcher is not None:
            img_sz = self.config.map_shape[0]
            left = max(0, img_sz // 2 - self.frequency_marcher.f)
            right = min(img_sz, img_sz // 2 + self.frequency_marcher.f + 1)
            fproj_gt = fproj_gt[:, :, left:right, left:right]  # B, 1, 2*f+1, 2*f+1
        output_dict['fproj_gt'] = fproj_gt
        if self.config.data_loss_domain == 'primal' or self.config.compute_proj:
            proj_gt = fourier_to_primal_2D(fproj_gt)
            output_dict['proj_gt'] = proj_gt.real'''

        return pred_rotmat, tans
