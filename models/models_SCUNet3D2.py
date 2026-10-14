"""
Adapted from code for
"CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks" by Zhong et al. (2021)
https://github.com/ml-struct-bio/cryodrgn/
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    import tinycudann as tcnn
except ImportError:
    tcnn = None  # only required for --pe-type hash_grid
import random

from itertools import product
from tqdm.auto import tqdm
from models.splsounet import Unet
import time

from core import lattice
from core import utils
from data import mrc
from core import fft
from pose_search.pose_find import pose_search
from models.UNet3D import UNet3D
from torch.utils.checkpoint import checkpoint
from models.scunet2 import SCUNet
from pose_search.pose_find import lie_tools

log = utils.log
BOX_OFFSETS = [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]


def unparallelize(model):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(
            self,
            lattice,  # Lattice object
            qlayers,
            qdim,
            players,
            pdim,
            in_dim,
            n_particles_dataset=None,
            zdim=0,
            encode_mode="latent_table",
            enc_mask=None,
            enc_type="hash_grid",  # "geom_ft",
            enc_dim=None,
            activation=nn.ReLU,
            domain='real',
            feat_sigma=None,
            ps_params=None,
            pose_ues="gt",
            refin_type="scunet",
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.coords = nn.Parameter(self.lattice.coords, requires_grad=False)
        self.coords_ht = nn.Parameter(self.lattice.coords_ht, requires_grad=False)
        self.freqs2d = nn.Parameter(self.lattice.freqs2d, requires_grad=False)
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if self.zdim > 0:
            enc_type = "geom_ft"
            if encode_mode == "conv":
                self.encoder = ConvEncoder(in_dim, qdim, zdim * 2)
            elif encode_mode == "resnet":
                self.encoder = ImageEncoder("resnet18", self.zdim, size=lattice.D, hartley=True)
            elif encode_mode == "resid":
                self.encoder = ResidLinearMLP(
                    in_dim,
                    qlayers,
                    qdim,
                    zdim * 2,
                    activation,  # nlayers  # hidden_dim  # out_dim
                )
            elif encode_mode == "mlp":
                self.encoder = MLP(
                    in_dim, qlayers, qdim, zdim * 2, activation  # hidden_dim  # out_dim
                )  # in_dim -> hidden_dim
            elif encode_mode == "latent_table":
                self.encoder = ConfTable(n_particles_dataset, zdim, True, 0.1)

            else:
                raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        self.encode_mode = encode_mode
        self.decoder = get_decoder(
            3 + zdim,
            lattice.D,
            players,
            pdim,
            enc_type,
            enc_dim,
            activation,
            domain,
            feat_sigma,
            refin_type,
        )

        if ps_params is not None:
            self.base_shifts = nn.Parameter(
                pose_search.get_base_shifts(ps_params['real']), requires_grad=False)
            self.base_rot = nn.Parameter(
                pose_search.get_base_rot(ps_params['real']), requires_grad=False)
            self.so3_base_quat = nn.Parameter(
                pose_search.get_so3_base_quat(ps_params['real']), requires_grad=False)
            self.base_inplane = nn.Parameter(
                pose_search.get_base_inplane(ps_params['real']), requires_grad=False)

        if pose_ues != 'gt':
            self.pose_table = PoseTable(n_particles_dataset, no_trans=False, resolution=lattice.D, use_gt_trans=False)

    @classmethod
    def load(self, config, weights=None, device=None):
        """Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        """
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg["lattice_args"]
        D = c["D"]
        lat = lattice.Lattice(c["D"], extent=c["extent"], device=device)
        c = cfg["model_args"]
        if c["enc_mask"] > 0:
            mask = lat.get_sphere_mask(c["enc_mask"])
            enc_mask = mask.view(D, D, D).sum(-1) > 0
            in_dim = int(enc_mask.sum())
        else:
            assert c["enc_mask"] == -1
            enc_mask = None
            in_dim = lat.D ** 2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
        model = HetOnlyVAE(
            lat,
            c["qlayers"],
            c["qdim"],
            c["players"],
            c["pdim"],
            in_dim,
            c["zdim"],
            encode_mode=c["encode_mode"],
            enc_mask=enc_mask,
            enc_type=c["pe_type"],
            enc_dim=c["pe_dim"],
            domain=c["domain"],
            activation=activation,
            feat_sigma=c["feat_sigma"],
        )
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt["model_state_dict"])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, img, ind=None):
        if self.encode_mode == "latent_table":
            conf, logvar = self.encoder(ind)
            return conf, logvar
        if self.encode_mode == "resnet":
            z = self.encoder(img.unsqueeze(1))
            # print("z", z.shape)
            return z[:, : self.zdim], z[:, self.zdim:]
        if self.enc_mask is not None:
            img = img[:, self.enc_mask]
        elif self.encode_mode == 'conv':
            img = img[:, None]
        else:
            img = img.view(img.shape[0], -1)
        z = self.encoder(img)
        if isinstance(z, dict):
            z = z['recon_real']
        return z[:, : self.zdim], z[:, self.zdim:]

    def cat_z(self, coords, z):
        """
        coords: Bx...x3
        z: Bxzdim
        """
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, mask=None, z=None, pretrain=False, domain='real', coords_ht=None, dict=None):
        """
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        """
        if z is not None:
            coords = self.cat_z(coords, z.repeat_interleave(coords.shape[0] // z.shape[0], dim=0))
            if coords_ht is not None:
                coords_ht = self.cat_z(coords_ht, z.repeat_interleave(coords.shape[0] // z.shape[0], dim=0))
        # print("cat_z coords", coords.shape) [2, 2097152, 11]
        return self.decoder(coords, mask=mask, pretrain=pretrain, domain=domain, coords_ht=coords_ht, dict=dict)

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, img=None, ind=None, coords=None, mask=None, encode_only=False, z=None,
                pretrain=False, img_ht=None, domain='real', coords_ht=None, dict=None):
        if self.zdim > 0:
            if z is None:
                if img_ht is not None:
                    # print("img_ht", img_ht.shape)
                    z_mu, z_logvar = self.encode(img_ht, ind)
                    # print("z_mu z_logvar have nan", torch.isnan(z_mu).any(),torch.isnan(z_logvar).any())
                    z = self.reparameterize(z_mu, z_logvar)
                    # print("z have nan", torch.isnan(z).any())
                    # print("z_mu z_logvar z",z_mu,z_logvar,z)
                elif img_ht is None:
                    img_ht = fft.symmetrize_ht(fft.ht2_center(img))
                    z_mu, z_logvar = self.encode(img_ht, ind)
                    z = self.reparameterize(z_mu, z_logvar)
            else:
                z_mu = z
                z_logvar = None
            if encode_only:
                return z_mu, z_logvar
            else:
                if self.training:
                    recon, similarity_per_sample = self.decode(coords, mask=mask, z=z, pretrain=pretrain,
                                                               domain=domain, coords_ht=coords_ht, dict=dict)
                    return recon, z_mu, z_logvar, similarity_per_sample
                else:
                    recon, similarity_per_sample = self.decode(coords, mask=mask, z=z, pretrain=pretrain,
                                                               domain=domain, coords_ht=coords_ht, dict=dict)
                    return recon, z_mu, z_logvar, similarity_per_sample
        elif self.zdim <= 0:
            recon, similarity_per_sample = self.decode(coords, mask=mask, z=None, pretrain=pretrain,
                                                       domain=domain, coords_ht=coords_ht, dict=dict)
            return recon, 0, 1, similarity_per_sample


'''def load_decoder(config, weights=None, device=None):
    """
    Instantiate a decoder model from a config.pkl

    Inputs:
        config (str, dict): Path to config.pkl or loaded config.pkl
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    """
    cfg = utils.load_pkl(config) if type(config) is str else config
    c = cfg["model_args"]
    D = cfg["lattice_args"]["D"]
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
    model = get_decoder(
        3,
        D,
        c["layers"],
        c["dim"],
        c["domain"],
        c["pe_type"],
        c["pe_dim"],
        activation,
        c["feat_sigma"],
    )
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model.to(device)
    return model'''


def get_decoder(
        in_dim,
        D,
        layers,
        dim,
        enc_type,
        enc_dim=None,
        activation=nn.ReLU,
        domain='real',
        feat_sigma=None,
        refin_type='scunet',
):
    return PositionalDecoder(
        in_dim,
        D,
        layers,
        dim,
        activation,
        domain=domain,
        enc_type=enc_type,
        refin_type=refin_type,
        enc_dim=enc_dim,
        feat_sigma=feat_sigma,
    )


class PositionalDecoder(nn.Module):
    def __init__(
            self,
            in_dim,
            D,
            nlayers,
            hidden_dim,
            activation,
            dec_type='mlp',
            # enc_type="geom_ft",
            domain='real',
            enc_type="hash_grid",
            refin_type="scunet",
            enc_dim=None,
            feat_sigma=.5,
            bias=True,
            base_radius=None,
            pos_embed=0,
            scale_correction=0,
            box_size=48,
    ):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = self.enc_dim * 6 + self.zdim
        self.dec_type = dec_type
        self.base_radius = base_radius
        self.scale_correction = False
        self.pos_est = False
        self.rand_freqs = None
        self.outer_cut = 12
        self.box_size = box_size - self.outer_cut
        self.mlp_out_dim = 1
        self.print_memory = True
        self.domain = domain

        if enc_type == "hash_grid":
            if tcnn is None:
                raise RuntimeError(
                    "tinycudann is required for --pe-type hash_grid; "
                    "install it or run with --pe-type geom_lowf")
            self.hash_grid_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.4472692012786865,
            }
            self.hash_grid_encoding = tcnn.Encoding(3, self.hash_grid_config, dtype=torch.float32,
                                                    seed=random.randint(0, 1524367))
            self.in_dim = self.hash_grid_config["n_levels"] * self.hash_grid_config[
                "n_features_per_level"] + self.zdim  # + self.enc_dim * 6
            '''hidden_dim = 128
            nlayers = 2'''
        if pos_embed > 0:
            self.rot_embed = nn.Embedding(pos_embed, 6)
            self.trans_embed = nn.Embedding(pos_embed, 2)
            # nn.init.uniform_(self.trans_embed.weight, a=-0.0001, b=0.0001)
            self.trans_embed.weight.data.zero_()
            self.pos_est = True
        if scale_correction > 0:
            self.scale = nn.Embedding(scale_correction, 1)
            nn.init.zeros_(self.scale.weight)
            self.background = nn.Embedding(scale_correction, 1)
            nn.init.zeros_(self.background.weight)
            self.scale_correction = True

        '''if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma

            rand_freqs = (
                    torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            )
            # rand_freqs = torch.Tensor(truncnorm.rvs(-np.pi*2/feat_sigma, np.pi*2/feat_sigma, scale = feat_sigma * self.D2, size = (3 * self.enc_dim, 3)))
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
            self.feat_sigma = feat_sigma
        else:
            self.rand_feats = None'''

        rand_freqs = (
                torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * 0.5
        )
        self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False)
        self.feat_sigma = 0.5

        if refin_type == "scunet":
            self.refin_net = SCUNet(in_nc=self.mlp_out_dim, input_resolution=box_size)

        if domain == 'real':
            if dec_type == 'mlp':
                self.decoder = [
                    MLP(self.in_dim, nlayers, hidden_dim, self.mlp_out_dim, activation, bias=bias, box_size=box_size)]
                self.decoder.append(self.refin_net)
                self.decoder = nn.ModuleList(self.decoder)
            elif dec_type == 'resid':
                # self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation, bias=bias)
                ###
                self.decoder = [
                    ResidLinearMLP(self.in_dim, nlayers, hidden_dim, self.mlp_out_dim, activation, bias=bias)]
                self.decoder.append(self.refin_net)
                self.decoder = nn.ModuleList(self.decoder)
                ###
        if domain == 'all':
            if dec_type == 'mlp':
                self.decoder = [
                    MLP_combine(self.in_dim, self.enc_dim * 6 + self.zdim, nlayers, hidden_dim, self.mlp_out_dim,
                                activation, bias=bias)]
                self.decoder.append(self.refin_net)
                self.decoder = nn.ModuleList(self.decoder)

            elif dec_type == 'resid':
                # self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation, bias=bias)
                ###
                self.decoder = [
                    ResidLinearMLP(self.in_dim, nlayers, hidden_dim, self.mlp_out_dim, activation, bias=bias)]
                self.decoder.append(self.refin_net)
                self.decoder = nn.ModuleList(self.decoder)
                ###

    def pick_centered_normal(self, lst, sigma=None):
        """
        Randomly select an element from the list, with the selection probability
        following a normal distribution centered around the middle value.

        Parameters:
        - lst (list): The list of elements to choose from.
        - sigma (float, optional): The standard deviation of the normal distribution.
                                    Defaults to len(lst)/2.

        Returns:
        - element: The selected element from the list.
        """
        n = len(lst)
        arr = np.array(lst, dtype=float)
        mu = (n - 1) / 2
        if sigma is None:
            sigma = n / 4.0

        weights = np.exp(-0.5 * ((np.arange(n) - mu) / sigma) ** 2)
        weights /= weights.sum()

        selected = np.random.choice(arr, p=weights)
        return int(selected)

    def positional_encoding_geom(self, coords, enc_dim=None, enc_type=None):
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        if enc_dim is None:
            enc_dim = self.enc_dim
        if enc_type is None:
            enc_type = self.enc_type
        if enc_type == "hash_grid":
            encoded_pos = self.hash_grid_encoding(coords.reshape(-1, 3))
            # print("*coords.shape[0:-2]", encoded_pos.shape)
            return encoded_pos
        freqs = torch.arange(enc_dim, dtype=torch.float, device=coords.device)
        if enc_type == "geom_ft" or enc_type == "hash_grid":
            freqs = (
                    self.DD * np.pi * (2.0 / self.DD) ** (freqs / (enc_dim - 1))
            )  # option 1: 2/D to 1
        elif enc_type == "geom_full":
            freqs = (
                    self.DD
                    * np.pi
                    * (1.0 / self.DD / np.pi) ** (freqs / (enc_dim - 1))
            )  # option 2: 2/D to 2pi
        elif enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (
                    freqs / (enc_dim - 1)
            )  # option 3: 2/D*2pi to 2pi
        elif enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (
                    freqs / (self.enc_dim - 1)
            )  # option 4: 2/D*2pi to 1
        elif enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        elif (enc_type == "gaussian") or (enc_type == "log_uniform"):
            return self.random_fourier_encoding(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(enc_type))

        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x 1 x 32
        # print("freqs", freqs.shape)
        # print("coords", coords)
        coords = coords.unsqueeze(-1)  # B x D^3 X 3 x 1
        # print("coords", coords.shape)
        x = torch.cat([torch.sin(coords * freqs), torch.cos(coords * freqs)], -1)  # B x D^3 x 3 x (32 + 32)
        # print("x", x.shape)
        if enc_dim is None:
            x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x D^3 x (in_dim-zdim)=(3 x D2)
        else:
            x = x.view(*coords.shape[:-2], -1)
        ###
        '''x_3d = x.reshape(coords.shape[0], self.D, self.D, self.D, -1)
        map_shape = np.shape(x_3d)
        padded_map = np.full((coords.shape[0], map_shape[1] + 2 * self.box_size, map_shape[2] + 2 * self.box_size, map_shape[3] + 2 * self.box_size, map_shape[4]),0).astype(np.float32)
        padded_map[:, self.box_size: self.box_size + map_shape[1], self.box_size: self.box_size + map_shape[2],self.box_size: self.box_size + map_shape[3], :] = x_3d
        padded_map_3D = np.transpose(padded_map, (0, 4, 3, 1, 2))
        ### padded_map_3D --> B x (in_dim-zdim)=(3 x D) x (D+ 2 x box_size) x (D+ 2 x box_size) x (D+ 2 x box_size)'''
        # x = torch.cat((x, encoded_pos), dim=-1)
        return x

    def positional_encoding_linear(self, coords):
        """Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2"""
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, x):
        """
        x: [batch_size(, n_tilts), n_pts, 3]

        output: [batch_size(, n_tilts), n_pts, 3 * 2 * pe_dim]
        """
        freqs = self.rand_freqs.reshape(1, 1, -1, 3) * ((self.D + 1) // 2)
        # print("x[..., None, :]", x[..., None, :].shape)
        # print("freqs", freqs.shape)
        kx_ky_kz = x[..., None, :] * freqs
        k = kx_ky_kz.sum(-1)
        s = torch.sin(k)
        c = torch.cos(k)
        x_encoded = torch.cat([s, c], -1)
        return x_encoded

    '''def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        k = (coords[..., None, :] * freqs).sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x'''

    def normalize_single_box(self, box_tensor, eps=1e-8):
        mean = box_tensor.mean(dim=(-3, -2, -1), keepdim=True)
        std = box_tensor.std(dim=(-3, -2, -1), keepdim=True)

        return (box_tensor - mean) / (std + eps)

    def reinitialize(self):
        if self.rand_freqs is not None:
            if self.symmetry is not None:
                rand_freqs = []
                rand_freqs_tmp = torch.randn((3 * self.enc_dim // len(self.symmetry), 3),
                                             dtype=torch.float) * self.feat_sigma
                for i in range(len(self.symmetry)):
                    rand_freqs.append(rand_freqs_tmp @ self.symmetry[i].permute([1, 0]))
                rand_freqs = torch.cat(rand_freqs)
            else:
                rand_freqs = (
                        torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * self.feat_sigma
                )
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs.data = rand_freqs.to(self.rand_freqs.device)
        for layer in self.decoder.main:
            if isinstance(layer, MyLinear):
                layer.reset_parameters()

    def forward(self, coords, l_current=None, mask=None, ctf=None, pretrain=False, batch_it=None,
                in_posesearch=False, domain='real', coords_ht=None, dict=None):
        """Input should be coordinates from [-.5,.5]"""
        # assert (coords[..., 0:3].abs() - 0.5 < 1e-4).all()
        if len(coords.shape) == 3:
            B, ncoords, c = coords.shape
            t = 1
        else:
            B, t, ncoords, c = coords.shape
            coords = coords.reshape(B * t, ncoords, c)
            mask = mask.repeat_interleave(t, dim=0) if mask is not None else None
        mask = mask.repeat_interleave(coords.shape[0] // mask.shape[0], dim=0) if mask is not None else None
        Dz = self.D
        D = self.D if l_current is None else l_current
        recon = torch.zeros([B, D, D, Dz], dtype=torch.float, device=coords.device, requires_grad=True)
        outer_cut = self.outer_cut
        recon_mlp = None
        stride = 12
        similarity_per_sample = 0
        unet_box = 0
        ran_i = None
        ran_j = None
        ###
        # coords = coords[mask>0]
        # pretrain = True
        ### mask [B D D D]
        # print("mask", mask.shape)
        mask_1d = rearrange(mask, 'b a e c -> b (a e c)') if mask is not None else None
        coords_mask = coords[..., :3][mask_1d > 0] if mask is not None else None
        for layer in self.decoder:
            if (isinstance(layer, ResidLinearMLP) or isinstance(layer, MLP)
                    or isinstance(layer, ConvEncoder) or isinstance(layer, MLP_combine)):
                # print("MLP MLP MLP")
                if domain != 'real':
                    coords_mask_positional_encoding_frequency = self.positional_encoding_geom(coords_ht[..., :3],
                                                                                              enc_type='gaussian')
                if domain != 'hartley':
                    # print("coords_mask", coords_mask.shape)  # ([8783312, 3])
                    coords_mask_positional_encoding = self.positional_encoding_geom(coords_mask)
                # print("coords_mask_positional_encoding.requires_grad", coords_mask_positional_encoding.requires_grad)
                # print("coords_mask_positional_encoding",coords_mask_positional_encoding.shape)  # ([8783312, 192])
                # print("coords", coords.shape)  # ([8, 2097152, 3])
                # print("coords[..., 3:][mask_1d > 0]", coords[..., 3:][mask_1d > 0].shape)([8783312, 0])
                # print("coords_mask_positional_encoding have nan", torch.isnan(coords_mask_positional_encoding).any())
                x_dict = None
                if domain == 'real':
                    x_dict = layer(torch.cat([coords_mask_positional_encoding,
                                              coords[..., 3:][mask_1d > 0]], -1))
                elif domain == 'hartley':
                    x_dict = layer(None,
                                   torch.cat([coords_mask_positional_encoding_frequency, coords_ht[..., 3:]],
                                             -1))
                elif domain == 'all':
                    if dict is not None:
                        if dict.get('transtoreal', False):
                            x_dict = layer(
                                torch.cat([coords_mask_positional_encoding, coords[..., 3:][mask_1d > 0]], -1),
                                None)
                        else:
                            x_dict = layer(
                                torch.cat([coords_mask_positional_encoding, coords[..., 3:][mask_1d > 0]], -1),
                                torch.cat([coords_mask_positional_encoding_frequency, coords_ht[..., 3:]],
                                          -1))
                    else:
                        x_dict = layer(
                            torch.cat([coords_mask_positional_encoding, coords[..., 3:][mask_1d > 0]], -1),
                            torch.cat([coords_mask_positional_encoding_frequency, coords_ht[..., 3:]],
                                      -1))
                # print("x_MLP", x.shape, x.dtype)
                mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, -1, self.mlp_out_dim) if mask is not None else None
                if in_posesearch:
                    '''total_memory = torch.cuda.get_device_properties(0).total_memory  # total GPU memory
                    reserved_memory = torch.cuda.memory_reserved(0)  # reserved GPU memory
                    allocated_memory = torch.cuda.memory_allocated(0)  # currently allocated GPU memory
                    free_memory = reserved_memory - allocated_memory  # free GPU memory
                    print(f"POSESEARCH Total memory UNet3D: {total_memory / (1024 ** 2):.2f} MB",
                          f"POSESEARCH Reserved memory UNet3D: {reserved_memory / (1024 ** 2):.2f} MB",
                          f"POSESEARCH Allocated memory UNet3D: {allocated_memory / (1024 ** 2):.2f} MB",
                          f"POSESEARCH Free memory UNet3D: {free_memory / (1024 ** 2):.2f} MB")'''
                    if dict and dict.get('transtoreal', False):
                        x_dict['recon_real'] = torch.zeros([B * t, D, D, Dz, self.mlp_out_dim],
                                                           device=coords.device, dtype=torch.float).masked_scatter(
                            mask_exp > 0, x_dict['recon_real'])
                        x_dict['recon_real'] = x_dict['recon_real'].permute(0, 4, 1, 2, 3)
                    return x_dict  # ['recon_hartley']

                x = torch.zeros([B * t, D, D, Dz, self.mlp_out_dim], device=coords.device, dtype=torch.float)
                # mask_exp = mask.unsqueeze(-1).expand(-1, -1, -1, -1, self.mlp_out_dim) if mask is not None else None
                if mask_exp is not None:
                    x = x.masked_scatter(mask_exp > 0, x_dict['recon_real'])
                x = x.permute(0, 4, 1, 2, 3)  # Reshape to 3D tensor
                '''x = (torch.zeros([B * t, D, D, Dz, self.mlp_out_dim], device=coords.device, dtype=torch.float)
                     .masked_scatter(mask.unsqueeze(-1).expand(-1, -1, -1, -1, self.mlp_out_dim) > 0,
                                     x_dict['recon_real'])
                     .permute(0, 4, 1, 2, 3))  # x -> 3D'''
                if pretrain:
                    recon = x[:, 0, ...]
                    '''if batch_it is not None and batch_it % 200 == 0:
                        total_memory = torch.cuda.get_device_properties(0).total_memory  # total GPU memory
                        reserved_memory = torch.cuda.memory_reserved(0)  # reserved GPU memory
                        allocated_memory = torch.cuda.memory_allocated(0)  # currently allocated GPU memory
                        free_memory = reserved_memory - allocated_memory  # free GPU memory
                        print(f"Total memory sampler_pre: {total_memory / (1024 ** 2):.2f} MB",
                              f"Reserved memory sampler_pre: {reserved_memory / (1024 ** 2):.2f} MB",
                              f"Allocated memory sampler_pre: {allocated_memory / (1024 ** 2):.2f} MB",
                              f"Free memory sampler_pre: {free_memory / (1024 ** 2):.2f} MB")'''
                recon_mlp = x[:, 0, ...].clone()
            if isinstance(layer, type(self.refin_net)) and not pretrain:
                # print("!!!refin_net!!!!!!!!!refin_net!!!!refin_net!!!!!!!!!")  # mask.shape(B, D, D, D)
                '''padded_map = torch.zeros(
                    (B, D + 2 * (outer_cut // 2), D + 2 * (outer_cut // 2), D + 2 * (outer_cut // 2)),
                    dtype=torch.float, device=coords.device, requires_grad=True)
                padded_map[:, outer_cut // 2: outer_cut // 2 + D,
                outer_cut // 2: outer_cut // 2 + D,
                outer_cut // 2: outer_cut // 2 + D] = x
                coords_mask_positional_encoding_24 = self.positional_encoding_geom(coords_mask, 4)
                # print("coords_mask_positional_encoding_24", coords_mask_positional_encoding_24.shape)
                coords_mask_positional_encoding_3d = torch.zeros([B, D, D, Dz, 24], device=coords.device,
                                                                 dtype=torch.float).masked_scatter(
                    mask.unsqueeze(-1) > 0,
                    coords_mask_positional_encoding_24).permute(0, 4, 1, 2, 3)
                # print("coords_mask_positional_encoding_3d", coords_mask_positional_encoding_3d.shape)

                z_Unet = coords[..., 7:11].reshape(B, D, D, Dz, 4).permute(0, 4, 1, 2, 3)

                # x = torch.cat((x, z_Unet, coords_mask_positional_encoding_3d), dim=1)
                # x = torch.cat((x, z_Unet), dim=1)

                # print("x_cat_PE", x_cat_PE.shape)'''
                padded_x = F.pad(x, (outer_cut // 2, outer_cut // 2,
                                     outer_cut // 2, outer_cut // 2,
                                     outer_cut // 2, outer_cut // 2))
                cut_line = (x[x > 0].mean() * 3).item()
                map_unet_pred = torch.zeros_like(x, dtype=torch.float32)
                denominator = torch.zeros_like(x, dtype=torch.float32)
                t0 = time.time()
                x_np = x.detach().cpu().numpy()
                positive_values = x_np[x_np > 0]
                maximum = np.percentile(positive_values, 99.999) if positive_values.size > 0 else 0

                if self.training:
                    not_learned = True
                    o = 0
                    i = np.arange(0, D - self.box_size, 8)
                    j = np.arange(0, D - self.box_size, 8)
                    k = np.arange(0, Dz - self.box_size, self.box_size)
                    if i[-1] + self.box_size > D:
                        i = i[:-1]
                        j = j[:-1]
                        k = k[:-1]
                    while not_learned:
                        o += 1
                        ran_i = self.pick_centered_normal(i)
                        ran_j = self.pick_centered_normal(j)
                        # ran_k = self.pick_centered_normal(k)
                        valid_batches = []
                        valid_b_indices = []
                        valid_ki_indices = []
                        for ki in k:
                            x_windows = padded_x[:, :, ran_i:ran_i + self.box_size + outer_cut,
                                        ran_j:ran_j + self.box_size + outer_cut,
                                        ki:ki + self.box_size + outer_cut]
                            # max_per_sample = x_windows.reshape(B, -1).max(dim=1).values  # shape: [B]
                            q = torch.quantile(x_windows.reshape(B, -1), 0.99, dim=1)

                            for b in range(B):
                                if q[b] > cut_line:
                                    valid_batches.append(
                                        x_windows[b])  # shape: (C, box+outer_cut, box+outer_cut, box+outer_cut)
                                    valid_b_indices.append(b)
                                    valid_ki_indices.append(ki)

                        if len(valid_batches) > 0:
                            batch_tensor_ori = torch.stack(valid_batches, dim=0)  # (N, C, ...)
                            batch_tensor, rotate_dicts = rotate_tensor(batch_tensor_ori, mean=cut_line / 8,
                                                                       add_nosie=False, use_rotate=True,
                                                                       maximum=maximum, layer=layer)
                            # print("batch_tensor", batch_tensor.shape)# ([13, 1, 48, 48, 48])
                            y = layer(batch_tensor / maximum * 100)

                            y, _ = rotate_tensor(y, rotated=rotate_dicts, reverse=True, add_nosie=False,
                                                 use_rotate=True)
                            y_inter = y[:, 0,
                                      outer_cut // 2:self.box_size + outer_cut // 2,
                                      outer_cut // 2:self.box_size + outer_cut // 2,
                                      outer_cut // 2:self.box_size + outer_cut // 2]
                            for idx, (b, ki) in enumerate(zip(valid_b_indices, valid_ki_indices)):

                                x[:, 0,
                                ran_i:ran_i + self.box_size,
                                ran_j:ran_j + self.box_size,
                                ki:ki + self.box_size][b] = y_inter[idx]
                                unet_box += torch.sum(torch.abs(y[idx]))

                            similarity_per_sample += F.l1_loss(self.normalize_single_box(batch_tensor_ori[:, 0,
                                outer_cut // 2:self.box_size + outer_cut // 2,
                                outer_cut // 2:self.box_size + outer_cut // 2,
                                outer_cut // 2:self.box_size + outer_cut // 2]), self.normalize_single_box(y_inter).detach())


                            not_learned = False
                            print(f"Used unet for {len(valid_batches)} samples!, use {time.time() - t0} second.")
                        if o > len(i):
                            break
                elif not pretrain:
                    range_i = range(0, D - self.box_size, stride)
                    range_j = range(0, D - self.box_size, stride)
                    range_k = range(0, Dz - self.box_size, stride)
                    total = len(range_i) * len(range_j) * len(range_k)

                    with tqdm(total=total, desc="3D sliding windows", unit="win") as pbar:
                        for i, j, k in product(range_i, range_j, range_k):
                            x_windows = padded_x[:, :,
                                        i:i + self.box_size + outer_cut,
                                        j:j + self.box_size + outer_cut,
                                        k:k + self.box_size + outer_cut
                                        ]
                            if x_windows.max() > 0.01:
                                y = layer(x_windows / maximum * 100)  #
                                map_unet_pred[:, 0, i:i + self.box_size, j:j + self.box_size, k:k + self.box_size] += \
                                    y[:, 0,
                                    outer_cut // 2:self.box_size + outer_cut // 2,
                                    outer_cut // 2:self.box_size + outer_cut // 2,
                                    outer_cut // 2:self.box_size + outer_cut // 2]
                                denominator[:, 0, i:i + self.box_size, j:j + self.box_size, k:k + self.box_size] += 1

                            pbar.update(1)
                            if pbar.n % 200 == 0:
                                pbar.set_postfix(i=i, j=j, k=k)

                    map_pred = map_unet_pred / denominator.clip(min=1)
                    x = torch.where(denominator != 0, map_pred, x)

                recon = x[:, 0, ...]
                if self.print_memory and torch.cuda.is_available():
                    for _ in range(10):
                        total_memory = torch.cuda.get_device_properties(0).total_memory  # total GPU memory
                        reserved_memory = torch.cuda.memory_reserved(0)  # reserved GPU memory
                        allocated_memory = torch.cuda.memory_allocated(0)  # currently allocated GPU memory
                        free_memory = reserved_memory - allocated_memory  # free GPU memory
                        print(f"Total memory UNet3D: {total_memory / (1024 ** 2):.2f} MB",
                              f"Reserved memory UNet3D: {reserved_memory / (1024 ** 2):.2f} MB",
                              f"Allocated memory UNet3D: {allocated_memory / (1024 ** 2):.2f} MB",
                              f"Free memory UNet3D: {free_memory / (1024 ** 2):.2f} MB")
                        self.print_memory = False
        if ctf is not None:
            x = torch.mean(x, -1)
            if not x.shape[-1] == D:
                x = x[:, self.box_size // 2:self.box_size // 2 + D, self.box_size // 2:self.box_size // 2 + D]
            x = torch.fft.fftshift(torch.fft.irfft2(
                torch.fft.rfft2(torch.fft.ifftshift(x.float())) * torch.fft.fftshift(ctf)[..., :D // 2 + 1]))
        # print("recon recon_mlp have nan", torch.isnan(recon).any(), torch.isnan(recon_mlp).any())
        reco = {'recon': recon, 'recon_mlp': recon_mlp, 'ij': (ran_i, ran_j)}
        if domain == 'all':
            reco['recon_hartley'] = x_dict['recon_hartley']
        reco['unet_pred'] = unet_box
        if self.training:
            return reco, similarity_per_sample
        else:
            return reco, similarity_per_sample


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias=True):
        super(ResidLinearMLP, self).__init__()
        layers = [
            ResidLinear(in_dim, hidden_dim, bias=bias)
            if in_dim == hidden_dim
            else MyLinear(in_dim, hidden_dim, bias=bias),
            activation(inplace=True), ]

        for _ in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation(inplace=True))

        layers.append(
            ResidLinear(hidden_dim, out_dim, bias=bias)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim, bias=bias)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret.float()


class ImageEncoder(nn.Module):
    def __init__(self, encoder_type: str, latent_dim: int, size=256, hartley=False) -> None:
        super().__init__()

        self.encoder = timm.create_model(encoder_type, num_classes=0, global_pool="", in_chans=1)
        self.output_layer = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 2 * latent_dim))

        # dummy input for initialization
        if hartley:
            self.output_layer(self.encoder(torch.randn(1, 1, size + 1, size + 1)).reshape(1, -1))
        else:
            self.output_layer(self.encoder(torch.randn(1, 1, size, size)).reshape(1, -1))

    def forward(self, images):
        x = F.relu(self.encoder(images).reshape(images.shape[0], -1))
        latent_variable = self.output_layer(x)

        return latent_variable


class Resid3dLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias=True, box_size=16):
        super(Resid3dLinearMLP, self).__init__()
        layers = [
            ResidLinear(in_dim, hidden_dim, bias=bias)
            if in_dim == hidden_dim
            else MyLinear(in_dim, hidden_dim, bias=bias),
            activation(inplace=True),
        ]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation(inplace=True))

        layers.append(
            ResidLinear(hidden_dim, out_dim, bias=bias)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim, bias=bias)
        )
        self.main = nn.Sequential(*layers)
        self.box_size = box_size

    def forward(self, x):
        # x = B x D^3 x (in_dim-zdim)=(3 x D2)
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        ###
        '''if reshape_3d:
            # print("ret", ret.size())
            ret_3d = ret.reshape(ret.shape[0], int(ret.shape[1] ** (1 / 3) + 1), int(ret.shape[1] ** (1 / 3) + 1),
                                 int(ret.shape[1] ** (1 / 3) + 1), -1).permute(0, 4, 1, 2, 3)
            # print("ret_3d", ret_3d.shape)
            map_shape = ret_3d.shape


            padded_map = torch.zeros((map_shape[0], map_shape[1] + 2 * (self.box_size // 2),
                                     map_shape[2] + 2 * (self.box_size // 2),
                                     map_shape[3] + 2 * (self.box_size // 2), map_shape[4]), dtype=torch.float,
                                    device=ret_3d.device)
            padded_map[:, self.box_size // 2: self.box_size // 2 + map_shape[1],
                       self.box_size // 2: self.box_size // 2 + map_shape[2],
                       self.box_size // 2: self.box_size // 2 + map_shape[3], :] = ret_3d
            padded_ret_3d = padded_map.permute(0, 4, 1, 2, 3)'''
        # padded_ret_3d = padded_map.permute(0, 4, 3, 1, 2)

        ###
        return ret
        # return ret, padded_ret_3d


def half_linear(input, weight, bias):
    # print('half', input.shape, weight.shape)
    if bias is not None:
        return F.linear(input, weight.half(), bias.half())
    else:
        return F.linear(input, weight.half())


def single_linear(input, weight, bias):
    # print('single', input.shape, weight.shape)
    # assert input.shape[0] < 10000

    return F.linear(input, weight, bias)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class MyLinear(nn.Linear):
    def forward(self, input):
        if input.dtype == torch.half:
            return half_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight.half(), self.bias.half())
        else:
            return single_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight, self.bias)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super(ResidLinear, self).__init__()
        self.linear = MyLinear(nin, nout, bias=bias)
        # self.linear = nn.utils.weight_norm(MyLinear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias=True, box_size=16):
        super(MLP, self).__init__()
        layers = [MyLinear(in_dim, hidden_dim, bias=bias), activation(inplace=True)]
        for n in range(nlayers):
            layers.append(MyLinear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation(inplace=True))
        layers.append(MyLinear(hidden_dim, out_dim, bias=bias))
        self.main = nn.Sequential(*layers)
        self.box_size = box_size

    def forward(self, x, reshape_3d=False):
        # x = B x D^3 x (in_dim-zdim)=(3 x D2)
        # print("encoder x", x.device)
        ret_dict = dict()
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1]).float()
        ret_dict['recon_real'] = ret
        ###
        if reshape_3d:
            # print("ret", ret.size())
            ret_3d = ret.reshape(ret.shape[0], int(ret.shape[1] ** (1 / 3) + 1), int(ret.shape[1] ** (1 / 3) + 1),
                                 int(ret.shape[1] ** (1 / 3) + 1), -1)
            # print("ret_3d", ret_3d.shape)
            map_shape = ret_3d.shape

            padded_map = torch.zeros((map_shape[0], map_shape[1] + 2 * (self.box_size // 2),
                                      map_shape[2] + 2 * (self.box_size // 2),
                                      map_shape[3] + 2 * (self.box_size // 2), map_shape[4]), dtype=torch.float,
                                     device=ret_3d.device)
            padded_map[:, self.box_size // 2: self.box_size // 2 + map_shape[1],
            self.box_size // 2: self.box_size // 2 + map_shape[2],
            self.box_size // 2: self.box_size // 2 + map_shape[3], :] = ret_3d
            padded_ret_3d = padded_map.permute(0, 4, 1, 2, 3)
            # padded_ret_3d = padded_map.permute(0, 4, 3, 1, 2)

        ###
        return ret_dict  # (B, 1, D, D, D)


class MLP_combine(nn.Module):
    def __init__(self, in_dim_real, in_dim_hartley, nlayers, hidden_dim, out_dim, activation, bias=True):
        super(MLP_combine, self).__init__()
        layers_real = [MyLinear(in_dim_real, hidden_dim, bias=bias), activation(inplace=True)]
        layers_hartley = [MyLinear(in_dim_hartley, hidden_dim, bias=bias), activation(inplace=True)]

        for n in range(nlayers - 1):
            linear = MyLinear(hidden_dim, hidden_dim, bias=bias)
            layers_real.append(linear)
            layers_hartley.append(linear)
            layers_real.append(activation(inplace=True))
            layers_hartley.append(activation(inplace=True))

        layers_real.append(MyLinear(hidden_dim, hidden_dim, bias=bias))
        layers_hartley.append(MyLinear(hidden_dim, hidden_dim, bias=bias))
        layers_real.append(activation(inplace=True))
        layers_hartley.append(activation(inplace=True))

        layers_real.append(MyLinear(hidden_dim, out_dim, bias=bias))
        layers_hartley.append(MyLinear(hidden_dim, out_dim, bias=bias))
        self.main_real = nn.Sequential(*layers_real)
        self.main_hartley = nn.Sequential(*layers_hartley)

    def forward(self, x_real=None, x_hartely=None):
        # x = B x D^3 x (in_dim-zdim)=(3 x D2)
        # print("encoder x", x.device)
        ret = {'recon_real': None, 'recon_hartley': None}
        if x_real is not None:
            flat_real = x_real.view(-1, x_real.shape[-1])
            ret_flat_real = self.main_real(flat_real)
            ret_real = ret_flat_real.view(*x_real.shape[:-1], ret_flat_real.shape[-1]).float()
            ret['recon_real'] = ret_real
        if x_hartely is not None:
            flat_hartley = x_hartely.view(-1, x_hartely.shape[-1])
            # print("flat_hartley", flat_hartley.shape)
            ret_flat_hartley = self.main_hartley(flat_hartley)
            ret_hartley = ret_flat_hartley.view(*x_hartely.shape[:-1], ret_flat_hartley.shape[-1]).float()
            ret['recon_hartley'] = ret_hartley

        return ret  # (B, 1, D, D, D)


# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(in_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        if x.shape[-1] % 64 != 0:
            pad_size = ((x // 64 + 1) * 64 - x) // 2
            x = F.pad(x, [pad_size] * 4)
        x = x.view(x.shape[0], -1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


class ConfTable(nn.Module):
    def __init__(self, n_imgs, z_dim, variational, std_z_init):
        """
        n_imgs: int
        z_dim: int
        variational: bool
        """
        super(ConfTable, self).__init__()
        self.variational = variational
        self.conf_init = torch.tensor(
            std_z_init * np.random.randn(n_imgs, z_dim)
        ).float()
        self.table_conf = nn.Parameter(self.conf_init, requires_grad=True)
        if variational:
            logvar_init = torch.tensor(np.ones((n_imgs, z_dim))).float()
            self.table_logvar = nn.Parameter(logvar_init, requires_grad=True)

    def initialize(self, conf):
        """
        conf: [n_imgs, z_dim] (numpy)
        """
        state_dict = self.state_dict()
        state_dict['table_conf'] = torch.tensor(conf).float()
        self.load_state_dict(state_dict)

    def forward(self, ind):
        """
        in_dict: dict
            index: [batch_size]
            y: [batch_size(, n_tilts), D, D]
            y_real: [batch_size(, n_tilts), D - 1, D - 1]
            R: [batch_size(, n_tilts), 3, 3]
            t: [batch_size(, n_tilts), 2]
            tilt_index: [batch_size( * n_tilts)]

        output: dict
            z: [batch_size, z_dim]
            z_logvar: [batch_size, z_dim] if variational and not pose_only
        """
        conf = self.table_conf[ind]
        # conf_dict = {'z': conf}
        logvar = None
        if self.variational:
            logvar = self.table_logvar[ind]
            # conf_dict['z_logvar'] = logvar
        return conf, logvar

    '''def reset(self):
        state_dict = self.state_dict()
        state_dict['table_conf'] = self.conf_init / 10.0
        self.load_state_dict(state_dict)'''


class PoseTable(nn.Module):
    def __init__(self, n_imgs, no_trans, resolution, use_gt_trans):
        """
        n_imgs: int
        no_trans: bool
        resolution: int
        use_gt_trans: bool
        """
        super(PoseTable, self).__init__()
        s2s2_init = torch.tensor(np.array([1., 0., 0., 0., 1., 0.]).reshape(1, 6).repeat(n_imgs, axis=0)).float()
        self.table_s2s2 = nn.Parameter(s2s2_init, requires_grad=True)
        self.no_trans = no_trans
        self.resolution = resolution
        self.use_gt_trans = use_gt_trans
        if not self.no_trans and not self.use_gt_trans:
            trans_init = torch.tensor(np.zeros((n_imgs, 2))).float()
            self.table_trans = nn.Parameter(trans_init, requires_grad=True)

    def initialize(self, rots, trans):
        """
        rots: [n_imgs, 3, 3] (numpy)
        trans: [n_imgs, 2] (numpy)
        """

        state_dict = self.state_dict()
        # rots must contain "corrected" rotations
        state_dict['table_s2s2'] = lie_tools.rotmat_to_s2s2(torch.tensor(rots).float())
        if 'table_trans' in state_dict:
            # trans must be order 1
            state_dict['table_trans'] = torch.tensor(trans).float()
        self.load_state_dict(state_dict)

    def forward(self, ind, in_dict=None):
        """
        in_dict: dict
            index: [batch_size]
            y: [batch_size(, n_tilts), D, D]
            y_real: [batch_size(, n_tilts), D - 1, D - 1]
            R: [batch_size(, n_tilts), 3, 3]
            t: [batch_size(, n_tilts), 2]
            tilt_index: [batch_size( * n_tilts)]

        output: dict
            R: [batch_size(, n_tilts), 3, 3]
            t: [batch_size(, n_tilts), 2]
        """
        rots_s2s2 = self.table_s2s2[ind]
        rots_matrix = lie_tools.s2s2_to_rotmat(rots_s2s2)
        pose_dict = {'R': rots_matrix}
        if not self.no_trans:
            if not self.use_gt_trans:
                pose_dict['t'] = self.table_trans[ind]

        '''if in_dict['y'].ndim == 4:
            pose_dict['R'] = pose_dict['R'].reshape(*in_dict['y'].shape[:-2], 3, 3)
            if not self.no_trans:
                pose_dict['t'] = pose_dict['t'].reshape(*in_dict['y'].shape[:-2], 2)'''
        return pose_dict


def rotate_tensor(x_windows, mean=0.0, rotated=None, reverse=False, add_nosie=True, use_rotate=True,
                  maximum=None, layer=None):
    batch_size, channels, depth, height, width = x_windows.shape
    rotate = {}

    if rotated is None:
        axis = random.choice(['x', 'y', 'z', 'y', 'z'])
        rotate['axis'] = axis
    else:
        axis = rotated['axis']

    if rotated is None:
        angle = random.choice([0, 90, 180, 270, 90, 270])
        rotate['angle'] = angle
    else:
        angle = rotated['angle']

    if reverse:
        angle = (360 - angle) % 360
    if angle == 0 and not reverse:
        return x_windows, rotate

    if add_nosie:
        x_windows = augment_batch(x_windows, mean=mean)

    if use_rotate:
        if axis == 'x':
            if angle == 90:
                x_windows = x_windows.transpose(3, 4).flip(3)  # 90 degrees clockwise
            elif angle == 180:
                x_windows = x_windows.flip(3).flip(4)  # 180 degrees
            elif angle == 270:
                x_windows = x_windows.transpose(3, 4).flip(4)  # 270 degrees clockwise (i.e. 90 counterclockwise)
        elif axis == 'y':
            if angle == 90:
                x_windows = x_windows.transpose(2, 4).flip(4)  # 90 degrees clockwise
            elif angle == 180:
                x_windows = x_windows.flip(2).flip(4)  # 180 degrees
            elif angle == 270:
                x_windows = x_windows.transpose(2, 4).flip(2)  # 270 degrees clockwise (i.e. 90 counterclockwise)
        elif axis == 'z':
            if angle == 90:
                x_windows = x_windows.transpose(2, 3).flip(3)  # 90 degrees clockwise
            elif angle == 180:
                x_windows = x_windows.flip(2).flip(3)  # 180 degrees
            elif angle == 270:
                x_windows = x_windows.transpose(2, 3).flip(2)  # 270 degrees clockwise (i.e. 90 counterclockwise)
    '''if layer is not None and any(p.requires_grad for p in layer.parameters()):
        x_windows *= maximum / 100'''
    return x_windows, rotate


def random_anisotropy(x, downsampling_factor=(1.5, 2.5), p=0.25):
    if random.random() < p:
        original_size = (x.shape[1], x.shape[2], x.shape[3])
        axis = random.choice([0, 1, 2])
        scale = [1.0, 1.0, 1.0]
        downsampling_factor = random.uniform(*downsampling_factor)
        scale[axis] = 1.0 / downsampling_factor

        x = F.interpolate(x.unsqueeze(0),
                          scale_factor=tuple(scale),
                          mode='nearest')
        x = F.interpolate(x,
                          size=original_size,
                          mode='trilinear',
                          align_corners=False)
        x = x.squeeze(0)
    return x


def random_blur(x, sigma_range=(1, 1), p=1):
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        if sigma <= 0:
            return x
        k = gaussian_kernel_3d(sigma, x.device)
        kernel_size = k.shape[0]
        k = k.view(1, 1, kernel_size, kernel_size, kernel_size)
        k = k.repeat(x.shape[0], 1, 1, 1, 1)
        x = F.conv3d(x.unsqueeze(0),
                     k,
                     padding=kernel_size // 2,
                     groups=x.shape[0]).squeeze(0)
    return x


def random_noise(x, mean=0.0, std=1, p=0.8):
    if random.random() < p:
        noise = torch.randn_like(x) * std + mean
        x = x + noise
    return x


def gaussian_kernel_3d(sigma, device):
    size = max(3, int(sigma * 6) // 2 * 2 + 1)
    ax = torch.arange(size, device=device) - size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2 + 1e-6))
    kernel /= kernel.sum()
    return kernel


def augment_batch(x_windows, mean=0.0):
    b_aug = []
    for b in range(x_windows.shape[0]):
        img = x_windows[b]  # (C, D, H, W)
        # img = random_anisotropy(img)
        # img = random_blur(img)
        img = random_noise(img, mean=mean)
        b_aug.append(img)
    return torch.stack(b_aug, dim=0)  # restore (B, C, D, H, W)


def add_gaussian_noise(x_windows, sigma=0.1):
    batch_size, channels, depth, height, width = x_windows.shape
    noise = torch.randn(batch_size, channels, depth, height, width).to(x_windows.device) * sigma
    x_noisy = x_windows + noise
    return x_noisy
