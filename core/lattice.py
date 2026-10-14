"""Lattice object"""

import numpy as np
import torch
import torch.nn.functional as F
from core import utils

log = utils.log
vlog = utils.vlog


class Lattice:
    def __init__(self, D, Dz=None, extent=0.5, indexing='ij', endpoint=False,
                 centered=False, ignore_DC=True, device=None):

        if not Dz:
            Dz = D

        coords_x = np.linspace(-extent, extent, D, endpoint=endpoint)
        coords_y = np.linspace(-extent, extent, D, endpoint=endpoint)
        coords_z = np.linspace(-extent, extent, Dz, endpoint=endpoint)

        if centered:
            coords_x += extent / D
            coords_y += extent / D
            coords_z += extent / D
        self.centered = centered

        self.coords_x = coords_x
        self.coords_y = coords_y
        self.coords_z = coords_z

        x0, x1, x2 = np.meshgrid(
            coords_x,
            coords_y,
            coords_z,
            indexing=indexing
        )

        coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], -1).astype(
            np.float32
        )

        self.coords = torch.tensor(coords, device=device)
        self.endpoint = endpoint
        self.extent = extent
        self.D = D
        self.Dz = Dz
        self.D2 = int(D / 2)

        # self.freqs2d = self.coords[D//2*D*D:D//2*D*D+D*D, 1:3] / extent / 2
        self.box_size = 16

        # todo: center should now just be 0,0; check Lattice.rotate...
        # c = 2/(D-1)*(D/2) -1
        self.center = torch.tensor([0.0, 0.0], device=device)

        x0, x1 = np.meshgrid(
            coords_x,
            coords_y,
            indexing=indexing
        )
        coords_2d = np.stack([x0.ravel(), x1.ravel()], -1).astype(
            np.float32
        )
        self.coords_2d = torch.tensor(coords_2d, device=device)

        x0_coords_freqs2d, x1_coords_freqs2d = np.meshgrid(
            np.linspace(-extent, extent, D + 1, endpoint=True),
            np.linspace(-extent, extent, D + 1, endpoint=True)
        )
        coords_freqs2d = np.stack([x0_coords_freqs2d.ravel(), x1_coords_freqs2d.ravel(), np.zeros((D + 1) ** 2)],
                                  1).astype(np.float32)
        self.coords_ht = torch.tensor(coords_freqs2d, device=device)
        self.freqs2d = self.coords_ht[:, 0:2] / extent / 2

        self.square_mask = {}
        self.circle_mask = {}

        self.rings_mask = {}
        self.sphere_mask = {}
        self.shell_index = {}

        self.indexing = indexing
        self.ignore_DC = ignore_DC
        self.device = device
        self.R = self.D // 2

    def get_circular_mask(self, R):
        """Return a binary mask for self.coords which restricts coordinates to a centered circular lattice"""
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert (
                2 * R <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using circular lattice with radius {}".format(R))

        r = R / (self.D // 2) * self.extent

        self.circle_mask[R] = self.coords_2d.pow(2).sum(-1) <= r ** 2
        return self.circle_mask[R]

    def get_rings_mask(self, R, minR=4, ignore_low=True):
        if R in self.rings_mask:
            return self.rings_mask[R]
        assert (
                2 * R <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using circular lattice with radius {}".format(R))

        r = R / (self.D // 2) * self.extent
        x0, x1 = np.meshgrid(self.coords_x, self.coords_y, indexing='ij')
        rr = np.sqrt(x0 ** 2 + x1 ** 2)
        if not ignore_low:
            rings = [rr <= minR / R * r]
            rings[0][self.D // 2, self.D // 2, 0] = False
        else:
            rings = []
        for i in range(minR, R):
            rings.append((rr > i / R * r) & (rr <= (i + 1) / R * r))
        rings = np.stack(rings, -1)
        rings = rings / rings.sum((0, 1), keepdims=True).astype(np.float32)
        self.rings_mask[R] = torch.tensor(rings, device=self.device)
        return self.rings_mask[R]

    def get_sphere_mask(self, R, soft_edge=0, is_3d=True):
        if R in self.sphere_mask:
            return self.sphere_mask[R]
        assert (
                2 * R <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using sphere lattice with radius {}".format(R))
        # coords=self.coords.view(self.D,self.D,self.Dz,3)
        self.R = R
        r = R / (self.D // 2) * self.extent
        soft_edge = soft_edge / (self.D // 2) * self.extent
        if soft_edge > 0:
            mask_edge = (self.coords.pow(2).sum(-1) <= r ** 2) & (self.coords.pow(2).sum(-1) > (r - soft_edge) ** 2)
            r_edge = (torch.sqrt(self.coords.pow(2).sum(-1)[mask_edge]) - (r - soft_edge)) / soft_edge * torch.pi
            mask = (self.coords.pow(2).sum(-1) <= (r - soft_edge) ** 2).to(torch.float32).masked_scatter(mask_edge,
                                                                                                         0.5 + 0.5 * torch.cos(
                                                                                                             r_edge))
        else:
            if is_3d:
                coords_3d = self.coords.reshape(self.D, self.D, self.D, 3)
                # print("coords_3d", coords_3d)
                '''map_shape = np.shape(coords_3d)
                padded_map = np.full((map_shape[0] + 2 * (self.box_size//2), map_shape[1] + 2 * (self.box_size//2),
                                      map_shape[2] + 2 * (self.box_size//2), map_shape[3]), 1).astype(np.float32)
                padded_map[self.box_size//2: self.box_size//2 + map_shape[0], self.box_size//2: self.box_size//2 + map_shape[1],
                self.box_size//2: self.box_size//2 + map_shape[2], :] = coords_3d.cpu().numpy()
                padded_map = np.transpose(padded_map, (3, 2, 0, 1))
                map = torch.tensor(padded_map)'''
                mask = coords_3d.pow(2).sum(-1) <= r ** 2
                # mask = torch.tensor(coords_3d).pow(2).sum(-1) <= r ** 2
                # print("mask", mask.shape, "\n", mask)
            else:
                mask = self.coords.pow(2).sum(-1) <= r ** 2
        self.sphere_mask[R] = mask
        return mask

    def get_posesearch_coords(self, l_current, batch_size):
        coords_x = np.linspace(-self.extent, self.extent, l_current, endpoint=self.endpoint)
        coords_y = np.linspace(-self.extent, self.extent, l_current, endpoint=self.endpoint)
        coords_z = np.linspace(-self.extent, self.extent, self.Dz, endpoint=self.endpoint)
        '''
        coords_x = np.linspace(-self.extent, self.extent, l_current, endpoint=self.endpoint)
        coords_y = np.linspace(-self.extent, self.extent, l_current, endpoint=self.endpoint)
        coords_z = np.linspace(-self.extent, self.extent, self.Dz, endpoint=self.endpoint)
        '''
        x0, x1, x2 = np.meshgrid(
            coords_x,
            coords_y,
            coords_z,
            indexing=self.indexing
        )
        coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], -1).astype(
            np.float32
        )
        coords = torch.tensor(coords, device=self.device)

        x0_2d, x1_2d = np.meshgrid(
            coords_x,
            coords_y,
            indexing=self.indexing
        )
        coords_2d = np.stack([x0_2d.ravel(), x1_2d.ravel()], -1).astype(
            np.float32
        )
        coords_2d = torch.tensor(coords_2d, device=self.device)

        r = self.R / (self.D // 2) * self.extent
        coords_3d = coords.reshape(l_current, l_current, self.Dz, 3)
        mask = (coords_3d.pow(2).sum(-1) <= r ** 2).repeat(batch_size, 1, 1, 1)
        coords_2d = coords_2d.reshape(l_current, l_current, 2)
        mask_2d = (coords_2d.pow(2).sum(-1) <= (self.extent * 115 / 128) ** 2).repeat(batch_size, 1, 1)

        return coords.repeat(batch_size, 1, 1), mask, mask_2d.reshape(batch_size, -1)

    def translate_ht(self, img, t, freqs2d=None, mask=None):
        """
        Translate an image by phase shifting its Hartley transform
        H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)

        img: HT of image [B, img_dims]
        t: shift in pixels [batch_size, n_trans, 2]
        mask: Mask for lattice coords [img_dims, 1]

        output: Shifted images [batch_size, n_trans, img_dims]

        img must be 1D unraveled image, symmetric around DC component
        """
        freqs = freqs2d if freqs2d is not None else self.freqs2d
        # print("freqs", freqs)
        # np.savetxt('/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/recon/9076/file.txt', freqs.cpu(), fmt='%0.2f')
        coords = freqs if mask is None else freqs[mask]
        img = img[:, None].to(coords.device)
        t = t[..., None]
        phase = coords @ t * 2 * np.pi
        phase = phase[..., 0]
        c = torch.cos(phase)
        s = torch.sin(phase)

        return c * img + s * img[:, :, torch.arange(len(coords) - 1, -1, -1)]

    def translate_rt(self, images, shifts):
        """
        Translate images.

        Args:
        images (torch.Tensor): input images of shape [batch_size, D, D].
        shifts (torch.Tensor): translations of shape [batch_size, T, 2].

        Returns:
        torch.Tensor: translated images of shape [batch_size, T, D, D].
        """
        batch_size, D, _ = images.shape
        T = shifts.shape[1]
        shifts_normalized = shifts / D * 2
        # Build the affine transformation matrix
        theta = torch.zeros(batch_size, T, 2, 3, device=images.device)

        theta[:, :, 0, 0] = 1  # x-axis scale factor
        theta[:, :, 1, 1] = 1  # y-axis scale factor
        theta[:, :, 0, 2] = shifts_normalized[:, :, 0]  # x-axis translation
        theta[:, :, 1, 2] = shifts_normalized[:, :, 1]  # y-axis translation

        # Generate the affine grid
        grid = F.affine_grid(theta.view(-1, 2, 3), (batch_size * T, 1, D, D), align_corners=False)
        grid = grid.view(batch_size, T, D, D, 2)

        # Sample the images
        translated_images = F.grid_sample(images.unsqueeze(1).repeat(1, T, 1, 1).view(-1, 1, D, D),
                                          grid.view(-1, D, D, 2), align_corners=False, padding_mode='zeros')

        return translated_images
