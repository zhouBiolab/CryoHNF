import torch
import torch.nn.functional as F
import numpy as np
from core import fft

from pose_search.pose_find import mask as masking, so3_grid, shift_grid, lie_tools

MAX_POSES = 2048


class Projector:
    def __init__(self, vol, tilt=None, device=None):
        nz, ny, nx = vol.shape
        assert nz == ny == nx, "Volume must be cubic"
        x2, x1, x0 = np.meshgrid(
            np.linspace(-1, 1, nz, endpoint=True),
            np.linspace(-1, 1, ny, endpoint=True),
            np.linspace(-1, 1, nx, endpoint=True),
            indexing="ij",
        )
        lattice = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], 1).astype(np.float32)
        # self.lattice = torch.from_numpy(lattice)
        self.lattice = torch.tensor(lattice, device=device)
        # self.vol = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.vol = torch.tensor(vol, device=device).unsqueeze(0).unsqueeze(0)
        self.nz = nz
        self.ny = ny
        self.nx = nx

        D = nx
        c = 2 / (D - 1) * (D / 2) - 1  # normalized coordinate of the center pixel
        self.center = torch.tensor([c, c, c], device=device)

        self.tilt = None
        if tilt is not None:
            assert tilt.shape == (3, 3)
            self.tilt = torch.tensor(tilt, dtype=torch.float32)

    def rotate(self, rot):
        B = rot.size(0)
        if self.tilt is not None:
            rot = self.tilt @ rot
        grid = self.lattice @ rot  # B x D^3 x 3
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = (
                self.center - grid[:, int(self.nz / 2), int(self.ny / 2), int(self.nx / 2)]
        )
        grid += offset[:, None, None, None, :]
        grid = grid.view(1, -1, self.ny, self.nx, 3)[..., [2, 1, 0]]
        vol = F.grid_sample(self.vol, grid)
        vol = vol.view(B, self.nz, self.ny, self.nx)
        return vol

    def project(self, rot):
        return self.rotate(rot).permute(0, 3, 2, 1)  # .mean(-1)


def get_base_shifts(ps_params):
    return torch.tensor(
        shift_grid.base_shift_grid(
            ps_params['base_healpy'] - 1,
            ps_params['t_extent'],
            ps_params['t_n_grid'],
            x_shift=ps_params['t_xshift'],
            y_shift=ps_params['t_yshift']
        )
    ).float()


def get_base_rot(ps_params):
    base_quat = so3_grid.s2_grid_so3(ps_params['base_healpy'])
    return lie_tools.quat_to_rotmat(to_tensor(base_quat))


def get_so3_base_quat(ps_params):
    return to_tensor(so3_grid.grid_so3(ps_params['base_healpy']))


def get_base_inplane(ps_params):
    return to_tensor(so3_grid.grid_s1(ps_params['base_healpy']))


def to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def get_l(step, res, ps_params):
    if ps_params['niter'] > 0:
        l_current = ps_params['l_min'] + int(step / ps_params['niter'] * (ps_params['l_max'] - ps_params['l_min']))
    else:
        l_current = ps_params['l_max']
    return min(l_current, res // 2)


def translate_images(
        images,
        shifts,
        l_current,
        lattice,
        freqs2d
):
    """
    images: [batch_size, D, D]
    shifts: [batch_size, T, 2] or [..., T, 2]

    output: [batch_size, T, n_pts] at resolution l_current
    """
    batch_size = images.shape[0]
    mask = masking.get_circular_mask(lattice, l_current).to(images.device)

    if images.dim() == 2:
        images = images.unsqueeze(0)  # [1, H, W]
    B, H, W = images.shape[-3], images.shape[-2], images.shape[-1]

    im_flat = images.reshape(B, -1)

    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask)
    else:
        mask_t = mask
    mask_t = mask_t.to(im_flat.device)
    # mask_t = mask_t.reshape(-1).to(dtype=torch.bool)
    assert mask_t.numel() == H * W, \
        f"mask numel {mask_t.numel()} != H*W {H * W} (H={H}, W={W})"

    im_sel = im_flat[:, mask_t]

    return lattice.translate_ht(
        im_sel,  # ,images.reshape(batch_size, -1)[:, mask]
        shifts,
        mask=mask,
        freqs2d=freqs2d
    )


def rot_2d(
        angle,
        out_d,
        device
):
    rot = torch.zeros((out_d, out_d), device=device)
    rot[0, 0] = torch.cos(angle)
    rot[0, 1] = -torch.sin(angle)
    rot[1, 0] = torch.sin(angle)
    rot[1, 1] = torch.cos(angle)
    return rot


def rot_2d_tensor(
        angles,
        out_d,
        device
):
    rot = torch.zeros((*angles.shape, out_d, out_d), device=device)
    rot[..., 0, 0] = torch.cos(angles)
    rot[..., 0, 1] = -torch.sin(angles)
    rot[..., 1, 0] = torch.sin(angles)
    rot[..., 1, 1] = torch.cos(angles)
    # rot[..., 2, 2] = torch.tensor(1)
    return rot


def interpolate(
        img,
        coords
):
    # print(f"Interpolating {img.shape} {coords.shape}")
    assert len(coords.shape) == 2
    assert coords.shape[-1] == 2
    grid = coords * 2  # careful here! grid_sample expects [-1,1] instead of [-0.5,0.5]
    grid = grid[None, None, ...].expand(img.shape[0], -1, -1, -1)

    output = (
        F.grid_sample(
            img.unsqueeze(1),
            grid,
            align_corners=False
        )
        .squeeze(2)
        .squeeze(1)
    )

    return output


def rotate_images(
        images,
        angles,
        l_current,
        masked_coords,
        lattice
):
    batch_size, d1, nq, yx = images.shape
    bnq = batch_size * nq
    device = images.device
    # print("images.shape", images.shape)
    squeezed_images = images.reshape(bnq, yx)
    res = lattice.D + 1
    output = torch.zeros((bnq, len(angles), yx), device=device)
    mask = masking.get_circular_mask(lattice, l_current).to(device)

    rot_matrices = torch.stack([rot_2d(a, 2, images.device) for a in angles], dim=0)
    lattice_coords = masked_coords[:, :2]
    # lattice_coords[:, :2] = lattice_coords[:, :2][:, [1, 0]]
    '''np.savetxt('/wanl/heterogeneity/Comp-het/Ribosembly/Ribosembly/recon/9076/masked_coords.txt',
               lattice_coords.cpu().numpy(), fmt='%0.2e')'''
    # lattice_coords = torch.flip(masked_coords[0, :, :, res // 2, :2], dims=[-1]).reshape(-1, 2)
    rot_coords = lattice_coords @ rot_matrices

    full_images = torch.zeros((bnq, res, res), device=device)
    full_images.reshape(bnq, res * res)[:, mask] = squeezed_images

    for angle_idx, interp_coords in enumerate(rot_coords):
        interpolated = interpolate(full_images, interp_coords)
        assert squeezed_images.shape == interpolated.shape
        # IMPORTANT TRICK HERE!
        interpolated *= squeezed_images.std(-1, keepdim=True) / interpolated.std(
            -1, keepdim=True
        )  # FIXME
        # TODO: ask about this

        output[:, angle_idx] = interpolated

    return output.reshape(batch_size, 1, nq * len(angles), yx)


def compute_err(
        model,
        images,
        rot,
        lattice,
        masked_coords,
        angles_inplane,
        z=None,
        nq=None,
        yx=None,
        l_current=None,
        ctf_i=None,
        tilting_func=None,
        apply_tilting_scheme=False,
        in_plane_in_image_space=True  # True
) -> torch.Tensor:
    device = images.device

    adj_angles_inplane = None
    n_in_planes = 1
    if angles_inplane is not None:  # for base grid only
        if in_plane_in_image_space and not apply_tilting_scheme:
            # apply a random in-plane rotation from the set
            # to avoid artifacts due to grid alignment
            rand_a = angles_inplane[np.random.randint(len(angles_inplane))]
            rand_inplane_rot = rot_2d(rand_a, 3, device)
            rand_inplane_rot[2, 2] = torch.tensor(1)
            '''rot = rot
            adj_angles_inplane = -angles_inplane'''
            rot = rand_inplane_rot @ rot
            adj_angles_inplane = angles_inplane - rand_a
            adj_angles_inplane *= -1
        else:
            rot = torch.matmul(rot_2d_tensor(-angles_inplane, 3, device), rot[..., None, :, :])
            rot = rot.reshape(-1, 3, 3)  # [nq * ip = 4608, 3, 3]
            n_in_planes = len(angles_inplane)  # 24

    n_tilts = 1
    if apply_tilting_scheme:
        rot = tilting_func(rot)
        n_tilts = rot.shape[-3]
        rot = rot.reshape(-1, 3, 3)  # [nq * ip * n_tilts, 3, 3]
    batch_size = torch.div(ctf_i.shape[0], n_tilts, rounding_mode='trunc')

    sphere_mask = lattice.get_sphere_mask(lattice.D // 2).repeat(1, 1, 1, 1)
    # x = masked_coords @ rot.to(device)
    x = lattice.coords.repeat(1, 1, 1)
    # x[..., [0, 1]] = x[..., [1, 0]]
    # print("lattice.coords", lattice.coords.shape) [2097152, 3]
    with torch.no_grad():
        # x: [b, nq, npts, 3], [b * nq, npts, 3]
        # z: [b, z_dim], [b * nq, z_dim]
        y_hat = model.decoder(x, pretrain=True, in_posesearch=True, mask=sphere_mask, dict={'transtoreal': True})
        # rot[..., [0, 1], :] = rot[..., [1, 0], :]
        y_hat = y_hat['recon_real'].float()  # [b, nq, npts], [b * nq, npts]
        # print("y_hat", y_hat.shape) ([1, 1, 128, 128, 128])
        projector = Projector(y_hat[0, 0, ...], device=lattice.device)
        # rot[..., [0, 1], :] = rot[..., [1, 0], :]
        # print("rot", rot.shape)([192, 3, 3])([384, 3, 3])
        y_hat = projector.project(rot.to(device))
        # print("proj", proj.shape) torch.Size([192, 128, 128, 128])
        y_hat = y_hat.mean(-1)
        y_hat = fft.symmetrize_ht(fft.ht2_center(y_hat)).to(device)
        y_hat = y_hat.reshape(-1, 1, nq * n_in_planes, n_tilts, y_hat.shape[-1] ** 2)
        mask = masking.get_circular_mask(lattice, l_current).view(1, 1, 1, 1, -1).to(device)  # ([16641])
        y_hat = torch.masked_select(y_hat, mask)
    y_hat = y_hat.reshape(-1, 1, nq * n_in_planes, n_tilts, yx)
    # [1/b, 1, nq * ip, n_tilts, yx] for base grid, [b, 1, 8, n_tilts, yx] for incremental grid
    y_hat = torch.permute(y_hat, (0, 3, 1, 2, 4)).reshape(-1, 1, nq * n_in_planes, yx)
    # [b * n_tilts, 1, nq * ip, yx], [b * n_tilts, 1, 8 * ip, yx]

    if ctf_i is not None:
        y_hat = y_hat * ctf_i

    if adj_angles_inplane is not None:
        y_hat = rotate_images(y_hat, adj_angles_inplane, l_current, masked_coords, lattice)

    images = images.unsqueeze(2)  # [b * n_tilts, t, 1, yx]
    batch_size, t, _, n_pts = images.shape

    # computational speed up of mse
    dots = (images.reshape(batch_size, -1, n_pts).to(y_hat.device)
            @ y_hat.reshape(y_hat.shape[0], -1, n_pts).transpose(-1, -2)
            ).reshape(batch_size, t, -1)
    norm = (y_hat * y_hat).sum(-1) / 2
    # norm_images =
    err = -dots + norm  # + (images * images).sum(-1) / 2 [b (* n_tilts), t, nq * ip]

    if apply_tilting_scheme:
        err = err.reshape(-1, n_tilts, *err.shape[-2:])  # [b, n_tilts, t, nq * ip]
        err = err.mean(1)  # [b, t, nq * ip]

    return err


def eval_grid(
        model,
        images,
        rot,
        lattice,
        coords,
        z=None,
        nq=None,
        l_current=None,
        angles_inplane=None,
        ctf_i=None,
        tilting_func=None,
        apply_tilting_scheme=False
):
    batch_size = images.shape[0]
    device = images.device

    mask = masking.get_circular_mask(lattice, l_current).to(device)
    # print("mask shape", mask.shape) torch.Size([16641])
    masked_coords = coords[mask]
    yx = masked_coords.size(-2)
    # print("masked_coords shape", masked_coords.shape) [12852, 3])

    ctf_i_masked = None
    if ctf_i is not None:
        ctf_i_masked = ctf_i.reshape(batch_size, 1, 1, -1)[..., mask]

    err = compute_err(
        model,
        images,
        rot,
        lattice,
        masked_coords,
        angles_inplane,
        z=z,
        nq=nq,
        yx=yx,
        l_current=l_current,
        ctf_i=ctf_i_masked,
        tilting_func=tilting_func,
        apply_tilting_scheme=apply_tilting_scheme
    )
    return err


def keep_matrix(
        loss,
        batch_size,
        max_poses
):
    """
    loss: [batch_size, t, q]: tensor of losses for each translation and rotation.

    output: 3 * [batch_size, max_poses]: bool tensor of rotations to keep, along with the best translation for each.
    """
    shape = loss.shape
    assert len(shape) == 3
    best_loss, best_trans_idx = loss.min(1)
    flat_loss = best_loss.reshape(batch_size, -1)
    flat_idx = flat_loss.topk(max_poses, dim=-1, largest=False, sorted=True)[1]
    # add the batch index in, to make it completely flat
    flat_idx += (
            torch.arange(batch_size, device=loss.device).unsqueeze(1) * flat_loss.shape[1]
    )
    flat_idx = flat_idx.reshape(-1)

    keep_idx = torch.empty(
        len(shape), batch_size * max_poses, dtype=torch.long, device=loss.device
    )
    keep_idx[0] = torch.div(flat_idx, shape[2], rounding_mode='trunc')
    keep_idx[2] = flat_idx % shape[2]
    keep_idx[1] = best_trans_idx[keep_idx[0], keep_idx[2]]

    return keep_idx


def get_neighbor_so3(
        quat,
        q_ind,
        res,
        device
):
    """
    quat: [nq, 4]
    q_ind: [nq, 2], np.array
    cur_res: int

    output: [nq, 8, 4], [nq, 8, 2] (np.array)
    """
    # Memoization has been removed here
    return so3_grid.get_neighbor_tensor(quat, q_ind, res, device)


def subdivide(
        quat,
        q_ind,
        cur_res,
        device
):
    """
    Subdivides poses for next resolution level.

    quat: [nq, 4]: quaternions
    q_ind: [nq, 2]: np.array, index of current S2xS1 grid
    cur_res: int: Current resolution level

    output:
        quat  [nq, 8, 4]
        q_ind [nq, 8, 2]
        rot   [nq * 8, 3, 3]
    """
    quat, q_ind = get_neighbor_so3(quat, q_ind, cur_res, device)
    rot = lie_tools.quat_to_rotmat(quat.reshape(-1, 4))
    return quat, q_ind, rot


def opt_trans(
        model,
        y_gt,
        y_pred,
        lattice,
        ps_params,
        current_radius
):
    """
    model: DrgnAI
    y_gt: [(sym_loss_factor * ) batch_size, D, D]
    y_pred: [(sym_loss_factor * ) batch_size, n_pts]
    lattice: Lattice
    ps_params: dict
        'l_min': int
        'l_max': int
        't_extent': float
        't_n_grid': int
        'niter': int
        'nkeptposes': int
        'base_healpy': int
        't_xshift': float
        't_yshift': float
    current_radius: int

    output: [(sym_loss_factor * ) batch_size, n_pts], [(sym_loss_factor * ) batch_size, 2]
    """
    freqs2d = model.freqs2d
    base_shifts = model.base_shifts  # [T, 2]
    best_trans = torch.zeros(1).float().to(base_shifts.device)
    mask_cpu = masking.get_circular_mask(lattice, current_radius).cpu()

    best_trans_idx = None
    translated_images = None
    for iter_ in range(0, ps_params['niter'] + 1):
        if iter_ < ps_params['niter']:
            l_current = min(get_l(iter_, lattice.D, ps_params), current_radius)
        else:
            l_current = current_radius
        mask = masking.get_circular_mask(lattice, l_current).cpu()
        trans = best_trans[:, None] + base_shifts / (2. ** iter_)  # [batch_size, T, 2]
        translated_images = translate_images(y_gt, trans, l_current, lattice, freqs2d)  # [batch_size, T, n_pts]
        y = y_pred[:, None][..., mask[mask_cpu]]
        loss = torch.sum((translated_images - y) ** 2, -1)  # [batch_size, T]
        best_loss, best_trans_idx = loss.min(1)  # [batch_size], [batch_size]
        if iter_ == 0:
            best_trans = trans[best_trans_idx]  # [batch_size, 2]
        else:
            best_trans = trans[np.arange(trans.shape[0]), best_trans_idx]  # [batch_size, 2]

    y_gt_translated = translated_images[np.arange(translated_images.shape[0]), best_trans_idx]

    return y_gt_translated, best_trans


def opt_theta_trans(
        model,
        images,
        lattice,
        ps_params,
        z=None,
        ctf_i=None,
        gt_trans=None,
        trans_search_factor=None
):
    """
    model: DrgnAI
    images: [batch_size(, n_tilts), D, D]
    lattice: Lattice
    ps_params: dict
        'l_min': int
        'l_max': int
        't_extent': float
        't_n_grid': int
        'niter': int
        'nkeptposes': int
        'base_healpy': int
        't_xshift': float
        't_yshift': float
    z: [batch_size, z_dim]
    ctf_i: [batch_size(, n_tilts), D, D]
    gt_trans: [batch_size(, n_tilts), 2]
    trans_search_factor: float

    output: [batch_size(, n_tilts), 3, 3], [batch_size(, n_tilts), 2], ?
    """
    # assert not model.hypervolume.training

    apply_tilting_scheme = False
    if images.ndim == 4:
        subtomogram_averaging = True
        assert ps_params['t_extent'] == 0.  # translations cannot be searched over when tilts are jointly optimized
        tilts = images[:, :ps_params['n_tilts_pose_search']]  # [batch_size, n_tilts, D, D]
        n_tilts_out = images.shape[1]
        if ps_params['average_over_tilts']:
            tilts = torch.mean(tilts, 1)  # [batch_size, D, D]
            particles = tilts  # [batch_size, D, D]
            ctf_selected = ctf_i[:, 0] if ctf_i is not None else None
            gt_trans_selected = gt_trans[:, 0] if gt_trans is not None else None
            apply_tilting_scheme = False
            n_tilts = 1
        else:
            particles = tilts[:, 0]  # [batch_size, D, D]
            n_tilts = tilts.shape[1]
            tilts = tilts.reshape(-1, *tilts.shape[-2:])  # [batch_size * n_tilts, D, D]
            ctf_selected = ctf_i.reshape(-1, *ctf_i.shape[-2:])
            gt_trans_selected = gt_trans.reshape(-1, 2) if gt_trans is not None else None  # [batch_size * n_tilts, 2]
            apply_tilting_scheme = True
    else:
        tilts = images  # [batch_size, D, D]
        particles = images  # [batch_size, D, D]
        subtomogram_averaging = False
        ctf_selected = ctf_i
        gt_trans_selected = gt_trans
        n_tilts = 1

    tilts = fft.ht2_center(-tilts)
    tilts = np.asarray(tilts, dtype=np.float32)
    tilts *= -1
    tilts = fft.symmetrize_ht(tilts)
    tilts = tilts
    particles = to_tensor(particles)
    tilts = to_tensor(tilts)
    z = to_tensor(z)

    batch_size = particles.shape[0]
    res = particles.shape[1]

    device = model.coords_ht.device
    coords = model.coords_ht
    freqs2d = model.freqs2d
    _base_shifts = gt_trans_selected.clone().to(device).unsqueeze(
        1) if gt_trans_selected is not None else model.base_shifts
    base_shifts = _base_shifts * trans_search_factor if trans_search_factor is not None else _base_shifts
    # Generate uniformly distributed rotation matrices over SO(3).
    base_rot = model.base_rot  # [192, 3, 3]
    nq = len(base_rot)  # 192
    # Build a uniform grid over SO(3) as a set of quaternions.
    so3_base_quat = model.so3_base_quat  # [4608, 4]
    # Generate a uniform grid on the unit circle; each entry is an angle in radians within [0, 2*pi].
    base_inplane = model.base_inplane  # [24]

    l_current = get_l(0, res, ps_params)  # 12 17 22 27 32: frequency selected by step 0..4
    rot = None

    if z is not None:
        base_rot = base_rot.expand(batch_size, *base_rot.shape)
    loss = eval_grid(
        model,
        translate_images(tilts, base_shifts, l_current, lattice, freqs2d),  # translate images by _base_shifts in Hartley space
        base_rot,
        lattice,
        coords,
        z=z,
        nq=nq,
        l_current=l_current,
        angles_inplane=base_inplane,
        ctf_i=ctf_selected,
        tilting_func=ps_params['tilting_func'],
        apply_tilting_scheme=apply_tilting_scheme
    )

    # From the loss, pick the ps_params['nkeptposes'] = 8 smallest-loss batch indices, translation indices and rotation indices, [b*8] each.
    keep_b, keep_t, keep_q = keep_matrix(loss, batch_size, ps_params['nkeptposes']).cpu()
    # Combine the 8 smallest translation/rotation indices into a [B,8,2] matrix.
    new_init_poses = (torch.cat((keep_t, keep_q), dim=-1)
                      .reshape(2, batch_size, ps_params['nkeptposes'])
                      .permute(1, 2, 0))

    # The 8 smallest-loss rotation indices with their quaternions. [8b,4]
    quat = so3_base_quat[keep_q]

    # Map SO(3) rotation indices back to S2 and S1 grid indices: rotation-axis direction and in-plane rotation angle. [8b,2]
    q_ind = so3_grid.get_base_ind(keep_q, ps_params['base_healpy'])

    # When ground-truth translations are provided, clone and expand them to match the number of kept poses
    # Select the corresponding translations from base_shifts by the keep_t indices.
    if gt_trans_selected is not None:
        trans = gt_trans_selected \
            .clone() \
            .to(device) \
            .unsqueeze(1) \
            .repeat(1, ps_params['nkeptposes'], 1) \
            .reshape(-1, 2).unsqueeze(1)  # batch_size * n_tilts * nkeptposes, 1, 2
        shifts = None
    elif ps_params['t_extent'] < 1e-6:
        trans = torch.zeros(batch_size * n_tilts * ps_params['nkeptposes'], 1, 2).float().to(device)
        shifts = None
    else:
        trans = base_shifts[keep_t]
        shifts = base_shifts.clone()

    for iter_ in range(1, ps_params['niter'] + 1):
        # keep_b is [b*8]; keep_b8 is [b*8*8] with repeated batch indices (e.g. 0000000011111111)
        keep_b8 = (
            keep_b.unsqueeze(1).repeat(1, 8).reshape(-1)
        )  # repeat each element 8 times
        # Repeat the 8 smallest-loss z of each b eight times. zb[b*8*8, zdim]
        zb = z[keep_b8] if z is not None else None

        # 17 22 27 32: frequency selected by iter_ 1..4
        l_current = get_l(iter_, res, ps_params)

        # Refine rotations at the iter_ resolution: around each of the 8 smallest-loss quaternions take 8 neighboring quaternions; q_ind is S2xS1.
        # quat[8b, 8, 4]; q_ind [8b, 8, 2]; rot [8b * 8, 3, 3]
        quat, q_ind, rot = subdivide(quat, q_ind, iter_ + ps_params['base_healpy'] - 1, device)

        # Refine translations at the iter_ resolution: from the 8 smallest-loss shifts, search offsets at half the previous grid spacing.
        if gt_trans_selected is None and ps_params['t_extent'] > 1e-6:
            shifts /= 2
            trans = trans.unsqueeze(1) + shifts.unsqueeze(0)

        tiltsb = tilts.reshape(-1, n_tilts, *tilts.shape[-2:])[keep_b].reshape(-1, *tilts.shape[-2:])
        ctfb = ctf_selected.reshape(-1, n_tilts, *ctf_selected.shape[-2:])[keep_b].reshape(-1, *ctf_selected.shape[
                                                                                                -2:]) if ctf_selected is not None else None
        loss = eval_grid(
            model,
            translate_images(tiltsb, trans, l_current, lattice, freqs2d),
            rot,
            lattice,
            coords,
            z=zb,
            nq=8,
            l_current=l_current,
            ctf_i=ctfb,
            tilting_func=ps_params['tilting_func'],
            apply_tilting_scheme=apply_tilting_scheme
        )
        nkeptposes = ps_params['nkeptposes'] if iter_ < ps_params['niter'] else 1

        keep_bn, keep_t, keep_q = keep_matrix(loss, batch_size, nkeptposes).cpu()  # B x (self.Nkeptposes*32)
        keep_b = torch.div(keep_bn * batch_size, loss.shape[0], rounding_mode='trunc')
        assert (
                len(keep_b) == batch_size * nkeptposes
        ), f"{len(keep_b)} != {batch_size} x {nkeptposes} at iter {iter_}"
        quat = quat[keep_bn, keep_q]
        q_ind = q_ind[keep_bn, keep_q]
        if gt_trans_selected is None and ps_params['t_extent'] > 1e-6:
            trans = trans[keep_bn, keep_t]

    assert loss is not None
    best_bn, best_t, best_q = keep_matrix(loss, batch_size, 1).cpu()
    assert len(best_bn) == batch_size
    assert rot is not None
    best_rot = rot.reshape(-1, 8, 3, 3)[best_bn, best_q]
    if gt_trans_selected is not None:
        best_trans = gt_trans_selected.clone().to(device)
    elif ps_params['t_extent'] < 1e-6:
        best_trans = torch.zeros(batch_size * n_tilts, 2).float().to(device)
    else:
        best_trans = trans.to(device)

    if subtomogram_averaging:
        best_rot = ps_params['tilting_func'](best_rot)
        # We need to connect the pose search and the tilting scheme,
        # which is a property of the dataset for now.
        # We should instead create a TiltingScheme object from the dataset,
        # make it an attribute of the model and pass it as an argument to the pose search function.
        if not apply_tilting_scheme:
            best_trans = best_trans[:, None].expand(-1, n_tilts_out, -1)
        else:
            best_trans = best_trans.reshape(-1, n_tilts_out, 2)

    return best_rot, best_trans, new_init_poses
