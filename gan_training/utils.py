from torch import autograd
import torch.nn.functional as F
import numpy as np
import torch
from kornia import create_meshgrid

# Misc
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Transformation mats
mat_blender2opengl = np.array(
      [[ 1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]]
)
mat_opengl2blender = np.array(
      [[1., 0.,  0., 0.],
       [0., 0., -1., 0.],
       [0., 1.,  0., 0.],
       [0., 0.,  0., 1.]]
)
mat_xright_ydown_zaway_to_xright_yup_zto = np.array(
      [[1.,  0.,  0., 0.],
       [0., -1.,  0., 0.],
       [0.,  0., -1., 0.],
       [0.,  0.,  0., 1.]]
)


def get_focal_from_fov(H, fov):
    focal = H / (2 * torch.tan(0.5 * torch.deg2rad(fov)))
    return focal


def reshape_to_image(output, res=None, permute=False):
    # output = B x -1 x 2
    if res is None:
        res = int(np.sqrt(output.shape[1]))
        res = [res, res]
    elif type(res) == int:
        res = [res, res]
    shape = output.shape
    img = output.reshape(shape[0], res[0], res[1], -1)
    if permute:
        img = img.permute(0, 3, 2, 1)
    else:
        img = img.permute(0, 3, 1, 2)
    return img


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg


def compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def get_rays(H, W, focal, c2w, negative_depth=True, K=None, P=None, device="cuda", focal_y=None):
    batch_size = focal.shape[0]
    
    focal = focal.to(device).float()
    if focal_y is None:
        focal_y = focal
    else:
        focal_y = focal_y.to(device).float()

    H, W = H[0], W[0] #, focal[0]
    sign = -1. if negative_depth else 1.
    normal = P is not None
    # Note: create_meshgrid takes the swapping of H and W already into account!
    grid = create_meshgrid(H, W, normalized_coordinates=normal, device=device)
    c2w = c2w.to(device)
    i, j = grid.unbind(-1)
    if P is not None:
        P_inv = torch.inverse(P)
        sign_d = 1. # -1.
        grid = torch.cat([grid, sign_d * torch.ones_like(grid)], dim=-1)
        p_world = (P_inv @ grid.reshape(1, -1, 4).permute(0, 2, 1)).permute(0, 2, 1).reshape(batch_size, W, H, 4)[..., :3]
        p_origin = grid[:, :1, :1, :].clone()
        p_origin[..., :3] = 0.
        rays_o = (P_inv @ p_origin.reshape(1, 4, 1)).permute(0, 2, 1).reshape(batch_size, 1, 1, 4)[..., :3]
        rays_d = p_world - rays_o
    elif K is not None:
        focal = K[0, 0, 0]
        cx = K[0, 0, 2]
        cy = K[0, 1, 2]
        x_lift = (i - cx) / focal
        y_lift = (j - cy) / focal
        dirs = torch.stack([x_lift, y_lift, torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs.reshape(1, H, W, 1, 3) * c2w[:, :3, :3].reshape(batch_size, 1, 1, 3, 3), -1)
        rays_o = c2w[:, :3, -1][:, None, None, :]
    else:
        focal = focal.reshape(-1, 1, 1)
        focal_y = focal_y.reshape(-1, 1, 1)
        sign_y =  sign
        dirs = torch.stack([(i-W*.5)/focal, sign_y*(j-H*.5)/focal_y, sign * torch.ones_like(i/focal)], -1)
        rays_d = torch.sum(dirs.reshape(batch_size, W, H, 1, 3) * c2w[:, :3, :3].reshape(batch_size, 1, 1, 3, 3), -1)
        rays_o = c2w[:, :3, -1][:, None, None, :]
    return rays_o, rays_d


def get_camera_rays(H, W, focal, c2w, device="cuda", focal_y=None, sampling_pattern=None):
    batch_size = focal.shape[0]
    
    if focal_y is None:
        focal_y = focal.clone()

    if focal.device != device:
        focal = focal.to(device).float()
    if focal_y.device != device:
        focal_y = focal_y.to(device).float()

    grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)
    if sampling_pattern is not None:
        grid = grid.permute(0, 3, 1, 2).repeat(batch_size, 1, 1, 1)
        grid = F.grid_sample(grid, sampling_pattern, align_corners=True, mode='bilinear')
        grid = grid.permute(0, 2, 3, 1)

    c2w = c2w.to(device)
    i, j = grid.unbind(-1)
    focal = focal.reshape(-1, 1, 1)
    focal_y = focal_y.reshape(-1, 1, 1)
    dirs = torch.stack([(i-W*.5)/focal, -1*(j-H*.5)/focal_y, -1 * torch.ones_like(i/focal)], -1)
    Hp, Wp = dirs.shape[1:3]
    rays_d = torch.sum(dirs.reshape(batch_size, Wp, Hp, 1, 3) * c2w[:, :3, :3].reshape(batch_size, 1, 1, 3, 3), -1)
    rays_o = c2w[:, :3, -1][:, None, None, :]
    return rays_o, rays_d


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def pose_spherical_ele(theta_mat, phi, radius):
    batch_size = radius.shape[0]
    device = radius.device
    c2w = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    c2w[:, -2, -1] = radius
    w1 = phi / 180. * np.pi
    m1 = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    m1[:, 1, 1] = torch.cos(w1)
    m1[:, 1, 2] = -torch.sin(w1)
    m1[:, 2, 1] = torch.sin(w1)
    m1[:, 2, 2] = torch.cos(w1)
    c2w = m1 @ c2w
    c2w = theta_mat @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).reshape(1, 4, 4).to(device) @ c2w
    return c2w


def pose_spherical_ele_rot(theta_mat, phi_mat, radius):
    batch_size = radius.shape[0]
    device = radius.device
    c2w = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    c2w[:, -2, -1] = radius
    c2w = phi_mat @ c2w
    c2w = theta_mat @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).reshape(1, 4, 4).to(device) @ c2w
    return c2w

def pose_spherical_b(theta, phi, radius):
    batch_size = radius.shape[0]
    device = radius.device
    c2w = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    c2w[:, -2, -1] = radius
    w1 = phi / 180. * np.pi
    m1 = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    m1[:, 1, 1] = torch.cos(w1)
    m1[:, 1, 2] = -torch.sin(w1)
    m1[:, 2, 1] = torch.sin(w1)
    m1[:, 2, 2] = torch.cos(w1)
    c2w = m1 @ c2w
    w2 = theta / 180. * np.pi
    m2 = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    m2[:, 0, 0] = torch.cos(w2)
    m2[:, 0, 2] = -torch.sin(w2)
    m2[:, 2, 0] = torch.sin(w2)
    m2[:, 2, 2] = torch.cos(w2)
    c2w = m2 @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).reshape(1, 4, 4).to(device) @ c2w
    return c2w

def get_rot_theta_from_2x2(rot_mat, make_rot_mat=True):
    rot_mat = rot_mat.reshape(-1, 2, 2)
    if make_rot_mat:
        rot_mat = project_to_so(rot_mat)
    # rot mat is B x 2 x 2
    mat = torch.eye(4, device=rot_mat.device)
    mat = mat.unsqueeze(0).repeat(rot_mat.shape[0], 1, 1)
    mat[:, 0, 0] = rot_mat[:, 0, 0]
    mat[:, 0, 2] = rot_mat[:, 0, 1]
    mat[:, 2, 0] = rot_mat[:, 1, 0]
    mat[:, 2, 2] = rot_mat[:, 1, 1]
    return mat

def get_ele_phi_from_2x2(rot_mat, make_rot_mat=True):
    rot_mat = rot_mat.reshape(-1, 2, 2)
    if make_rot_mat:
        rot_mat = project_to_so(rot_mat)
    # rot mat is B x 2 x 2
    mat = torch.eye(4, device=rot_mat.device)
    mat = mat.unsqueeze(0).repeat(rot_mat.shape[0], 1, 1)
    mat[:, 1, 1] = rot_mat[:, 0, 0]
    mat[:, 1, 2] = rot_mat[:, 0, 1]
    mat[:, 2, 1] = rot_mat[:, 1, 0]
    mat[:, 2, 2] = rot_mat[:, 1, 1]
    return mat

def project_to_so(mat):
    device = mat.device
    mat = mat.cpu()
    batch_size = mat.shape[0]
    u, s, v = torch.svd(mat)
    det_uvt = torch.det(torch.bmm(u, v.permute(0, 2, 1)))
    s_plus = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1)
    s_plus[:, -1, -1] = det_uvt
    rot_mat = torch.bmm(u, torch.bmm(s_plus, v.permute(0, 2, 1)))
    rot_mat = rot_mat.to(device)
    return rot_mat


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_tau = lambda th : torch.Tensor([
    [np.cos(th),-np.sin(th),0, 0],
    [np.sin(th), np.cos(th),0, 0],
    [0,0,1,0],
    [0,0,0,1]]).float()


def sample_patch_mask(H, W, patch_H, patch_W, scale=1., return_coords=False):
    
    assert(scale <= H * 1.0 / patch_H)

    # Do everything in [0, 1] now

    size_patch = [patch_H * scale / H, patch_W * scale / W]    
    assert((size_patch[0] <= 1) and (size_patch[1] <= 1))
    
    start_x = np.random.rand() * (1 - size_patch[0])
    start_y = np.random.rand() * (1 - size_patch[1])

    x_coord = np.linspace(start_x, start_x + size_patch[0], num=patch_H)
    y_coord = np.linspace(start_y, start_y + size_patch[1], num=patch_H)
    coords = np.stack(np.meshgrid(x_coord, y_coord), -1)
    
    # go back to [H, W]
    coords[..., 0] *= H - 1
    coords[..., 1] *= W - 1
    coords = np.round(coords).astype(int)
    if return_coords:
        return coords

    # init mask
    mask = np.zeros((H, W)).astype(bool)
    mask[coords[..., 0], coords[..., 1]] = True
    return mask


def sample_patch(H, W, patch_H, patch_W, scale=1.):
    # This works continiously in [-1, 1]
    assert(scale <= H * 1.0 / patch_H)

    size_patch = [patch_H * scale / H, patch_W * scale / W]    
    assert((size_patch[0] <= 1) and (size_patch[1] <= 1))

    start_x = np.random.rand() * (1 - size_patch[0])
    start_y = np.random.rand() * (1 - size_patch[1])
    x_coord = np.linspace(start_x, start_x + size_patch[0], num=patch_H)
    y_coord = np.linspace(start_y, start_y + size_patch[1], num=patch_W)
    coords = np.stack(np.meshgrid(x_coord, y_coord), -1).astype(np.float32)

    #Scale to [-1, 1]
    coords = coords * 2. - 1.
    return coords


def listify_int(val, batch_size=1):
    if type(val) == int or type(val) == torch.Tensor:
        return [val for i in range(batch_size)]
    else:
        return val


def sample_patch_batch(H, W, patch_H, patch_W, scale=1., batch_size=1, device="cuda"):
    H = listify_int(H, batch_size)
    W = listify_int(W, batch_size)
    patch_H = listify_int(patch_H, batch_size)
    patch_W = listify_int(patch_W, batch_size)
    scale = listify_int(scale, batch_size)

    out = torch.stack([
        torch.from_numpy(sample_patch(H[i], W[i], patch_H[i], patch_W[i], scale[i])).to(device) for i in range(batch_size)
    ])
    return out


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2

    return z


def get_ray_box_intersection(ray_origin, ray_direction,  box_radius=0.55, winding=torch.ones(1, 1)):
    mask_i, d_i = get_one_ray_box_intersection(ray_origin, ray_direction, box_radius, winding)

    ray_origin_inside = ray_origin + ray_direction * (d_i[..., None] + 1e-3)

    mask_i2, d_i2 = get_one_ray_box_intersection(ray_origin_inside, ray_direction, box_radius, -1. * winding)
    # assert(torch.all(mask_i == mask_i2))
    mask_out = mask_i & mask_i2
    d_out = torch.stack([d_i, d_i - 1e-3 + d_i2], dim=-1)
    return mask_out, d_out


def get_one_ray_box_intersection(ray_origin, ray_direction,  box_radius=0.55, winding=torch.ones(1, 1)):
    '''
    Calculates the intersection of the given ray with a orientated bounding box centered
    at the origin and with size of 0.5 * box_radius (default is unit cube). Note that the algorithm
    finds the ray entrance points if exists. 
    This is a custom implementation of the algorithm described in
    "Ray-box intersection algorithm and efficient dynamic voxel rendering" by Majercik et. al. (JCGT 2018)
    Args:
        ray_origin (tensor): ray origin of size B x N x 3
        ray_direction (tensor): ray direction of size B x N x 3
        box_radius (tensor): bounding box size * 0.5
        winding (tensor): whether the ray is coming from outside (1) or inside (-1) the bouding box
    '''
    box_radius = torch.full((1, 1, 3), box_radius)
    
    device = ray_origin.device
    box_radius = box_radius.to(device)
    winding = winding.to(device)

    sgn = -torch.sign(ray_direction)
    d = box_radius * winding.unsqueeze(-1) * sgn - ray_origin

    d /= ray_direction

    def test(d, ray_origin, ray_direction, u, vw):
        #u integer, vw list
        cond_1 = d[..., u] >= 0.
        cond_2 = torch.all(torch.abs(ray_origin[..., vw] + ray_direction[..., vw] * d[..., u].unsqueeze(-1)) < box_radius[..., vw], dim=-1)
        return cond_1 & cond_2
        
    test_val = torch.stack([
        test(d, ray_origin, ray_direction, 0, [1,2]),
        test(d, ray_origin, ray_direction, 1, [2,0]),
        test(d, ray_origin, ray_direction, 2, [0,1]),
    ], dim=-1)

    sgn_2 = torch.zeros_like(sgn)
    sgn_2[test_val] = sgn[test_val]
    sgn_2[test_val == False] = 0
    
    d_mask_0 = sgn_2[..., 0] != 0
    d_mask_1 = (d_mask_0 == False) & (sgn_2[..., 1] != 0)

    distance_out = d[..., 2].clone()
    distance_out[d_mask_0] = d[..., 0][d_mask_0]
    distance_out[d_mask_1] = d[..., 1][d_mask_1]

    found_intersection = (sgn_2[..., 0] != 0) | (sgn_2[..., 1] != 0) | (sgn_2[..., 2] != 0)
    return found_intersection, distance_out


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis=2)

    return r_mat


def sample_pose(range_u=(0, 1), range_v=(0, 0.499), radius=(10, 10)):
    # sample location on unit sphere
    loc = sample_on_sphere(range_u, range_v)
    
    # sample radius if necessary
    radius = radius[0] + np.random.rand() * (radius[1] - radius[0])

    loc = loc * radius
    R = look_at(loc)[0]

    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    out = np.eye(4, dtype=np.float32)
    out[:3, :] = RT
    return out

def sample_pose_batch(range_u=(0, 1), range_v=(0, 0.499), radius=(10., 10.), batch_size=1, to_pytorch=True):
    poses = np.stack([sample_pose(range_u, range_v, radius) for i in range(batch_size)])
    if to_pytorch:
        poses = torch.from_numpy(poses)
    return poses

def log_and_print(logger, txt):
    logger.info(txt)
    print(txt)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception('Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2

def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def normalize_tensor(tensor, start_offset=0., end_offset=0., invert=False):
    min, max = tensor.min(), tensor.max()
    tensor = (tensor - min) / (max - min)
    if invert:
        tensor = 1 - tensor
    tensor = start_offset + tensor * (1- start_offset - end_offset)
    return tensor


def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-6) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


def color_depth_map_tensor(depth):
    # B x H x W or B x 1 x H x W
    if len(depth.shape) == 4:
        depth = depth.squeeze(1)
    
    depth_colored = np.stack([color_depth_map(depth[i].cpu().numpy()) for i in range(depth.shape[0])])
    depth_colored = torch.from_numpy(depth_colored).float() / 255.
    depth_colored = depth_colored.permute(0, 3, 1, 2)
    return depth_colored

def color_depth_map(depths, scale=None):
    """
    Color an input depth map.

    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)

    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """

    _color_map_depths = np.array([
      [0, 0, 0],  # 0.000
      [0, 0, 255],  # 0.114
      [255, 0, 0],  # 0.299
      [255, 0, 255],  # 0.413
      [0, 255, 0],  # 0.587
      [0, 255, 255],  # 0.701
      [255, 255, 0],  # 0.886
      [255, 255, 255],  # 1.000
      [255, 255, 255],  # 1.000
    ]).astype(float)
    _color_map_bincenters = np.array([
      0.0,
      0.114,
      0.299,
      0.413,
      0.587,
      0.701,
      0.886,
      1.000,
      2.000,  # doesn't make a difference, just strictly higher than 1
    ])
  
    mask = depths == 0
    amin, amax = depths[mask ==0].min(), depths.max()
    depths = (depths - amin) / (amax-amin)
    depths[mask] = 0
    scale = depths.max()
    values = np.clip(depths.flatten() / scale, 0, 1)
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
      lower_bin + 1] * alphas.reshape(-1, 1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)
