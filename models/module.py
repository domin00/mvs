import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO

        # Layer 1 filter 3x3 stride = 1
        self.conv1 = nn.Conv2d(3, 8, 3,1,1)
        self.bn1 = nn.BatchNorm2d(8)

        # Layer 2 filter 3x3 stride = 1
        self.conv2 = nn.Conv2d(8, 8, 3,1,1)
        self.bn2 = nn.BatchNorm2d(8)

        # Layer 3 filter 5x5 stride = 2
        self.conv3 = nn.Conv2d(8, 16, 5, 2,2)
        self.bn3 = nn.BatchNorm2d(16)

        # Layer 4 filter 3x3 stride = 1
        self.conv4 = nn.Conv2d(16, 16, 3, 1,1)
        self.bn4 = nn.BatchNorm2d(16)

        # Layer 5 filter 3x3 stride = 1
        self.conv5 = nn.Conv2d(16, 16, 3, 1,1)
        self.bn5 = nn.BatchNorm2d(16)

        # Layer 6 filter 5x5 stride = 2
        self.conv6 = nn.Conv2d(16, 32, 5, 2,2)
        self.bn6 = nn.BatchNorm2d(32)

        # Layer 7 filter 3x3 stride = 1
        self.conv7 = nn.Conv2d(32, 32, 3, 1,1)
        self.bn7 = nn.BatchNorm2d(32)

        # Layer 8 filter 3x3 stride = 1
        self.conv8 = nn.Conv2d(32, 32, 3, 1,1)
        self.bn8 = nn.BatchNorm2d(32)

        # Layer 9 filter 3x3 stride = 1
        self.conv9 = nn.Conv2d(32, 32, 3, 1,1)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        # x: [B,3,H,W]

        # 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)

        x = self.conv9(x)

        return x


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

        self.conv1 = nn.Conv2d(G, 8, 3, 1, 1)

        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)

        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)

        self.convtrans1 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)

        self.convtrans2 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        self.conv4 = nn.Conv2d(8, 1, 3, 1, 1)


        self.relu = nn.ReLU(True)

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO

        # reshape x to aggregate the depth dimension

        x = x.transpose(1, 2)
        B = x.size(0)
        D = x.size(1)
        H = x.size(3)
        W = x.size(4)

        x = x.reshape(B*D, x.size(2), x.size(3), x.size(4))

        # print(x.shape)

        # 1
        x_0 = self.conv1(x)

        # 2
        x_1 = self.conv2(x_0)

        # 3
        x_2 = self.conv3(x_1)

        # 4
        x_3 = self.convtrans1(x_2)

        # 5
        x_4 = self.convtrans2(x_3 + x_1)

        # 6
        output = self.conv4(x_4 + x_0)

        # print(output.view(B, D, H, W).shape)

        return output.view(B, D, H, W)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        xx = depth_values.view(B,D,1) @ x.view(1, H*W)
        yy = depth_values.view(B,D,1) @ y.view(1, H*W)
        zz = depth_values.view(B,D,1) @ torch.ones_like(x).view(1, H*W)

        xyz = torch.stack((xx, yy, zz), dim=2)

        rot_xyz = torch.matmul(rot.view(B,1,3,3), xyz)

        proj_xyz = rot_xyz + trans.view(B, 1, 3, 1)
        grid = proj_xyz[:, :, :2, :] / proj_xyz[:, :, 2:3, :]  # get to 2D by dividing by z coordinate
        x_normalized = grid[:,:,0,:] / ((W-1)/2) - 1
        y_normalized = grid[:, :, 1, :] / ((H - 1) / 2) - 1
        print(x_normalized.shape)
        print(y_normalized.shape)
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # stack x and y grid
        print(grid.shape)


    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea,grid,mode='bilinear', padding_mode='zeros', align_corners=True)


    return warped_src_fea.view(B, C, D, H, W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO

    # rearrange dimensions of reference and warped to include G number of groups, split channels into G groups
    # also add D dimension = 1 for reference for uniform dimensions
    ref_fea = ref_fea.view(ref_fea.size(0), G, ref_fea.size(1)//G, 1, ref_fea.size(2), ref_fea.size(3))
    warped_src_fea = warped_src_fea.view(ref_fea.size(0), G, warped_src_fea.size(1)//G, warped_src_fea.size(2), warped_src_fea.size(3), warped_src_fea.size(4))

    # print(ref_fea.shape)
    # print(warped_src_fea.shape)

    # element-wise multiplication of both features
    # mean in the C/G dimension removes channel dimension and applies C/G to tensor
    sim = torch.mul(warped_src_fea, ref_fea).mean(2)
    # print("Group wise corr:")
    # print(sim.shape)


    return sim


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO

    # print(p.shape)
    # print(depth_values.shape)

    d = depth_values.view(depth_values.size(0), depth_values.size(1), 1, 1)
    # print(d.shape)

    depth = torch.sum(p*d, dim=1)
    # print(depth.shape)

    # expected out from loss function: [B,1,H,W]
    # depth = depth.unsqueeze(1)

    return depth

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    mask = mask.bool()

    loss = nn.L1Loss()

    l1 = loss(depth_est[mask], depth_gt[mask])

    return l1
