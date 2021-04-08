import torch
from net import PConvUNet, VGG16FeatureExtractor
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torch.utils.data import DataLoader
from dataset import Dataset
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torchvision.utils import save_image
from Losses import AdversarialLoss
import Discriminator
import os
import time

class PconvUNetFull():
    def __init__(self, opt):
        self.opt = opt
        self.G = PConvUNet()
        self.lossNet = VGG16FeatureExtractor()
        self.D = Discriminator.Discriminator(3)
        if opt.finetune:
            self.lr = opt.finetune_lr
            self.G.freeze_enc_bn = True
        else:
            self.lr = opt.train_lr
        self.adv_loss = AdversarialLoss()
        self.start_iter = opt.start_iter
        print(self.start_iter)
        self.optm_G = optim.Adam(self.G.parameters(), lr = self.lr)
        self.optm_D = optim.Adam(self.D.parameters(), lr = self.lr*0.1)
        if opt.resume:
            start_iter = load_ckpt(opt.save_dir + "/ckpt/g_{:d}.pth".format(self.start_iter), [('generator', self.G)])
            self.optm_G = optim.Adam(self.G.parameters(), lr = self.lr)
            for param_group in self.optm_G.param_groups:
                param_group['lr'] = self.lr
            print('Starting from iter ', start_iter)
            self.start_iter = start_iter
        
        
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss = 0.0
        self.D_loss = 0.0

        
        if torch.cuda.is_available():
            self.device = torch.device(opt.device)
            if opt.device == "cuda":
                self.G.cuda()
                self.D.cuda()
                self.lossNet.cuda()
                self.adv_loss.cuda()
        else:
            self.device = torch.device("cpu")
        
        if self.opt.mode == 2:
            self.test_dataset = Dataset(opt, opt.test_root, opt.test_edge_root, opt.test_mask_root, augment=False, mask_reverse = True)
        else:
            self.train_dataset = Dataset(opt, opt.train_root, opt.train_edge_root, opt.train_mask_root, augment=True, training=True, mask_reverse = True)
            self.val_dataset = Dataset(opt, opt.val_root, opt.val_edge_root, opt.val_mask_root, augment=False, training=True, mask_reverse = True)
            self.sample_iterator = self.val_dataset.create_iterator(opt.batch_size)
        
    def train(self):
        writer = SummaryWriter(log_dir="log_info")

        self.G.train(freeze_enc_bn=self.opt.finetune)
        if self.opt.finetune:
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = self.lr)
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.n_threads,
            drop_last=True,
            shuffle=True
        )
        keep_training = True
        epoch = 0
        i = self.start_iter
        print("starting training")
        s_time = time.time()
        while keep_training:
            epoch += 1
            print("epoch: {:d}".format(epoch))
            for items in train_loader:
                i += self.opt.batch_size
                gt_images, _, _, masks = self.cuda(*items)
                # masks = torch.cat([masks]*3, dim = 1)
                masked_images = gt_images * masks
             #   masks = torch.cat([masks], dim = 1)
                self.forward(masked_images, masks, gt_images)
                self.update_parameters()

                if i % self.opt.log_interval == 0:

                    e_time = time.time()
                    int_time = e_time - s_time
                    masked_images = masked_images.cpu()
                    fake_images = self.fake_B.cpu()
                    images = torch.cat([masked_images[0:3], fake_images[0:3]], dim=0)
                    writer.add_image("imgs", images, i)
                    print("epoch:{:d}, iteration:{:d}".format(epoch, i), ", l1_loss:", self.l1_loss*self.opt.batch_size/self.opt.log_interval, ", time_taken:", int_time)
                    writer.add_scalars("loss_val", {"l1_loss":self.l1_loss*self.opt.batch_size/self.opt.log_interval}, i)
                    s_time = time.time()
                    self.l1_loss = 0.0
                    self.D_loss = 0.0
                    
                if i % self.opt.save_interval == 0:
                    save_ckpt('{:s}/ckpt/g_{:d}.pth'.format(self.opt.save_dir, i ), [('generator', self.G)], [('optimizer_G', self.optm_G)], i )
                    print('Save to {:s}/ckpt/g_{:d}.pth'.format(self.opt.save_dir, i ))
               
                if i % self.opt.vis_interval == 0:
                    val_loader = DataLoader(
                                    dataset=self.val_dataset,
                                    batch_size=self.opt.batch_size,
                                    drop_last=True,
                                    shuffle=True
                                )
                    # self.G.eval()
                    count = 0
                    if not os.path.exists('{:s}/images/iter_{:d}'.format(self.opt.save_dir, i)):
                        os.makedirs('{:s}/images/iter_{:d}'.format(self.opt.save_dir, i))
                    for items in val_loader:
                        
                        gt_images, _, gt_edges, masks = self.cuda(*items)
                        masked_images = gt_images * masks
                        masks = torch.cat([masks]*3, dim = 1)
                        fake_B, mask = self.G(masked_images, masks)

                        fake_B = fake_B.cpu()
                        masks = masks.cpu()
                        gt_images = gt_images.cpu()
                        
                        masked_imaged = gt_images * masks
                        comp_B = fake_B * (1 -  masks) + gt_images * masks

                        
                        for k in range(comp_B.size(0)):
                            count += 1
                            file_path = '{:s}/images/iter_{:d}/img_{:d}.jpg'.format(self.opt.save_dir, i, count)
                            grid = make_grid(torch.cat([gt_images[k:k+1], masked_imaged[k:k+1], fake_B[k:k+1], comp_B[k:k+1]], dim=0))
                            save_image(grid, file_path)
                    val_loader = None
                    self.G.train()
        writer.close()
        
    def test(self):
        test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=6
                )
        # self.G.eval()
        count = 0
        for items in test_loader:
        
            gt_images, _, gt_edges, masks = self.cuda(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks]*3, dim = 1)
            fake_B, mask = self.G(masked_images, masks)

            fake_B = fake_B.cpu()
            masks = masks.cpu()
            gt_images = gt_images.cpu()
            comp_B = fake_B * (1 - masks) + gt_images * masks

            start_iter = self.opt.start_iter

            if not os.path.exists('{:s}/images/result_final_{:d}'.format(self.opt.save_dir,self.opt.start_iter)):
                os.makedirs('{:s}/images/result_final_{:d}'.format(self.opt.save_dir,self.opt.start_iter))
            for k in range(comp_B.size(0)):
                count += 1
                grid = make_grid(comp_B[k:k+1])
                file_path = '{:s}/images/result_final_{:d}/img_{:d}.png'.format(self.opt.save_dir,self.opt.start_iter, count)
                save_image(grid, file_path)
                
                grid = make_grid(masked_images[k:k+1])
                file_path = '{:s}/images/result_final_{:d}/masked_img_{:d}.png'.format(self.opt.save_dir,self.opt.start_iter, count)
                save_image(grid, file_path)
    
    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
    
    def update_parameters(self):
        self.updateG()
        self.updateD()
    
    def updateG(self):
        self.optm_G.zero_grad()
        ##calculate the loss of G
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.calculate_TV_loss(comp_B * (1 - self.mask))
        style_loss = self.calculate_style_loss(real_B_feats, fake_B_feats) + self.calculate_style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.calculate_preceptual_loss(real_B_feats, fake_B_feats) + self.calculate_preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = torch.mean(torch.abs(real_B - fake_B)* self.mask)
        hole_loss = torch.mean(torch.abs(real_B - fake_B) * (1 - self.mask))
        
        pred_fake = self.D(fake_B)
        adv_loss_G = self.adv_loss(pred_fake, True, False)
        
        loss_G = (  tv_loss * self.opt.lambda_tv
                  + style_loss * self.opt.lambda_style
                  + preceptual_loss * self.opt.lambda_preceptual
                  + valid_loss * self.opt.lambda_valid
                  + hole_loss * self.opt.lambda_hole
                  + adv_loss_G * self.opt.lambda_adv)
        self.l1_loss += (hole_loss + valid_loss).cpu().detach().numpy()
        loss_G.backward()
        self.optm_G.step()
    ### Added
    def updateD(self):
        self.optm_D.zero_grad()


        real_edge = self.real_B
        fake_edge = self.fake_B
        loss_D = 0


        real_edge = real_edge.detach()
        fake_edge = fake_edge.detach()
    
        pred_real = self.D(real_edge)
        pred_fake = self.D(fake_edge)
            
        loss_D += (self.adv_loss(pred_real, True, True)  + self.adv_loss(pred_fake, False, True))/2

        loss_D.sum().backward()
        self.optm_D.step()
        self.D_loss += loss_D.cpu().detach().numpy()
    
    def l1_losses(f1, f2, contain_l1 = True):
        return torch.mean(torch.abs(f1 - f2))
    
    def calculate_style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    
    def calculate_TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    
    def calculate_preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def cuda(self, *args):
        return (item.to(self.device) for item in args)
            