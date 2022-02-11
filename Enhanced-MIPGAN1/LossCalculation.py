import torch
import torch.nn as nn
from piq import MultiScaleSSIMLoss, FSIMLoss, MDSILoss, PieAPP

class Losses():
  def weighted_loss(loss, weight):
      farthest_setted = False 
      print('weight: ', weight, "----", (1-weight))
      if loss[0] > loss[1]:
        farthest_setted = True
      else:
        farthest_setted = False

      if farthest_setted:
        if weight > 0.525:
          loss = weight*loss[0] + (1-weight)*loss[1]
        else:
          loss = weight*loss[1] + (1-weight)*loss[0]
      else:
        if weight > 0.525:
          loss = weight*loss[1] + (1-weight)*loss[0]
        else:
          loss = weight*loss[0] + (1-weight)*loss[1]

      return loss

  def caluclate_loss(synth_img, images, perceptual_net, img_p, upsample2d, weights, epoch, weighted_loss):
      synth_img_t = (synth_img - torch.min(synth_img))/(torch.max(synth_img)-torch.min(synth_img)).detach()
      
      ms_ssim_loss1 = MultiScaleSSIMLoss(data_range=1., reduction='none')(images[0], synth_img_t)
      ms_ssim_loss2 = MultiScaleSSIMLoss(data_range=1., reduction='none')(images[1], synth_img_t)
      fsim_loss1 = FSIMLoss(data_range=1., reduction='none')(images[0], synth_img_t)
      fsim_loss2 = FSIMLoss(data_range=1., reduction='none')(images[1], synth_img_t)

      MSE_Loss = nn.MSELoss(reduction="mean")
      #calculate Perceptual Loss
      real1_0,real1_1,real1_2,real1_3=perceptual_net(img_p[0])
      real2_0,real2_1,real2_2,real2_3=perceptual_net(img_p[1])
      synth_p=upsample2d(synth_img) #(1,3,256,256)
      synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)
  
      perceptual_loss_1=0
      perceptual_loss_2=0
      perceptual_loss_1 = perceptual_loss_1 + MSE_Loss(synth_0,real1_0) 
      perceptual_loss_1 = perceptual_loss_1 + MSE_Loss(synth_1,real1_1)
      perceptual_loss_1 = perceptual_loss_1 + MSE_Loss(synth_2,real1_2)
      perceptual_loss_1 = perceptual_loss_1 + MSE_Loss(synth_3,real1_3)
  
      perceptual_loss_2 = perceptual_loss_2 + MSE_Loss(synth_0,real2_0) 
      perceptual_loss_2 = perceptual_loss_2 + MSE_Loss(synth_1,real2_1)
      perceptual_loss_2 = perceptual_loss_2 + MSE_Loss(synth_2,real2_2)
      perceptual_loss_2 = perceptual_loss_2 + MSE_Loss(synth_3,real2_3)


      #ms_ssim_loss = (ms_ssim_loss1+ms_ssim_loss2)
      if epoch % 20 == 0:
        print('mssim: ',ms_ssim_loss1,"----",ms_ssim_loss2)
        print('fsim: ',fsim_loss1,"----",fsim_loss2)
        print('perceptual: ',perceptual_loss_1,"----",perceptual_loss_2)

      fsim_loss = (fsim_loss1 + fsim_loss2)/2
      ms_ssim_loss = (ms_ssim_loss1 + ms_ssim_loss2)/2
      perceptual_loss = (perceptual_loss_1 + perceptual_loss_2)/2
      
      if epoch % 10 == 0:

        if weighted_loss in ['f','fp','fm','fmp']:   
          fsim_loss = weighted_loss([fsim_loss1, fsim_loss2], weights[0], epoch)
          
        if weighted_loss in ['m','mp','fm','fmp']:
          ms_ssim_loss = weighted_loss([ms_ssim_loss1, ms_ssim_loss2], weights[1], epoch)

        if weighted_loss in ['p','mp','fp','fmp']:
          perceptual_loss = weighted_loss([perceptual_loss_1, perceptual_loss_2], weights[2], epoch)

      fsim_loss = [fsim_loss, fsim_loss1, fsim_loss2]
      ms_ssim_loss = [ms_ssim_loss, ms_ssim_loss1, ms_ssim_loss2]
      perceptual_loss = [perceptual_loss, perceptual_loss_1, perceptual_loss_2]
      
      if epoch%10==0:
        print('fsim weight: ', weights[0].item(), '--**--', 'msssim weight: ', weights[1].item(), '--**--', 'perceptual weight: ', weights[2].item())

      return fsim_loss, ms_ssim_loss, perceptual_loss

  def identity_loss_calc(embedding1, embedding2):
      morph = (embedding1 + embedding2)/2.0
      identity_term1n = torch.mm(embedding1, torch.transpose(morph,0,1))
      identity_term1d = torch.norm(embedding1) * torch.norm(morph)
      identity_term2n = torch.mm(embedding2, torch.transpose(morph,0,1))
      identity_term2d = torch.norm(embedding2) * torch.norm(morph)
      identity_loss = ((1 - identity_term1n/identity_term1d) + (1 - identity_term2n/identity_term2d))/2
      identity_diff = torch.abs(((1 - identity_term1n/identity_term1d) + (1 - identity_term2n/identity_term2d)))
      return identity_loss.to('cuda:0'), identity_diff.to('cuda:0')