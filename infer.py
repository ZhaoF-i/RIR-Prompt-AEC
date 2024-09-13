import torch
## ICCRN
# from Network.ICCRN_RIR_Prompt_D_AEC import NET as AEC
## MTFAA
from Network.mtfaa_origen import MTFAANet as AEC
from Network.ICCRN_1layer import NET as SE
import soundfile as sf

# SE_model = '/ddnstor/imu_zhaofei/result/rir_prompt/ICCRN_1layer_estCleanRIR_mask/checkpoints/best.ckpt'
SE_model_save = '/ddnstor/imu_zhaofei/result/rir_prompt/ICCRN_1layer_estCleanRIR_mask/checkpoints/SE.ckpt'
SE_NET = SE()
# SE_NET.load_state_dict(torch.load(SE_model).state_dict)
# torch.save(SE_NET.state_dict(), SE_model_save)

AEC_model = '/ddnstor/imu_zhaofei/result/rir_prompt/ReDesignDataSet/MTFAA_se-rir-prompt-20_conv-far-inTime_withMaskModel/checkpoints/best.ckpt'
AEC_model_save = '/ddnstor/imu_zhaofei/result/rir_prompt/ReDesignDataSet/MTFAA_se-rir-prompt-20_conv-far-inTime_withMaskModel/checkpoints/MTFAA_AEC.ckpt'
## ICCRN
#AEC_NET = AEC()
# MTFAA
AEC_NET = AEC(n_sig=3)
AEC_NET.load_state_dict(torch.load(AEC_model).state_dict)
torch.save(AEC_NET.state_dict(), AEC_model_save)


SE_NET.load_state_dict(torch.load(SE_model_save))
AEC_NET.load_state_dict(torch.load(AEC_model_save))

mic, _ = sf.read('')
farend, _ = sf.read('')
rir = ...

mic = torch.Tensor(mic).unsqueeze(dim=0)
farend = torch.Tensor(farend).unsqueeze(dim=0)

with torch.no_grad():
    # mic, farend, rir : shape : [B, T]
    SE_rir_prompt = SE_NET(rir)
    new_far = torch.cat([torch.zeros(farend.shape[0], SE_rir_prompt[:, :3200].shape[-1] - 1).to('cuda'), farend],
                        dim=-1)
    echo_prompt = torch.nn.functional.conv1d(new_far.unsqueeze(dim=0),
                                             torch.flip(SE_rir_prompt[:, :3200], dims=[1]).unsqueeze(dim=1),
                                             groups=farend.shape[0])[0]

    ## ICCRN
    # est_time = AEC_NET(torch.stack([mic, farend, echo_prompt], dim=1))
    ## MTFAA
    mag, cspec, est_time = AEC_NET([mic, farend, echo_prompt])

print(est_time.shape)