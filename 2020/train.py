from modules import pretrain_actor, pretrain_critic
from modules import inference_actor, enhance

'''

pretrain_critic('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic/', 200)

pretrain_critic('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic_1/', 200)   


pretrain_actor('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_3/', 290)  


enhance('/nobackup/anakuzne/data/experiments/speech_enhancement/chkt/',
        '/nobackup/anakuzne/data/experiments/speech_enhancement/chkt/', 
        '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_1/actor_best.pth', 
        '/nobackup/anakuzne/data/experiments/speech_enhancement/chkt_res/') 

inference_actor('/nobackup/anakuzne/data/voicebank-demand/clean_testset_wav/',
          '/nobackup/anakuzne/data/voicebank-demand/noisy_testset_wav/', 
          '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_1/actor_best.pth', 
          '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_test/')  

''' 
 
pretrain_actor('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor_4/', 290) 

