from model import pretrain_actor, pretrain_critic
from model import inference

'''
pretrain_actor('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/', 290)

pretrain_critic('/nobackup/anakuzne/data/voicebank-demand/clean_trainset_28spk_wav/', 
                '/nobackup/anakuzne/data/voicebank-demand/noisy_trainset_28spk_wav/',
                '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_critic/', 200)
'''
inference('/nobackup/anakuzne/data/voicebank-demand/clean_testset_wav/',
          '/nobackup/anakuzne/data/voicebank-demand/noisy_testset_wav/', 
          '/nobackup/anakuzne/data/experiments/speech_enhancement/2020/pre_actor/actor_best.pth')

