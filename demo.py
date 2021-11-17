import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import pdb
#import yaml 
import numpy as np
from torch.utils.data import DataLoader
import os
import librosa
import pickle
from pathlib import Path
from PANNs import ResNet38, MobileNetV2
from fintune import finetunePANNs, Identity

class ModelTester:
    def __init__(self, boar_model, test_loader, ckpt, device):
        self.device = torch.device('cuda:{}'.format(device))
        self.boar_model = boar_model
        self.ckpt = ckpt 

        print("Start to Load Boar model")
        checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
        #pdb.set_trace()
        boar_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        #self.boar_model = boar_model.cuda()
        self.boar_model.eval()
        print("Done with Loading Boar model")
        
        self.test_loader = test_loader
        
    def test(self):

        batch_size = len(self.test_loader)
        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):
                inputs, audio_path = batch
                try:
                    B, T = inputs.size()  
                    print('Inference Results: {}'.format(audio_path[0].split('/')[-1]))
                except:
                    print('No Audio Signal Detected in {}'.format(audio_path[0].split('/')[-1]))
                    continue

                #inputs = inputs.cuda()
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                
                self.boar_model.eval() 
                bo_outputs = self.boar_model(inputs)
                #bo_outputs = bo_outputs['clipwise_output']
                if bo_outputs.max(1)[1].mode()[0].item() == 0:
                    print('Boar')
                else:
                    print('Not-Boar')

                       
class Data_Reader(Dataset):
    def __init__(self, datalist):
        super(Data_Reader, self).__init__()
        self.datalist = datalist
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'
        self.nan_cnt = 0
    
    def new_data(self,datalist):
        self.datalist = datalist

    def __len__(self):
        #print('Number of wav data to inference: {}'.format(len(self.datalist)))
        return len(self.datalist)

    def LogMelExtractor(self, sig):
        def logmel(sig):
            D = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
            SS = librosa.feature.melspectrogram(S=D,sr=16000)
            S = librosa.amplitude_to_db(SS)
            #S[:10, :] = 0.0
            return S

        def transform(audio):
            feature_logmel = logmel(audio)
            return feature_logmel
        
        return transform(sig)

    def __getitem__(self, idx):
        audio_path = self.datalist[idx]
        try:
            audio, _ = librosa.load(audio_path, sr=32000, dtype=np.float32)
        except:
            self.nan_cnt +=1
            print('No Audio Signal Detected in {}'.format(audio_path.split('/')[-1]))
            return torch.zeros(1,1,1), audio_path.split('/')[-1]
        
        if audio.sum() == 0.0:
            self.nan_cnt +=1
            print('No Audio Signal Detected in {}'.format(audio_path.split('/')[-1]))
            return torch.zeros(1,1,1), audio_path.split('/')[-1]

        else:
            #feature = self.LogMelExtractor(audio)
            #return torch.FloatTensor(feature).transpose(0,1), np.array([class_num])
            return audio, audio_path.split('/')[-1]


class Inference:
    
    def __init__(self, args):
        
        '''1. Dataset Preparation'''
        self.test_data_path = './data'
        test_list = []
        #print(test_list)
        self.test_dataset = Data_Reader(test_list)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=0)

        '''2. Model'''
        n_classes = 2
        boar_PANNs = MobileNetV2(32000, 1024, 1024//4,64, 0.0, None, n_classes)
        boar_model = finetunePANNs(boar_PANNs, n_classes)
    
        '''3. Tester'''
        self.tester = ModelTester(boar_model, test_loader, './model/mobileNetV2-boar-61.pt', 0)
        print('========  Ready for SED Inference  =======')
    
    def infer(self):
        while 1:
            test_list = [os.path.join(self.test_data_path,f) for f in os.listdir(self.test_data_path) if f.split('.')[-1] == 'wav']
            if len(test_list) >= 1:
                test_list = sorted(test_list)
                print(test_list)
                self.test_dataset.new_data(test_list)
                self.tester.test()
                sys.exit("======= Inference Finished =======")

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-d', '--dataset', type=str, default='config', help='configuration file')
    args = parser.parse_args()

    #'''Load Config'''
    #with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
    #    config = yaml.load(f,Loader=yaml.FullLoader)

    '''Inference Mode'''
    a = Inference(args)
    a.infer()
