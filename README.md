# Korea Electronics Technology Institute   
### Boar Sound Detection Demo on Raspberry Pi   
   
   
## Github Installation   
### 1. Install github using the command below.
#### 아래 명령어를 통해서 라즈베리파이에 깃허브를 설치해주세요.
```
$ sudo apt install git
```      

   
## Preparation    
### 1. Get the code on Raspberry Pi.
#### 라즈베리파이에 코드를 받으세요.  
```
$ git clone https://github.com/shleee47/RaspberryPi-boar-detection.git
```     
### 2. Create the conda environment.   
#### 콘다 환경을 만들어주세요.   
```
RaspberryPi-boar-detection/  
$ sh environment.sh
```      
   
### 3. Run demo.sh for demo   
#### 데모를 실행해주세요.
```
RaspberryPi-boar-detection/  
$ sh demo.sh
```            
#### 아래 문구가 생기면 모델 준비 완료.
```
====== Ready for SED Inference ======
```      
   
### 4. Move the wave files to the path below.
#### 아래 경로에 음원을 이동해주세요.
```
RaspberryPi-boar-detection/  
  └── data/
```          
   
