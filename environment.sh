sudo apt-get install python3-scipy
sudo apt-get install python3-sklearn
## 6. Asia, 69. Seoul 선택

sudo apt-get install llvm-7
sudo LLVM_CONFIG=/usr/bin/llvm-config-7 pip3 install llvmlite
## llvmlite 안깔리면 심볼링 링크 형성 후 설치
## cd /usr/bin && ln -s /usr/bin/llvm-config-7 llvm-config
## sudo LLVM_CONFG=/usr/bin/llvm-config pip3 install llvmlite
sudo LLVM_CONFIG=/usr/bin/llvm-config-7 pip3 install numba

## ERROR: Cannot uninstall 'joblib'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall. 와 같은 에러 발생 시
## pip3 install --ignore-installed joblib
sudo LLVM_CONFIG=/usr/bin/llvm-config-7 pip3 install librosa
pip3 install dcase_util