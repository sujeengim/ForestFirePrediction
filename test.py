# gpu 테스트

import torch
cpu_tensor = torch.zeros(2,3)

device=torch.device("cuda")
gpu_tensor = cpu_tensor.to(device)
print(gpu_tensor)



# 문자열 테스트
test = "coupang"
print(test[1:])


#경로 테스트
import os
f = r'C:\Users\user\Desktop\ksj\ForestFirePrediction\data\synthetic\train\Fire\m_3912105_sw_10_h_20160713_early_morning_with_a_wild_fire_0.png'
print(os.path.basename(f))
