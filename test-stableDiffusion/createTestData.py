'''
프롬프트 테스트할 이미지를 생성하는 파일
인풋 디렉토리: jm-data/train & val
아웃풋 디렉토리: jm-test-data/train & val

'''
import os
from PIL import Image

in1 = 'test-stableDiffusion/jm-data/train'
in2 = 'test-stableDiffusion/jm-data/val'

out1 = 'test-stableDiffusion/jm-test-data/train'
out2 = 'test-stableDiffusion/jm-test-data/val'


def save1(input, class_name):
    for i, img in enumerate(os.listdir(input)):
        if i>70:
            break
        print(i)
        os.makedirs(f'{out1}/{class_name}', exist_ok=True)
        image = Image.open(f'{input}/{img}')
        # image.show()
        
        image.save(f'{out1}/{class_name}/{img}', 'JPEG')
def save2(input, class_name):
    for i, img in enumerate(os.listdir(input)):
        if i>30:
            break
        print(i)
        os.makedirs(f'{out2}/{class_name}', exist_ok=True)
        image = Image.open(f'{input}/{img}')
        # image.show()
        
        image.save(f'{out2}/{class_name}/{img}', 'JPEG')

for class_name in ['fire', 'no_fire']:
    save1(in1+'/'+class_name, class_name)
    save2(in2+'/'+class_name, class_name)
