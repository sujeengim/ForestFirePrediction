### 레포지토리 설명
- https://github.com/IntelSoftware/ForestFirePrediction
인텔 코드로 stable diffusion 공부하는 레포지토리이다.
- test-stableDiffusion 디렉토리에서 주로 작업했다. 

### test-stableDiffusion 설명
- jm-data : jm학생에게 받은 1000개 실제 위성 이미지 데이터 디렉토리

- jm-test-data : jm-data로 createTestData.py를 실행시켜 fire, nofire 각각 train 70, val 30개씩 추출하여 저장한 이미지 데이터 디렉토리. jm-data가 너무 많아서 간단한 테스트를 위해 생성함. 

- jm-test-outdata : jm-test-data로 img2imgTest.py를 실행시킨 이미지 데이터 디렉토리

- jm-test-augdata : jm-test-data + jm-test-outdata. trainTest.py에 인풋으로 들어가 정확도를 확인할 이미지 데이터 디렉토리.