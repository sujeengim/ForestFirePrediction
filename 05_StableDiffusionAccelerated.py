#진짜 이미지로 프롬프트 주어 가짜 이미지 생성하기 

import os
import warnings

warnings.filterwarnings("ignore")

import random
import requests
import torch
# import intel_extension_for_pytorch as ipex
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline


import torch.nn as nn
import time
from typing import List, Dict, Tuple


class Img2ImgModel:
    """
    This class creates a model for transforming images based on given prompts.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        optimize: bool = True,
    ) -> None:
        """
        Initialize the model with the specified parameters.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            device (str, optional): The device to run the model on. Defaults to "xpu".
            torch_dtype (torch.dtype, optional): The data type to use for the model. Defaults to torch.float16.
            optimize (bool, optional): Whether to optimize the model. Defaults to True.
        """
        self.device = device
        self.pipeline = self._load_pipeline(model_id_or_path, torch_dtype)
        
        
        '''
        if optimize:
            start_time = time.time()
            print("Optimizing the model...")
            self.optimize_pipeline()
            print(
                "Optimization completed in {:.2f} seconds.".format(
                    time.time() - start_time
                )
            )
        '''

    def _load_pipeline(
        self, model_id_or_path: str, torch_dtype: torch.dtype
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Load the pipeline for the model.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            torch_dtype (torch.dtype): The data type to use for the model.

        Returns:
            StableDiffusionImg2ImgPipeline: The loaded pipeline.
        """
        print("Loading the model...")
        # 🟥🟥🟥🟥🟥stable diffusion v1-5pipeline 설정
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path, torch_dtype=torch_dtype
        )
        # pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline = pipeline.to(self.device)
        print("Model loaded.")
        return pipeline
    '''
    def _optimize_pipeline(
        self, pipeline: StableDiffusionImg2ImgPipeline
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Optimize the pipeline of the model.

        Args:
            pipeline (StableDiffusionImg2ImgPipeline): The pipeline to optimize.

        Returns:
            StableDiffusionImg2ImgPipeline: The optimized pipeline.
        """
        for attr in dir(pipeline):
            if isinstance(getattr(pipeline, attr), nn.Module):
                setattr(
                    pipeline,
                    attr,
                    ipex.optimize(
                        getattr(pipeline, attr).eval(),
                        dtype=pipeline.text_encoder.dtype,
                        inplace=True,
                    ),
                 )
    
    
    

    def optimize_pipeline(self) -> None:
        """
        Optimize the pipeline of the model.
        """
        self.pipeline = self._optimize_pipeline(self.pipeline)
    '''
    
    def get_image_from_url(self, url: str, path: str) -> Image.Image:
        """
        Get an image from a URL or from a local path if it exists.

        Args:
            url (str): The URL of the image.
            path (str): The local path of the image.

        Returns:
            Image.Image: The loaded image.
        """
        if os.path.exists(path):
            img = Image.open(path).convert("RGB") #이미 존재하는 이미지면 리사이즈만 하고 리턴
        else: #존재하지 않으면 저장 후 리사이즈&리턴
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to download image. Status code: {response.status_code}"
                )
            if not response.headers["content-type"].startswith("image"):
                raise Exception(
                    f"URL does not point to an image. Content type: {response.headers['content-type']}"
                )
            img = Image.open(BytesIO(response.content)).convert("RGB")
            #이미지 저장 🔶🔶🔶🔶🔶
            img.save(path) 
        img = img.resize((768, 512))
        return img

    @staticmethod
    def random_sublist(lst):
        sublist = []
        for _ in range(random.randint(1, len(lst))):
            item = random.choice(lst)
            sublist.append(item)
        return sublist

    def generate_images(
        self,
        prompt: str,
        image_url: str, #처음부터 주어진 진짜 위성이미지 10개 중 하나
        class_name: str,
        seed_image_identifier: str,
        variations: List[str],
        num_images: int = 5, #만들 사진 수
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        save_path: str = "output",
        seed_path: str = "intput", 
    ) -> List[Image.Image]:
        """
        Generate images based on the provided prompt and variations.

        Args:
            prompt (str): The base prompt for the generation.
            image_url (str): The URL of the seed image.
            class_name (str): The class of the image (e.g. "fire" or "no_fire").
            seed_image_identifier (str): The identifier of the seed image.
            variations (List[str]): The list of variations to apply to the prompt.
            num_images (int, optional): The number of images to generate. Defaults to 5.
            strength (float, optional): The strength of the transformation. Defaults to 0.75.
            guidance_scale (float, optional): The scale of the guidance. Defaults to 7.5.
            save_path (str, optional): The path to save the generated images. Defaults to "output".
            seed_path (str, optional): The path to save the input images. Defaults to "input".

        Returns:
            List[Image.Image]: The list of generated images.
        """
        input_image_path = f"{seed_path}/{seed_image_identifier}.png" #input/확장자 없는 이미지 파일 이름 
        init_image = self.get_image_from_url(image_url, input_image_path) #이미지 리사이즈 후 저장
        images = []
        for i in range(num_images): #5번 반복, 한 이미지에 대해 5가지vari 적용 => 결과 fire25개 nofire25개
            variation = variations[i % len(variations)]
            final_prompt = f"{prompt} {variation}"
            image = self.pipeline(
                prompt=final_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
            ).images
            output_image_path = os.path.join(#"output/fire or nofire"+확장자없는 파일이름~~.png
                save_path, 
                f"{seed_image_identifier}_{'_'.join(variation.split())}_{i}.png",
            )
            #이미지 저장🔶🔶🔶🔶🔶
            image[0].save(output_image_path)
            images.append(image)
        return images


if __name__ == "__main__":
    
    # model_id = "runwayml/stable-diffusion-v1-5"
    model_id = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    base_prompt = (
        # "A close image to this original satellite image with slight change in location"
        'A close image to this image with slight change'
    )
    fire_variations = [#어둡고 연기 6가지
        "early morning with a wild fire",
        "late afternoon",
        "mid-day",
        "night with wild fire",
        "smoky conditions",
        "visible fire lines",
    ]
    no_fire_variations = [#밝고 깨끗 7가지
        "early morning with clear skies",
        "no signs of fire",
        "night",
        "late afternoon with clear skies",
        "mid-day with clear skies",
        "with dense vegetation",
        "with sparse vegetation",
    ]


    image_urls = { #test image
        'fire' : [ #sight
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/afternoonCloudSky.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/midCloudSky.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/field.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/sea.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/mapinskhu.png?raw=true',
        ],
        'nofire' : [ #object
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/doll.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/face.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/frenchToast.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/stake.jpg?raw=true',
            'https://github.com/sujeengim/ForestFirePrediction/blob/main/testImage/life4cut.jpg?raw=true',

        ]
    }

    '''
    image_urls = { #미국정부출처에서 10개 이미지 가져왔다는게 이건가봄 real 근데 넘 가짜같이 생김
        "fire": [
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_3912105_sw_10_h_20160713.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_3912113_sw_10_h_20160713.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_3912114_se_10_h_20160806.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_3912120_ne_10_h_20160713.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/Fire/m_4012355_se_10_h_20160713.png?raw=true",
        ],
        "no_fire": [
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/NoFire/m_3912045_ne_10_h_20160712.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/NoFire/m_3912057_sw_10_h_20160711.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/NoFire/m_3912142_sw_10_h_20160711.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/NoFire/m_3912343_se_10_h_20160529.png?raw=true",
            "https://github.com/intelsoftware/ForestFirePrediction/blob/main/data/real_USGS_NAIP/train/NoFire/m_4012241_se_10_h_20160712.png?raw=true",
        ],
    }
    '''
    # model = Img2ImgModel(model_id, device="xpu")
    model = Img2ImgModel(model_id_or_path=model_id, device="cuda")
    num_images = 3
    gen_img_count = 0

    try:
        start_time = time.time() #시간측정
        for class_name, urls in image_urls.items(): # (fire, [https://github~~, ~~, ...])
            for url in urls: #https://github~~
                seed_image_identifier = os.path.basename(url).split(".")[0] # 확장자 없는 이미지 파일 이름 
                # 😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️🩳🩳🩳🩳🩳🩳🩳changge
                input_dir = f"./testImage/input/{class_name}"
                output_dir = f"./testImage/output/{class_name}"
                # input_dir = f"./input/{class_name}" #/input/fire or nofire
                # output_dir = f"./output/{class_name}"#/output/fire or nofire
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                variations = (
                    fire_variations if class_name == "fire" else no_fire_variations
                )
                model.generate_images(
                    prompt=base_prompt,
                    image_url=url,
                    class_name=class_name,
                    seed_image_identifier=seed_image_identifier,
                    variations=variations,
                    num_images=num_images,
                    save_path=output_dir,
                    seed_path=input_dir,
                    
                )
                gen_img_count += num_images
    except KeyboardInterrupt:
        print("\nUser interrupted image generation...")
    finally:
        print(
            f"Complete generating {gen_img_count} images in {'/'.join(output_dir.split('/')[:-1])} in {time.time() - start_time:.2f} seconds."
        )
