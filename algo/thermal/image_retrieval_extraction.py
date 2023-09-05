from algo.common.baseAlgo import BaseAlgo, BaseAlgoParams
from typing import List
from os.path import isfile, join
from os import listdir
from utils.irutils import ToRGBTensor, NetVLAD, load_state_dict, embed_keyframes, find_pair_indexes
from utils.mathutils import corr2
from utils.datautils import make_path_compatible
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import csv
import logging


class ImageRetrievalExtractionParams(BaseAlgoParams):
    def __init__(self):
        super(ImageRetrievalExtractionParams, self).__init__()
        self.image_retrieval_model_path = None
        self.nv_corr_score = 0.25
        self.inputImagePath = None
        self.keyframes_file = None
        self.outputPath = None
        self.save_embedding_file = False
        self.save_file_name = None

    def serialize(self):
        super(ImageRetrievalExtractionParams, self).serialize()
        dict.__init__(self, ImageRetrievalExtractionParams={"image_retrieval_model_path":self.image_retrieval_model_path, "nv_corr_score":self.nv_corr_score, "inputImagePath":self.inputImagePath, "keyframes_file":self.keyframes_file, "outputPath":self.outputPath, "save_embedding_file":self.save_embedding_file, "save_file_name":self.save_file_name})

    def deserialize(self, data):
        super(ImageRetrievalExtractionParams, self).deserialize(data)
        self.image_retrieval_model_path = data["AlgoParams"]["ImageRetrievalExtractionParams"]["image_retrieval_model_path"]
        self.nv_corr_score = data["AlgoParams"]["ImageRetrievalExtractionParams"]["nv_corr_score"]
        self.inputImagePath = data["AlgoParams"]["ImageRetrievalExtractionParams"]["inputImagePath"]
        self.keyframes_file = data["AlgoParams"]["ImageRetrievalExtractionParams"]["keyframes_file"]
        self.outputPath = data["AlgoParams"]["ImageRetrievalExtractionParams"]["outputPath"]
        self.save_embedding_file = data["AlgoParams"]["ImageRetrievalExtractionParams"]["save_embedding_file"]
        self.save_file_name = data["AlgoParams"]["ImageRetrievalExtractionParams"]["save_file_name"]
        dict.__init__(self, image_retrieval_model_path=self.image_retrieval_model_path, nv_corr_score=self.nv_corr_score, inputImagePath=self.inputImagePath, keyframes_file=self.keyframes_file, outputPath=self.outputPath, save_embedding_file=self.save_embedding_file, save_file_name=self.save_file_name)


class ImageRetrievalExtraction(BaseAlgo):
    def __init__(self, AlgoParams: object=None) -> object:
        super(ImageRetrievalExtraction, self).__init__()
        self.name = "ImageRetrievalExtraction"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input = None):
        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting extracting NetVLAD descriptors")
            if self.inputData != 'None':
                ir_input = input[self.inputData]
            if self.AlgoParams.keyframes_file != 'None':
                df = pd.read_csv(self.AlgoParams.keyframes_file)
                imgs_paths = [join(self.AlgoParams.inputImagePath, make_path_compatible(f)) for f in df['filename'].values]
            else:
                imgs_paths = sorted([join(self.AlgoParams.inputImagePath, make_path_compatible(f)) for f in listdir(self.AlgoParams.inputImagePath) if
                          isfile(join(self.AlgoParams.inputImagePath, f))])
            # Initialize the data transformation
            augmentations = []
            img_size = None
            if img_size is None:
                resize_transform = []
            else:
                resize_transform = [transforms.Resize((img_size, img_size))]
            transforms_list = [transforms.ToPILImage(), *resize_transform, *augmentations, ToRGBTensor(),
                               transforms.Normalize(
                                   mean=[123.68, 116.779, 103.939], std=[1.0, 1.0, 1.0])]

            # Create the model for image embedding
            netvlad_transform = transforms.Compose(transforms_list)
            netvlad = NetVLAD()
            # Load the model for image embedding
            load_state_dict(netvlad, self.AlgoParams.image_retrieval_model_path)

            # Embed the keyframes
            kf_embedding, kf_embedding_T = embed_keyframes(netvlad_transform, self.device, netvlad, imgs_paths)

            # Save the Embeddings
            if self.AlgoParams.save_embedding_file and self.AlgoParams.save_file_name:
                torch.save([kf_embedding_T, netvlad_transform], self.AlgoParams.save_file_name + "_netvlad_embedding.pth")
                cols = ["path"] + ["vi_{}".format(i) for i in range(4096)]
                df_emd = pd.DataFrame(
                    np.concatenate((np.array(imgs_paths).reshape(-1, 1),kf_embedding), axis=1),columns=cols)
                df_emd.to_csv(self.AlgoParams.save_file_name + "_netvlad_embedding.csv")

            # Calculate score matrix
            scores = np.zeros((len(kf_embedding), len(kf_embedding)))
            for i in range(len(kf_embedding)):
                for j in range(len(kf_embedding)):
                    scores[i, j] = corr2(kf_embedding[i], kf_embedding[j])

            # keep on same convention
            nv_data_mat = {"scores": scores}

            # Find the required pairs by using NV scores
            pairs: List[None] = find_pair_indexes(nv_data_mat['scores'], self.AlgoParams.nv_corr_score)

            # Save into output dir
            with open(join(self.AlgoParams.outputPath, "NV_pairs" + ".csv"), 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(pairs)
            logging.info("Finished extracting NetVLAD descriptors")
        else:
            output = loaded

        image_retrival = {'imgs_paths': imgs_paths, 'kf_embedding': kf_embedding, 'scores': scores, 'pairs': pairs}

        if self.inputData == 'None':
            output = image_retrival
            self.save_to_file(output)
        else:
            output: None = image_retrival.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(ImageRetrievalExtraction, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, ImageRetrievalExtraction={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(ImageRetrievalExtraction, self).deserialize(data)
        self.name = data["ImageRetrievalExtraction"]["name"]
        self.AlgoParams = ImageRetrievalExtractionParams()
        self.AlgoParams.deserialize(data)
        dict.__init__(self, name=self.name, ImageRetrievalExtractionParams=self.AlgoParams)
