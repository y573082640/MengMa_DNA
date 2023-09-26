__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"

from evaluation import EvaluationPipeline

from team1011 import Coder

import os

if __name__ == "__main__":
    # "0001" is provided by the competition management party.
    # please see "record.txt" for the process and score records for details.
    coder = Coder(team_id="0001")
    # pipeline = EvaluationPipeline(coder=coder, error_free=False)
    pipeline = EvaluationPipeline(coder=coder, error_free=False)
    # pipeline(input_image_path='test_data/images_0713/15DPI_3.bmp', output_image_path="error_free/obtained.bmp",
    #          source_dna_path="error_free/o.fasta", target_dna_path="error_free/p.fasta", random_seed=59648)
    directory_path = "test_data/images_0713"
    for f in os.listdir(directory_path):
        file_name = os.path.join(directory_path, f)
        print(file_name)
        pipeline(input_image_path=file_name, output_image_path="error/obtained.bmp",
                source_dna_path="error/o.fasta", target_dna_path="error/p.fasta", random_seed=966518)
# 966518