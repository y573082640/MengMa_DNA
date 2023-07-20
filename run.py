__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"


from evaluation import EvaluationPipeline

from coder import Coder


if __name__ == "__main__":
    # "0001" is provided by the competition management party.
    # please see "record.txt" for the process and score records for details.
    coder = Coder(team_id="0001")
    # pipeline = EvaluationPipeline(coder=coder, error_free=True)
    # pipeline(input_image_path='United Nations Flag.bmp', output_image_path="error_free/obtained.bmp",
    #          source_dna_path="error_free/o.fasta", target_dna_path="error_free/p.fasta", random_seed=2023)

    pipeline = EvaluationPipeline(coder=coder, error_free=False)
    pipeline(input_image_path="downsampled_image.bmp", output_image_path="error/obtained.bmp",
             source_dna_path="error/o.fasta", target_dna_path="error/p.fasta", random_seed=2023)
