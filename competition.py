__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"

from argparse import ArgumentParser
from cv2 import imread
from hashlib import md5
from numpy import zeros, mean
from random import seed, shuffle, random, randint, choice
from skimage.metrics import structural_similarity
from os import listdir, path, makedirs, remove
from shutil import rmtree
from zipfile import ZipFile, ZIP_DEFLATED
import os

from team1011 import Coder
from evaluation import DefaultCoder


class CompetitionPipeline:

    def __init__(self, coder, repeat: int, random_seed: int):
        """
        Initialize the competition pipeline.

        :param coder: image-dna coder provided by the participant.

        :param repeat: number of repetitions required for the same image.
        :type repeat: int

        :param random_seed: random seed of the pipeline.
        :type random_seed: int
        """
        if not isinstance(coder, DefaultCoder):
            raise ValueError("The coder needs to inherit from DefaultCoder.")

        self.coder, self.repeat = coder, repeat
        self.random_seed = random_seed

    def __call__(self, round_index, image_folder_path: str, cache_folder_path: str, record_folder_path: str):
        """
        Execute the competition process (large-scale).

        :param round_index: round index.
        :type round_index: int

        :param image_folder_path: folder path for the test images.
        :type image_folder_path: str

        :param cache_folder_path: temporary folder path (used to save the generated files for the evaluation phase).
        :type cache_folder_path: str

        :param record_folder_path: folder of the score record.
        :type record_folder_path: str
        """
        image_paths = [image_folder_path + child_path for child_path in listdir(image_folder_path)]

        temp_path = cache_folder_path + self.coder.team_id + "/"
        print('temp path:', temp_path)
        if not path.exists(temp_path):
            makedirs(temp_path)

        # execute the evaluation tasks.
        source_dna_paths = self.task_1(image_paths=image_paths, temp_folder_path=temp_path)
        merged_dna_paths = self.task_2(dna_paths=source_dna_paths, temp_folder_path=temp_path)
        records = self.task_3(image_paths=image_paths, dna_paths=merged_dna_paths, temp_folder_path=temp_path)

        if not path.exists(record_folder_path):
            makedirs(record_folder_path)

        # output the score records.
        with open(record_folder_path + self.coder.team_id +
                  "[" + str(round_index) + "." + str(self.random_seed) + "].csv", "w") as file:
            file.write("image id,repeat id,density score,compatibility score,recovery score,total score\n")
            for record in records:
                index_1, index_2 = record[0], record[1]
                score_1, score_2, score_3 = record[2], record[3], record[4]
                total_score = 0.2 * score_1 + 0.3 * score_2 + 0.5 * score_3
                file.write("%d,%d,%.3f,%.3f,%.3f,%.3f\n" % (index_1, index_2,
                                                            score_1 * 100, score_2 * 100, score_3 * 100,
                                                            total_score * 100))

        rmtree(temp_path)

    def task_1(self, image_paths: list, temp_folder_path: str) -> list:
        """
        Complete the large-scale encoding process based on the inputted image-dna coder.

        :param image_paths: all image paths for the competition.
        :type image_paths: list

        :param temp_folder_path: temporary path folder.
        :type temp_folder_path: str

        :return: all paths to the FASTA file for DNA sequences.
        :rtype: list
        """
        dna_paths = []
        for image_index, image_path in enumerate(image_paths):
            for repeat_index in range(self.repeat):
                # obtain DNA sequences from the coder.
                dna_sequences = self.coder.image_to_dna(input_image_path=image_path, need_logs=False)

                # check the correctness of DNA sequences.
                for dna_sequence in dna_sequences:
                    if len(dna_sequence) < 100 or len(dna_sequence) > 200:
                        raise ValueError("The total length of every DNA sequence needs to be between 100nt and 200nt.")

                    valid_length = 0
                    for base in ["A", "C", "G", "T"]:
                        valid_length += dna_sequence.count(base)

                    if len(dna_sequence) != valid_length:
                        raise ValueError("There are illegal characters in the sequence, "
                                         "which do not belong to any of A/C/G/T.")

                # write to the FASTA file.
                number = len(dna_sequences)
                dna_path = temp_folder_path + str(image_index) + "." + str(repeat_index) + ".txt"
                with open(dna_path, "w") as file:
                    for dna_index, dna_sequence in enumerate(dna_sequences):
                        file.write(">" + str(dna_index).zfill(len(str(number))) + "_" + str(number) + "\n")
                        file.write(dna_sequence + "\n")

                # save by compression.
                zip_file = ZipFile(dna_path + ".zip", "w")
                print()
                zip_file.write(dna_path, arcname=dna_path.replace("/home/hit/", ""), compress_type=ZIP_DEFLATED)
                zip_file.close()

                remove(dna_path)

                # record the file path and its attributes.
                dna_paths.append([image_index, repeat_index, dna_path])

        return dna_paths

    def task_2(self, dna_paths: list, temp_folder_path: str) -> list:
        """
        Conduct wet experiment simulation to obtain DNA sequence FASTA files with errors and disordered sequences.

        :param dna_paths: paths (and their corresponding attributes) of the original FASTA file for all DNA sequences.
        :type dna_paths: list

        :param temp_folder_path: temporary path folder.
        :type temp_folder_path: str

        :return: corresponding to all FASTA file paths of DNA sequences with errors.
        :rtype: list
        """
        seed(self.random_seed)

        practical_dna_paths = []
        for image_index, repeat_index, source_dna_path in dna_paths:
            # load by decompression.
            print('----------------------------------unzip file from:', source_dna_path+'.zip')

            temp_fold = source_dna_path[:-len(source_dna_path.split('/')[-1])]
            print('----------------------------------unzip file to  :', temp_fold)
            with ZipFile(source_dna_path+'.zip', 'r') as zip:
                for zip_info in zip.infolist():
                    if zip_info.filename[-1]=='/':
                        continue
                    zip_info.filename=os.path.basename(zip_info.filename)
                    # print(zip_info)
                    zip.extract(zip_info, temp_fold)
            # zip_file = ZipFile(source_dna_path + ".zip", "r")
            # zip_file.extract(zip_file.namelist()[0])
            # zip_file.extractall('/home/testjs1/test/temp/0004/')
            # zip_file.close()

            dna_sequences = []
            with open(source_dna_path, "r") as file:
                for line in file.readlines():
                    if line[0] != ">":
                        source_dna_sequence = line[:-1]
                        # introduce 3% of edit errors in DNA sequences,
                        # including 1.5% mutations, 0.75% insertions, and 0.75% deletions.
                        mutate_number = int(0.0150 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)
                        insert_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)
                        delete_number = int(0.0075 * len(source_dna_sequence)) + (0 if random() > 0.5 else 1)

                        target_dna_sequence = list(source_dna_sequence)
                        while True:
                            for _ in range(mutate_number):
                                location = randint(0, len(target_dna_sequence) - 1)
                                source = target_dna_sequence[location]
                                target = choice(list(filter(lambda base: base != source, ["A", "C", "G", "T"])))
                                target_dna_sequence[location] = target

                            for _ in range(insert_number):
                                location = randint(0, len(target_dna_sequence))
                                target_dna_sequence.insert(location, choice(["A", "C", "G", "T"]))

                            for _ in range(delete_number):
                                location = randint(0, len(target_dna_sequence) - 1)
                                del target_dna_sequence[location]

                            if "".join(target_dna_sequence) != source_dna_sequence:
                                target_dna_sequence = "".join(target_dna_sequence)
                                break

                            target_dna_sequence = list(source_dna_sequence)

                        dna_sequences.append(target_dna_sequence)

            remove(source_dna_path)

            # shuffle the obtained DNA sequences.
            shuffle(dna_sequences)

            # write to the FASTA file.
            number = len(dna_sequences)
            target_dna_path = temp_folder_path + str(image_index) + "." + str(repeat_index) + ".p.txt"
            with open(target_dna_path, "w") as file:
                for dna_index, dna_sequence in enumerate(dna_sequences):
                    file.write(">" + str(dna_index).zfill(len(str(number))) + "_" + str(number) + "\n")
                    file.write(dna_sequence + "\n")

            # save by compression.
            zip_file = ZipFile(target_dna_path + ".zip", "w")
            zip_file.write(target_dna_path, arcname=target_dna_path.replace("/home/hit/", ""),
                           compress_type=ZIP_DEFLATED)
            zip_file.close()

            remove(target_dna_path)

            practical_dna_paths.append([image_index, repeat_index, source_dna_path, target_dna_path])

        seed(None)

        return practical_dna_paths

    def task_3(self, image_paths: list, dna_paths: list, temp_folder_path: str) -> list:
        """
        Rate each task.

        :param image_paths: all image paths for the competition.
        :type image_paths: list

        :param dna_paths: paths (and attributes) of the original/shuffled FASTA file for all DNA sequences.
        :type dna_paths: list

        :param temp_folder_path: temporary path folder.
        :type temp_folder_path: str

        :return: score records.
        :rtype: list
        """
        records = []
        for image_index, repeat_index, source_dna_path, target_dna_path in dna_paths:
            # load by decompression.

            # zip_file = ZipFile(target_dna_path + ".zip", "r")
            # zip_file.extract(zip_file.namelist()[0])
            # zip_file.close()
            print('----------------------------------unzip file from:', target_dna_path+'.zip')
            temp_fold = target_dna_path[:-len(target_dna_path.split('/')[-1])]
            print('----------------------------------unzip file to  :', temp_fold)
            with ZipFile(target_dna_path+'.zip', 'r') as zip:
                for zip_info in zip.infolist():
                    if zip_info.filename[-1]=='/':
                        continue
                    zip_info.filename=os.path.basename(zip_info.filename)
                    # print(zip_info)
                    zip.extract(zip_info, temp_fold)

            # convert DNA sequences to image.
            dna_sequences = []
            with open(target_dna_path, "r") as file:
                for line in file.readlines():
                    if line[0] != ">":
                        dna_sequences.append(line[:-1])
            target_path = temp_folder_path + image_paths[image_index][image_paths[image_index].rindex("/") + 1:]
            self.coder.dna_to_image(dna_sequences=dna_sequences, output_image_path=target_path, need_logs=False)

            # remove the target dna sequences.
            remove(target_dna_path)

            # calculate the score.
            source_path = image_paths[image_index]
            scores = self.calculate_score(path_1=source_path, path_2=target_path, source_dna_path=source_dna_path)
            records.append([image_index, repeat_index, scores[0], scores[1], scores[2]])

            # remove the target image.
            remove(target_path)

        return records

    @staticmethod
    def calculate_score(path_1: str, path_2: str, source_dna_path: str):
        """
        Calculate the score for the current converting task.

        :param path_1: original image path.
        :type path_1: str

        :param path_2: ath to save the decoded image.
        :type path_2: str

        :param source_dna_path: original DNA sequence list.
        :type source_dna_path: str

        :return: density score, compatibility score, retrieval score.
        :rtype: float, float, float

        .. note::
            The total score = 20% density score + 30% compatibility score + 50% recovery score.
        """
        # load by decompression.
        zip_file = ZipFile(source_dna_path + ".zip", "r")
        zip_file.extract(zip_file.namelist()[0])
        zip_file.close()

        # load original DNA sequences from the FASTA file.
        dna_sequences = []
        with open(source_dna_path, "r") as file:
            for line in file.readlines():
                if line[0] != ">":
                    dna_sequences.append(line[:-1])

        # remove the target dna sequences.
        remove(source_dna_path)

        # calculate the density score.
        expected_image = imread(path_1)
        b_number, d_number = expected_image.shape[0] * expected_image.shape[1] * 24, 0
        for dna_sequence in dna_sequences:
            d_number += len(dna_sequence)
        density_score = 1 - d_number / b_number if d_number < b_number else 0

        # calculate the compatibility score.
        h_statistics, gc_statistics = [], []
        for dna_sequence in dna_sequences:
            homopolymer = 1
            while True:
                found = False
                for nucleotide in ["A", "C", "G", "T"]:
                    if nucleotide * (1 + homopolymer) in dna_sequence:
                        found = True
                        break
                if found:
                    homopolymer += 1
                else:
                    break
            gc_bias = abs((dna_sequence.count("G") + dna_sequence.count("C")) / len(dna_sequence) - 0.5)

            h_statistics.append(homopolymer)
            gc_statistics.append(gc_bias)

        maximum_homopolymer, maximum_gc_bias = mean(h_statistics), mean(gc_statistics)
        h_score = (1.0 - (maximum_homopolymer - 1) / 5.0) / 2.0 if maximum_homopolymer < 6 else 0
        c_score = (1.0 - maximum_gc_bias / 0.3) / 2.0 if maximum_gc_bias < 0.3 else 0

        compatibility_score = h_score + c_score

        # calculate the recovery score.
        # noinspection PyBroadException
        try:
            obtained_image, rate = imread(path_2), 1.0
            if expected_image.shape != obtained_image.shape:
                minimum_w = min(expected_image.shape[0], obtained_image.shape[0])
                minimum_h = min(expected_image.shape[1], obtained_image.shape[1])
                expected_image = expected_image[:minimum_w, :minimum_h]
                obtained_image = obtained_image[:minimum_w, :minimum_h]
                rate = (minimum_w * minimum_h) / (expected_image.shape[0] * expected_image.shape[1])
            ssim_value = structural_similarity(expected_image, obtained_image, multichannel=True) * rate
            recovery_score = (ssim_value - 0.84) / 0.16 if ssim_value > 0.84 else 0

        except AssertionError:
            recovery_score = 0.0  # unable to parse as image, SSIM value cannot be calculated, the recovery score is 0.

        except Exception:
            recovery_score = 0.0  # unable to parse as image, SSIM value cannot be calculated, the recovery score is 0.

        return density_score, compatibility_score, recovery_score


def generate_random_seed(coder_paths):
    """
    Create a random seed for the competition process using the MD5 values of all existing participant code scripts.

    :param coder_paths: folder where all participant codes are placed.
    :type coder_paths: list

    :return: random seed.
    :rtype: int
    """
    mapping = {"0": [0, 0, 0, 0], "1": [0, 0, 0, 1], "2": [0, 0, 1, 0], "3": [0, 0, 1, 1],
               "4": [0, 1, 0, 0], "5": [0, 1, 0, 1], "6": [0, 1, 1, 0], "7": [0, 1, 1, 1],
               "8": [1, 0, 0, 0], "9": [1, 0, 0, 1], "a": [1, 0, 1, 0], "b": [1, 0, 1, 1],
               "c": [1, 1, 0, 0], "d": [1, 1, 0, 1], "e": [1, 1, 1, 0], "f": [1, 1, 1, 1]}
    seed_values = zeros(shape=(128,), dtype=int)
    for coder_path in coder_paths:
        with open(coder_path, "rb") as file:
            file_md5 = md5(file.read()).hexdigest()
            for digit_index, digit in enumerate(file_md5):
                seed_values[digit_index * 4: (digit_index + 1) * 4] += mapping[digit]
    seed_values %= 2

    seed_value = 0
    for value in seed_values:
        seed_value += value
        seed_value *= 2
        if seed_value > 1000000:  # constrain the seed value between 0 and 1000000.
            seed_value %= 1000000

    return seed_value


def read_args():
    """
    Read arguments from the command line.

    :return: parameters.
    """
    parser = ArgumentParser()
    parser.add_argument("-r", "--round_index", required=True, type=int,
                        help="current round index (i.e. 1, 2, or 3).")
    parser.add_argument("-i", "--team_index", required=True, type=str,
                        help="team index.")
    parser.add_argument("-p", "--photo_folder", required=True, type=str,
                        help="folder path of photos.")
    parser.add_argument("-a", "--record_path", required=True, type=str,
                        help="record file path of all coder script paths.")
    parser.add_argument("-t", "--repeat_time", required=True, type=int, default=2,
                        help="repeat time per photo.")
    parser.add_argument("-c", "--cache_path", required=True, type=str,
                        help="temporary folder path.")
    parser.add_argument("-s", "--saved_path", required=True, type=str,
                        help="score saved path.")

    return parser.parse_args()


def tasks(round_index, team_index, photo_folder, record_path, repeat, cache_path, saved_path):
    with open(record_path, "r") as file:
        coder_paths = [line[:-1] for line in file.readlines()]

    random_seed = generate_random_seed(coder_paths=coder_paths)
    coder = Coder(team_id=team_index)
    pipeline = CompetitionPipeline(coder=coder, repeat=repeat, random_seed=random_seed)
    pipeline(round_index=round_index, image_folder_path=photo_folder, cache_folder_path=cache_path,
             record_folder_path=saved_path)


if __name__ == "__main__":
    params = read_args()

    print("Your parameters are:")
    print("round        = ", params.round_index)   # r
    print("team index   = ", params.team_index)    # i
    print("photo folder = ", params.photo_folder)  # p
    print("record path  = ", params.record_path)   # a
    print("repeat time  = ", params.repeat_time)   # t
    print("cache path   = ", params.cache_path)    # c
    print("saved path   = ", params.saved_path)    # s
    print()

    tasks(round_index=params.round_index, team_index=params.team_index, photo_folder=params.photo_folder,
          record_path=params.record_path, repeat=params.repeat_time,
          cache_path=params.cache_path, saved_path=params.saved_path)
