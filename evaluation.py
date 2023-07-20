__author__ = "Zhang, Haoling [zhanghaoling@genomics.cn]"


from copy import deepcopy
from datetime import datetime
from cv2 import imread
from random import seed, shuffle, random, randint, choice
from skimage.metrics import structural_similarity


class DefaultCoder:

    def __init__(self, team_id: str):
        """
        Initialize the image-DNA coder.

        :param team_id: team id provided by sponsor.
        :type team_id: str

        .. note::
            The competition process is automatically created.

            Thus,
            (1) Please do not add parameters other than "team_id".
                All parameters should be declared directly in this interface instead of being passed in as parameters.
                If a parameter depends on the input image, please assign its value in the "image_to_dna" interface.
            (2) Please do not divide "coder.py" into multiple script files.
                Only the script called "coder.py" will be automatically copied by
                the competition process to the competition script folder.

        """
        self.team_id = team_id
        self.monitor = Monitor()

    def image_to_dna(self, input_image_path: str, need_logs: bool = True):
        """
        Convert an image into a list of DNA sequences.

        :param input_image_path: path of the image to be encoded.
        :type input_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        :return: a list of DNA sequences.
        :rtype: list

        .. note::
            Each DNA sequence is suggested to carry its address information in the sequence list.
            Because the DNA sequence list obtained in DNA sequencing is inconsistent with the existing list.
        """
        raise NotImplementedError("The process of creating an image to a DNA sequence list is required.")

    def dna_to_image(self, dna_sequences: list, output_image_path: str, need_logs=True):
        """
        Convert a list of DNA sequences to an image.

        :param dna_sequences: a list of DNA sequences (obtained from DNA sequencing).
        :type dna_sequences: list

        :param output_image_path: path for storing image data.
        :type output_image_path: str

        :param need_logs: print process logs if required.
        :type need_logs: bool

        .. note::
           The order of the samples in this DNA sequence list input must be different from
           the order of the samples output by the "image_to_dna" interface.
        """
        raise NotImplementedError("The process of creating a DNA sequence list to an image is required.")

    def __str__(self):
        return "TEAM: " + self.team_id


class Monitor(object):

    def __init__(self):
        """
        Initialize the monitor to identify the task progress.
        """
        self.last_time = None

    def __call__(self, current_state, total_state, extra=None):
        """
        Output the current state of process.

        :param current_state: current state of process.
        :type current_state: int

        :param total_state: total state of process.
        :type total_state: int

        :param extra: extra vision information if required.
        :type extra: dict
        """
        if self.last_time is None:
            self.last_time = datetime.now()

        if current_state == 0:
            return

        position = int(current_state / total_state * 100)

        string = "|"

        for index in range(0, 100, 5):
            if position >= index:
                string += "█"
            else:
                string += " "

        string += "|"

        pass_time = (datetime.now() - self.last_time).total_seconds()
        wait_time = int(pass_time * (total_state - current_state) / current_state)

        string += " " * (3 - len(str(position))) + str(position) + "% ("

        string += " " * (len(str(total_state)) - len(str(current_state))) + str(current_state) + "/" + str(total_state)

        if current_state < total_state:
            minute, second = divmod(wait_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") wait " + "%04d:%02d:%02d" % (hour, minute, second)
        else:
            minute, second = divmod(pass_time, 60)
            hour, minute = divmod(minute, 60)
            string += ") used " + "%04d:%02d:%02d" % (hour, minute, second)

        if extra is not None:
            string += " " + str(extra).replace("\'", "").replace("{", "(").replace("}", ")") + "."
        else:
            string += "."

        print("\r" + string, end="", flush=True)

        if current_state >= total_state:
            self.last_time = None
            print()


class EvaluationPipeline:

    def __init__(self, coder, error_free: bool = True):
        """
        Initialize the evaluation pipeline.

        :param coder: image-dna coder.

        :param error_free: creating an error-free process simulation of wet experiment.
        :type error_free: bool
        """
        if not isinstance(coder, DefaultCoder):
            raise ValueError("Your coder needs to inherit from DefaultCoder.")
        self.coder = coder
        self.error_free = error_free
        self.monitor = Monitor()

    def __call__(self, input_image_path: str, output_image_path: str, source_dna_path: str, target_dna_path: str,
                 random_seed: int = None, need_logs: bool = True):
        """
        execute the self-assessment process (case study).

        :param input_image_path: original image path.
        :type input_image_path: str

        :param output_image_path: path to save the decoded image.
        :type output_image_path: str

        :param source_dna_path: path to the FASTA file of the original DNA sequence list generated by encoding.
        :type source_dna_path: str

        :param target_dna_path: path to the FASTA file for the DNA sequence list confused by wet experiments.
        :type target_dna_path: str

        :param random_seed: random seed for wet experimental process if required.
        :type random_seed: int or None

        :param need_logs: need to print process logs if required.
        :type need_logs: bool
        """
        if need_logs:
            if self.error_free:
                print("Error-free simulation (random seed = " + str(random_seed) + ").")
            else:
                print("Practical 3% simulation (random seed = " + str(random_seed) + ").")
            print("*" * 100)
            print()

        source_dna_sequences = self.coder.image_to_dna(input_image_path=input_image_path, need_logs=need_logs)

        for dna_sequence in source_dna_sequences:
            if len(dna_sequence) < 100 or len(dna_sequence) > 200:
                raise ValueError("The total length of every DNA sequence needs to be between 100nt and 200nt.")

            valid_length = 0
            for nucleotide in ["A", "C", "G", "T"]:
                valid_length += dna_sequence.count(nucleotide)

            if len(dna_sequence) != valid_length:
                raise ValueError("There are illegal characters in the sequence, which do not belong to any of A/C/G/T.")

        if need_logs:
            print("Write the original DNA sequence list to the FASTA file (" + source_dna_path + ").")
        number = len(source_dna_sequences)
        with open(source_dna_path, "w") as file:
            for dna_index, dna_sequence in enumerate(source_dna_sequences):
                file.write(">" + str(dna_index).zfill(len(str(number))) + "_" + str(number) + "\n")
                file.write(dna_sequence + "\n")

        if need_logs:
            print("Simulate the wet pipeline.")
        target_dna_sequences = self.simulate_wet_pipeline(source_dna_sequences=source_dna_sequences,
                                                          random_seed=random_seed)

        if need_logs:
            print("Write the practical DNA sequence list to the FASTA file (" + target_dna_path + ").")
        number = len(target_dna_sequences)
        with open(target_dna_path, "w") as file:
            for dna_index, dna_sequence in enumerate(target_dna_sequences):
                file.write(">" + str(dna_index).zfill(len(str(number))) + "_" + str(number) + "\n")
                file.write(dna_sequence + "\n")

        self.coder.dna_to_image(dna_sequences=target_dna_sequences, output_image_path=output_image_path,
                                need_logs=need_logs)

        scores = self.calculate_score(path_1=input_image_path, path_2=output_image_path,
                                      source_dna_path=source_dna_path)

        print("\nFor the figure (" + input_image_path + "), the score of your coder are as follow:")
        print("Density score (20% of the total): " + ("%.3f" % (scores[0] * 100)))
        print("Compatibility score (30% of the total): " + ("%.3f" % (scores[1] * 100)))
        if scores[2] > 0:
            print("Recovery score (50% of the total): " + ("%.3f" % (scores[2] * 100)))
        else:
            print("Recovery score (50% of the total): 0.000 (the obtained data unable to parse as image).")

        print("Total score: %.3f" % ((0.2 * scores[0] + 0.3 * scores[1] + 0.5 * scores[2]) * 100))

    def simulate_wet_pipeline(self, source_dna_sequences: list, random_seed: int = None, need_logs: bool = True):
        """
        Simulate the wet experimental process.

        :param source_dna_sequences: original DNA sequence list.
        :type source_dna_sequences: list

        :param random_seed: random seed for wet experimental process if required.
        :type random_seed: int or None

        :param need_logs: need to print process logs if required.
        :type need_logs: bool

        :return: a list of DNA sequences with 3% of edit errors and disordered order.
        :rtype: list

        .. note::
            The random seed of your coder should be passed in during its initialization.
        """
        if random_seed is not None:
            seed(random_seed)

        if not self.error_free:
            if need_logs:
                print("Introduce 3% of edit errors, including 1.50% mutations, 0.75% insertions, and 0.75% deletions.")
            target_dna_sequences = []
            for index, source_dna_sequence in enumerate(source_dna_sequences):
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

                if need_logs:
                    self.monitor(index + 1, len(source_dna_sequences))

                target_dna_sequences.append(target_dna_sequence)
        else:
            target_dna_sequences = deepcopy(source_dna_sequences)

        if need_logs:
            print("Shuffle the obtained DNA sequences.")
        # shuffle(target_dna_sequences)

        if random_seed is not None:
            seed(None)

        return target_dna_sequences

    @staticmethod
    def calculate_score(path_1: str, path_2: str, source_dna_path: str):
        """
        Calculate the score for the current converting task.

        :param path_1: original image path.
        :type path_1: str

        :param path_2: path to save the decoded image.
        :type path_2: str

        :param source_dna_path: original DNA sequence list.
        :type source_dna_path: str

        :return: density score, compatibility score, retrieval score.
        :rtype: float, float, float

        .. note::
            The total score = 20% density score + 30% compatibility score + 50% recovery score.
        """
        # load original DNA sequences from the FASTA file.
        dna_sequences = []
        with open(source_dna_path, "r") as file:
            for line in file.readlines():
                if line[0] != ">":
                    dna_sequences.append(line[:-1])

        # calculate the density score.
        expected_image = imread(path_1)
        b_number, d_number = expected_image.shape[0] * expected_image.shape[1] * 24, 0
        for dna_sequence in dna_sequences:
            d_number += len(dna_sequence)
        density_score = 1 - d_number / b_number if d_number < b_number else 0

        # calculate the compatibility score.
        maximum_homopolymer, maximum_gc_bias = 1, 0
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
            maximum_homopolymer = max(homopolymer, maximum_homopolymer)
            gc_bias = abs((dna_sequence.count("G") + dna_sequence.count("C")) / len(dna_sequence) - 0.5)
            maximum_gc_bias = max(gc_bias, maximum_gc_bias)
        h_score = (1.0 - (maximum_homopolymer - 1) / 5.0) / 2.0 if maximum_homopolymer < 6 else 0
        c_score = (1.0 - maximum_gc_bias / 0.3) / 2.0 if maximum_gc_bias < 0.3 else 0
        compatibility_score = h_score + c_score

        # calculate the recovery score.
        try:
            obtained_image = imread(path_2)
            ssim_value = structural_similarity(expected_image, obtained_image, channel_axis=-1)
            recovery_score = (ssim_value - 0.84) / 0.16 if ssim_value > 0.84 else 0
        except AttributeError:
            recovery_score = 0.0  # unable to parse as image, SSIM value cannot be calculated, the recovery score is 0.

        return density_score, compatibility_score, recovery_score
