"""
    This script contains supporting function for the audio data processing.
    It is used in several other scripts:
    for sequences and calculation of speech features

    other source of code authors:
    @author: Che-Jui Chang
    @author: Taras Kucherenko

    Test in the script:
    key1: overall test of main funtionality of audio features from tacotrone git dir

"""
import os.path
import sys
import librosa
import librosa.feature
import parselmouth as pm
import numpy as np
from pydub import AudioSegment
from Config.FeaturesConfig import features_config


class AudioProcessor:

    """
        Actually    i don't know if it is better to:
        *) Load everything from a directorty once called, retrieve all data into an
           attribute and give it back in a timed way at call time eg call_(time_stamp)

        *) Load the sample related to a given time_stamp
    """

    def __init__(self, audio_parameters:dict):

        self.NFFT = audio_parameters["NFFT"]
        self.MFCC_INPUTS = audio_parameters["MFCC_INPUTS"]
        self.HOP_LENGTH = audio_parameters["HOP_LENGTH"]
        self.DIM = audio_parameters["DIM"]

    def __call__(self, audio_path):

        print(f"Audio working '{audio_path}'")
        audio, sr = self.load_audio(audio_path)
        prosody = self.extract_prosodic_features(audio_path)
        mfcc = self.calculate_mfcc(audio, sr)
        melspec = self.calculate_spectrogram(audio, sr)

        print(audio.shape, prosody.shape, mfcc.shape, melspec.shape)

        audio_features = np.concatenate((prosody, mfcc[-prosody.shape[0]:, ...], melspec[-prosody.shape[0]:, ...]), axis=1)

        return audio_features

    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr

    def extract_prosodic_features(self, audio_filename):
        """
        Extract all 5 prosodic features
        Args:
            audio_filename:   file name for the audio to be used
        Returns:
            pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
        """

        # Read audio from file
        sound = AudioSegment.from_file(audio_filename, format="wav")

        # Alternative prosodic features
        pitch, energy = self.compute_prosody(audio_filename, self.HOP_LENGTH / 10)

        duration = len(sound) / 1000
        t = np.arange(0, duration, self.HOP_LENGTH / 10)

        energy_der = self.derivative(t, energy)
        pitch_der = self.derivative(t, pitch)

        # Average everything in order to match the frequency
        energy = self.average(energy, 10)
        energy_der = self.average(energy_der, 10)
        pitch = self.average(pitch, 10)
        pitch_der = self.average(pitch_der, 10)

        # Cut them to the same size
        min_size = min(len(energy), len(energy_der), len(pitch_der), len(pitch_der))
        energy = energy[:min_size]
        energy_der = energy_der[:min_size]
        pitch = pitch[:min_size]
        pitch_der = pitch_der[:min_size]

        # Stack them all together
        pros_feature = np.stack((energy, energy_der, pitch, pitch_der))  # , pitch_ind))

        # And reshape
        pros_feature = np.transpose(pros_feature)

        return pros_feature

    def compute_prosody(self, audio_filename, time_step=0.05):

        audio = pm.Sound(audio_filename)

        # Extract pitch and intensity
        pitch = audio.to_pitch(time_step=time_step)
        intensity = audio.to_intensity(time_step=time_step,
                                       minimum_pitch=100.0)

        # Evenly spaced time steps
        times = np.arange(0, audio.get_total_duration() - time_step, time_step)

        # Compute prosodic features at each time step
        pitch_values = np.nan_to_num(
            np.asarray([pitch.get_value_at_time(t) for t in times]))
        intensity_values = np.nan_to_num(
            np.asarray([intensity.get_value(t) for t in times]))

        intensity_values = np.clip(
            intensity_values, np.finfo(intensity_values.dtype).eps, None)

        # Normalize features [Chiu '11]
        pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
        intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

        return pitch_norm, intensity_norm

    def average(self, arr, n):
        """ Replace every "n" values by their average
        Args:
            arr: input array
            n:   number of elements to average on
        Returns:
            resulting array
        """
        end = n * int(len(arr) / n)
        return np.mean(arr[:end].reshape(-1, n), 1)

    def calculate_spectrogram(self, audio, sr):

        """
        Calculate spectrogram for the audio file
        Args:
            audio_filename: audio file name
            duration: the duration (in seconds) that should be read from the file (can be used to load just a part of the audio file)
        Returns:
            log spectrogram values
        """

        # Make stereo audio being mono
        if len(audio.shape) == 2:
            audio = (audio[:, 0] + audio[:, 1]) / 2

        spectr = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=self.NFFT,
                                                hop_length=int(self.HOP_LENGTH * sr),
                                                n_mels=self.DIM)

        # Shift into the log scale
        eps = 1e-10
        log_spectr = np.log(abs(spectr) + eps)

        return np.transpose(log_spectr)

    def calculate_mfcc(self, audio, sr):

        """
        Calculate MFCC features for the audio in a given file
        Args:
            audio_filename: file name of the audio
        Returns:
            feature_vectors: MFCC feature vector for the given audio file
        """

        # Make stereo audio being mono
        if len(audio.shape) == 2:
            audio = (audio[:, 0] + audio[:, 1]) / 2

        # Calculate MFCC feature with the window frame it was designed for
        input_vectors = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.MFCC_INPUTS, n_fft=self.NFFT,
                                             hop_length=int(self.HOP_LENGTH * sr),
                                             n_mels=self.DIM)

        return input_vectors.transpose()

    def derivative(self, x, f):

        """
        Calculate numerical derivative (by FDM) of a 1d array
        Args:
            x: input space x
            f: Function of x
        Returns:
            der:  numerical derivative of f wrt x
        """

        x = 1000 * x  # from seconds to milliseconds

        # Normalization:
        dx = (x[1] - x[0])

        cf = np.convolve(f, [1, -1]) / dx

        # Remove unstable values
        der = cf[:-1].copy()
        der[0] = 0

        return der


def serial_shape_printer(**kwargs):

    # use this function to print multiple element shape

    for ii, arg in enumerate(kwargs.items()):
        print(f"Shape of {arg[0]} is {arg[1].shape}")


if __name__ == "__main__":

    # tested 11-12-23: working

    key1 = True
    if key1:
        fparam = features_config()
        """
        audio_processor = AudioProcessor(audio_main_dir=parameters.wav_main_dir,
                                         audio_interloctr_dir=parameters.wav_interloctr_dir,
                                         audio_parameters=parameters.audio_parameters)
        """
        audio_processor = AudioProcessor(audio_parameters=fparam.audio_parameters)
        audio_path = r"C:\Users\faval\genea2023_dataset\trn\main-agent\wav\trn_2023_v0_001_main-agent.wav"
        audio_features = audio_processor(audio_path)

        serial_shape_printer(audio_features=audio_features)


