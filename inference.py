import torch
from cnn import CNN
from urbansoundDataset import UrbanSoundDataset
import torchaudio.transforms as tf
from training import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNN()
    state_dict = torch.load("cnn_audio.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset
    mel_spectrogram = tf.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    # hop_length is usually n_fft/2
    # ms = mell_spectogram(signal)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            'cpu')

    # get a sample from the urban sound dataset for inference
    input, target = usd[5][0], usd[5][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0) #to introduce the extra dim for batch_size (to 1)


    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")