import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from model import EncoderDecoder
from PIL import Image
from vocabulary import Vocabulary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms = T.Compose([T.Resize(224), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

vocabulary = Vocabulary()
vocabulary.load_vocab()

model = EncoderDecoder(embed_size=512, vocab_size=len(vocabulary),
                       attention_dim=512, encoder_dim=2048, decoder_dim=512).to(device)

model.load_state_dict(torch.load('settings/model_weights.pth', map_location=torch.device('cpu')))
model.eval()


def show_image(inp, title=None):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    input_image = Image.open("images/9aaa53e4b685cc7ebe46cddc780e30d3.jpg")
    img = transforms(input_image)
    img = img.unsqueeze(0)
    features = model.encoder(img[0:1].to(device))
    caps, alphas = model.decoder.generate_caption(features, vocab=vocabulary)
    caption = ' '.join(caps)
    show_image(img[0], title=caption)
