from lib import *
from config import *
from utils import load_model
from ImageTransform import ImageTransform

class_index = ["ants", "bees"]


class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict(self, output):
        max_id = np.argmax(output.detach().numpy())
        return self.class_index[max_id]


predictor = Predictor(class_index)


def predict(img):
    use_pretrain = True
    net = models.vgg16(use_pretrain)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()

    # Prepare model
    model = load_model(net, save_path)

    # Prepare input images
    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, phase='test')
    img_transformed = img_transformed.unsqueeze_(0)

    # Predict
    output = model(img_transformed)
    response = predictor.predict(output)

    return response
