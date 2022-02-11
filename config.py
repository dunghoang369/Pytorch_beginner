from lib import *

# Gieo hạt
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
epochs = 2

save_path = './weight_fine_tuning.pth'