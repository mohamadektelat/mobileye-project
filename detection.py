try:
    from Configurations import *
except ImportError:
    print("Need to fix the installation")
    raise

FILE = "model.pth"
model = torch.load(FILE).to('cpu')
model.eval()

image = np.asarray(io.imread(
    r"C:\Users\Mohamad-PC\Desktop\mobileye\mobileye-project-mobileye-group-4\cropped_images\Traffic Light\g_True_False"
    r"_1_aachen_000127_000019_leftImg8bit.png"))
image2 = np.asarray(io.imread(
    r"C:\Users\Mohamad-PC\Desktop\mobileye\mobileye-project-mobileye-group-4\cropped_images\Not Traffic Light\g_False"
    r"_False_1_aachen_000084_000019_leftImg8bit.png"))

result = model(functional.to_tensor(image))
result2 = model(functional.to_tensor(image2))


print(torch.nn.Sigmoid()(result.item()))
print(torch.nn.Sigmoid()(result2.item()))