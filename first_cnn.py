# figure out how to create an iterable dataset for our data

path = 'C:/Users/lajja/Documents/Fall2020/ML/vqa/processed-test/'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
     
dataset = datasets.ImageFolder(path, transform=transform)
