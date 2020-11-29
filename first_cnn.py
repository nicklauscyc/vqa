# figure out how to create an iterable dataset for our data

path = 'C:/Users/lajja/Documents/Fall2020/ML/vqa/test/test/'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
     
dataset = datasets.ImageFolder(path, transform=transform)

https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2