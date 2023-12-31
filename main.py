import torch
import pickle
import cv2
import os
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor, AdamW, ViTImageProcessor, VisionEncoderDecoderModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

model_path = 'model'
train_pickle_path = 'train_data.pickle'
valid_pickle_path = 'valid_data.pickle'
image_directory = 'images' 
test_image_path = 'test.jpg'
num_epochs = 5    # Fine-tune the model
label_list = ["小白", "巧巧", "冏媽", "乖狗", "花捲", "超人", "黑胖", "橘子"]
label_dictionary = {"小白": 0, "巧巧": 1, "冏媽": 2, "乖狗": 3, "花捲": 4, "超人": 5, "黑胖": 6, "橘子": 7}
num_classes = len(label_dictionary)  # Adjust according to your classification task
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

def data_generate(dataset):
    images = []
    labels = []
    image_processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    for folder_name in os.listdir(image_directory):
        folder_path = os.path.join(image_directory, folder_name)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                if image_file.startswith(dataset):
                    image_path = os.path.join(folder_path, image_file)
                    # print(image_path)

                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(img)
                    img = img.resize((224, 224))
                    inputs = image_processor(images=img, return_tensors="pt")
                    images.append(inputs['pixel_values'].squeeze(0).numpy())
                    labels.append(int(folder_name.split('_')[0]))

    images = np.array(images)
    labels = np.array(labels)

    # Now you can pickle this data
    train_data = {'img': images, 'label': labels}
    with open(f'{dataset}_data.pickle', 'wb') as f:
        pickle.dump(train_data, f)

def train_model():
    
    if not os.path.exists(valid_pickle_path):
        data_generate('valid')
    if not os.path.exists(train_pickle_path):
        data_generate('train')

    # Load the train and vaild
    with open("train_data.pickle", "rb") as f:
        train_data = pickle.load(f)

    with open("valid_data.pickle", "rb") as f:
        valid_data = pickle.load(f)

    # Convert the dataset into torch tensors
    train_inputs = torch.tensor(train_data["img"])
    train_labels = torch.tensor(train_data["label"])
    valid_inputs = torch.tensor(valid_data["img"])
    valid_labels = torch.tensor(valid_data["label"])

    # Create the TensorDataset
    train_dataset = TensorDataset(train_inputs, train_labels)
    valid_dataset = TensorDataset(valid_inputs, valid_labels)

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    # Define the model and move it to the GPU
    
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k', num_labels=num_classes)
    model.to(device)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):

            # Move batch to the GPU
            batch = [r.to(device) for r in batch]

            # Unpack the inputs from our dataloader
            inputs, labels = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, labels=labels)

            # Compute loss
            loss = outputs.loss

            # Backward pass
            loss.backward()


            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the loss
            total_loss += loss.item()

            # print(f'{i}/{len(train_loader)} ')
            
        # Get the average loss for the entire epoch
        avg_loss = total_loss / len(train_loader)
        
        # Print the loss
        print('Epoch:', epoch + 1, 'Training Loss:', avg_loss)
        
        
    # Evaluate the model on the validation set
    model.eval()
    total_correct = 0

    for batch in valid_loader:
        # Move batch to the GPU
        batch = [t.to(device) for t in batch]

        # Unpack the inputs from our dataloader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)

        # Get the predictions
        predictions = torch.argmax(outputs.logits, dim=1)

        # Update the total correct
        total_correct += torch.sum(predictions == labels)
        
    # Calculate the accuracy
    accuracy = total_correct / len(valid_dataset)
    print('Validation accuracy:', accuracy.item())

    model.save_pretrained("model")

def predict():
    # Load the model
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=num_classes)

    image_processor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')


    # Load the test data
    # Load the image

    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to 224x224 pixels
    img = Image.fromarray(img)
    img = img.resize((224, 224))

    # img to tensor
    # Preprocess the image and generate features
    inputs = image_processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = logits.argmax(-1).item()

    return label_list[predicted_class_idx] if probabilities.max().item() > 0.90 else '不是校狗'

def captioning():
    
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    images = []
    for image_path in [test_image_path]:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[-1]

def output(predict_class, caption):
    conj = ['are', 'is', 'dog']
    if predict_class == '不是校狗' or caption.find('dog') == -1:
        print(f'{caption} ({predict_class})')
    else:
        for c in conj:
            if caption.find(c) != -1:
                print(f'{predict_class} is{caption[caption.find(c) + len(c):]}')
                return
        print(f'{caption} ({predict_class})')


if __name__ == '__main__':

    if not os.path.exists(model_path):
        train_model()
    output(predict(), captioning())