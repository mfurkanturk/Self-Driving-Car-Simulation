from Utils import *
from sklearn.model_selection import train_test_split

path = "Data"
data = importData(path)

## Drop useless data
data = dataPartition(data, display=False)

## Turn data into np array
imagesPath, steering = loadData(path, data)
print(steering)

## Create Train, Test and Validation data
x_Train, x_Val, y_Train, y_Val = train_test_split(
    imagesPath, steering, test_size=0.2, random_state=5
)

# create model
model = createModel()
model.summary()

# train model
history = model.fit(
    batchGen(x_Train, y_Train, 100, 1),
    steps_per_epoch=50,
    epochs=50,
    validation_data=batchGen(x_Val, y_Val, 100, 0),
    validation_steps=30,
)

model.save("model.keras")
print("model saved")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "Validation"])
plt.ylim([0, 1])
plt.title("Loss")
plt.xlabel("Epoch")
plt.show()
