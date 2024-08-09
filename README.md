# Machine Learning Image Classification

A machine learning model that classifies images from the CIFAR-10 dataset.

Image goes here...

## About

This machine learning model is trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and classifies images into one of ten different classes based on the object the image is portraying. It's built with PyTorch and uses other libraries like Matplotlib for plotting the training results and PrettyTable for printing the results to the console in table format.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images containing one of ten object classes, with 6000 images per class. The ten different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. Automobile includes sedans, SUVs and other similar vehicles while Truck includes only big trucks and neither class includes pickup trucks.

## Results

The neural network was trained for 70 epochs with a batch size of 4 and a learning rate of 0.001 using SGD as the optimizer and CrossEntropyLoss as the criterion. Training was done on 50,000 images (83.33% of the dataset) while the remaining 10,000 images (16.67%) were used to validate the results.

```
+-------+---------------------+
| Epoch |         Loss        |
+-------+---------------------+
|   1   |  2.231081960735321  |
|   2   |  1.9301664771604539 |
|   3   |  1.6754022343444823 |
|   4   |  1.5297706640434265 |
|   5   |  1.4493177846026422 |
|   10  |  1.1808797737580539 |
|   15  |  1.0170512462495267 |
|   20  |  0.8937877064520866 |
|   25  |  0.7943712613232061 |
|   30  |  0.7081599121124483 |
|   35  |  0.6319251888247394 |
|   40  |  0.5638960482444847 |
|   45  |  0.504354892890777  |
|   50  |  0.4453727933392429 |
|   55  |  0.3989951912998638 |
|   60  | 0.35246151381471424 |
|   65  | 0.31777741762888717 |
|   70  |  0.2826915991319276 |
+-------+---------------------+
```

Image goes here...

The model achieved an accuracy of 62.56% on the testing set.

```
Test Accuracy: 62.56% (6256/10000)
```

## License

This software is licensed under the [MIT license](LICENSE).