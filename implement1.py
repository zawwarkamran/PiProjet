import argparse
import numpy as np
import numpy.random as npr
import cv2 as cv
import os
import time
from torchdetector import *
import torch.optim as optim
import matplotlib.pyplot as plt
print(os.getcwd())


def TakeImg(name, samples=10):

	cap = cv.VideoCapture(0)

	def rescale_frame(frame):
		width = 128
		height = 128
		dim = (width, height)
		return cv.resize(frame, dim, interpolation = cv.INTER_AREA)

	for i in range(samples):
		# if np.mod(i, 5) == 0:
		# 	print('Please adjust face angle')
		ret, fram = cap.read()
		frame = cv.cvtColor(fram, cv.COLOR_BGR2GRAY)
		frame = rescale_frame(fram)

		cv.imwrite('{}{}.jpg'.format(name, i), frame)

	cap.release()
	cv.destroyAllWindows

# TakeImg('zawwar', samples=10)

data_lst = []
for i in os.listdir('/Users/zawwarkamran/Documents/GitHub/PiProjet'):
	if i.endswith('.jpg'):
		cv_to_append = np.moveaxis(cv.imread(i), [0,1,2], [2,1,0])
		cv_to_append = torch.from_numpy(cv_to_append)
		data_lst.append([cv_to_append, cv_to_append])


model = FacialDetection()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(data_lst, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        inputs = inputs.float()
        inputs = inputs.unsqueeze(0)
        labels = labels.long()
        labels = labels.unsqueeze(0)
        print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



