import numpy as np
import matplotlib.pyplot as plt
import random

def get_data():
    data = np.loadtxt('data.dat', dtype=np.float64)
    return data

def mistake(w, x, y):
    if y * np.dot(w, x) < 0:
        return True
    elif y * np.dot(w, x) == 0 and y > 0: # sign(0) = -1
        return True
    return False

def random_example(data, x0):
    example = random.choice(data)
    example = np.insert(example, 0, x0)
    example_x = example[0:-1]
    example_y = example[-1]
    return example, example_x, example_y

if __name__ == "__main__":
    updatesnumber = []
    for i in range(1000):
        x0 = 1
        w = np.zeros(13)
        updates = 0
        data = get_data()
        # print(5*data.shape[0])
        example, example_x, example_y =  random_example(data, x0)
        # print(example)
        
        check = 0
        while check <= 5 * data.shape[0]:
            if mistake(w, example_x, example_y):
                w = w + example_y * example_x
                updates += 1
                check = 0
            else:
                check += 1
                example, example_x, example_y =  random_example(data, x0)
            
        print(w)
        updatesnumber.append(updates)
        
    print(updatesnumber)
    print(np.median(updatesnumber))
    plt.hist(updatesnumber, bins=200)
    # plt.show()
    plt.savefig('PLA_12.png')
