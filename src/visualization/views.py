import matplotlib.pyplot as plt

from data.dataset_commonlib import *

def main():
    y2015 = read_csv()
    y2015.plot()

    plt.show()

if __name__ == "__main__":
    main()
