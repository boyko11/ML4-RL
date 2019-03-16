import matplotlib.pyplot as plt


def plot_line(title, x_label, y_label, x_values, y_values ):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()

    plt.plot(x_values, y_values)

    #plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    plt.show()


def plot_two_lines(title, x_label, y_label, x_values, line1_label, line1_values,  line2_label, line2_values ):

    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()

    plt.plot(x_values, line1_values, color="C0", label=line1_label)
    plt.plot(x_values, line2_values, color="C1", label=line2_label)

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    plt.show()


episodes = range(1, 1940)
episode_rewards = [26.0, 16.0, 11.0, 22.0, 15.0, 21.0, 15.0, 14.0, 13.0, 12.0, 24.0, 18.0, 11.0, 9.0, 11.0, 9.0, 11.0, 12.0, 11.0, 8.0, 9.0, 8.0, 11.0, 10.0, 13.0, 9.0, 10.0, 11.0, 9.0, 10.0, 10.0, 10.0, 11.0, 10.0, 11.0, 11.0, 8.0, 10.0, 13.0, 14.0, 28.0, 43.0, 15.0, 20.0, 37.0, 21.0, 21.0, 19.0, 29.0, 18.0, 8.0, 10.0, 9.0, 8.0, 32.0, 10.0, 12.0, 10.0, 14.0, 45.0, 11.0, 29.0, 17.0, 57.0, 44.0, 37.0, 28.0, 30.0, 34.0, 20.0, 35.0, 22.0, 10.0, 37.0, 23.0, 16.0, 24.0, 13.0, 15.0, 26.0, 44.0, 31.0, 35.0, 55.0, 64.0, 94.0, 65.0, 30.0, 19.0, 31.0, 35.0, 15.0, 19.0, 24.0, 29.0, 34.0, 32.0, 11.0, 21.0, 25.0, 27.0, 21.0, 69.0, 30.0, 20.0, 26.0, 24.0, 20.0, 14.0, 20.0, 19.0, 19.0, 51.0, 34.0, 12.0, 12.0, 19.0, 22.0, 29.0, 25.0, 39.0, 54.0, 44.0, 73.0, 49.0, 25.0, 28.0, 20.0, 37.0, 25.0, 18.0, 16.0, 11.0, 14.0, 21.0, 14.0, 18.0, 12.0, 13.0, 15.0, 55.0, 31.0, 12.0, 56.0, 23.0, 16.0, 17.0, 15.0, 47.0, 17.0, 15.0, 9.0, 9.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 8.0, 12.0, 10.0, 25.0, 18.0, 10.0, 13.0, 18.0, 9.0, 10.0, 8.0, 9.0, 10.0, 9.0, 9.0, 10.0, 10.0, 9.0, 10.0, 10.0, 11.0, 20.0, 21.0, 17.0, 15.0, 38.0, 14.0, 12.0, 20.0, 31.0, 20.0, 26.0, 52.0, 40.0, 33.0, 15.0, 14.0, 25.0, 12.0, 21.0, 28.0, 15.0, 61.0, 19.0, 14.0, 11.0, 11.0, 14.0, 13.0, 10.0, 10.0, 10.0, 9.0, 9.0, 18.0, 28.0, 52.0, 44.0, 38.0, 29.0, 54.0, 37.0, 51.0, 20.0, 18.0, 37.0, 37.0, 29.0, 37.0, 76.0, 18.0, 9.0, 11.0, 11.0, 10.0, 9.0, 11.0, 18.0, 66.0, 12.0, 15.0, 18.0, 22.0, 60.0, 25.0, 32.0, 20.0, 23.0, 19.0, 20.0, 20.0, 37.0, 39.0, 23.0, 96.0, 41.0, 33.0, 30.0, 24.0, 16.0, 24.0, 35.0, 24.0, 34.0, 45.0, 55.0, 45.0, 50.0, 25.0, 63.0, 71.0, 51.0, 29.0, 29.0, 24.0, 68.0, 69.0, 63.0, 62.0, 32.0, 62.0, 47.0, 87.0, 20.0, 25.0, 54.0, 26.0, 47.0, 40.0, 200.0, 37.0, 90.0, 52.0, 160.0, 53.0, 39.0, 29.0, 36.0, 51.0, 59.0, 30.0, 29.0, 60.0, 48.0, 50.0, 172.0, 200.0, 54.0, 52.0, 36.0, 65.0, 54.0, 71.0, 164.0, 39.0, 65.0, 38.0, 156.0, 37.0, 30.0, 39.0, 33.0, 25.0, 43.0, 36.0, 59.0, 44.0, 196.0, 32.0, 36.0, 57.0, 92.0, 86.0, 20.0, 27.0, 23.0, 35.0, 29.0, 56.0, 37.0, 27.0, 39.0, 47.0, 37.0, 196.0, 28.0, 54.0, 100.0, 44.0, 42.0, 79.0, 58.0, 43.0, 68.0, 76.0, 26.0, 119.0, 40.0, 43.0, 54.0, 42.0, 39.0, 25.0, 28.0, 35.0, 53.0, 58.0, 52.0, 82.0, 30.0, 33.0, 107.0, 22.0, 55.0, 200.0, 143.0, 200.0, 60.0, 68.0, 59.0, 33.0, 70.0, 43.0, 33.0, 62.0, 52.0, 25.0, 29.0, 24.0, 27.0, 15.0, 17.0, 35.0, 56.0, 200.0, 118.0, 104.0, 59.0, 96.0, 60.0, 38.0, 55.0, 82.0, 48.0, 74.0, 200.0, 46.0, 124.0, 42.0, 47.0, 70.0, 69.0, 87.0, 54.0, 50.0, 66.0, 60.0, 54.0, 38.0, 164.0, 53.0, 54.0, 38.0, 50.0, 64.0, 32.0, 48.0, 29.0, 36.0, 143.0, 51.0, 200.0, 37.0, 39.0, 54.0, 94.0, 40.0, 49.0, 52.0, 128.0, 45.0, 69.0, 60.0, 200.0, 200.0, 34.0, 33.0, 55.0, 25.0, 77.0, 64.0, 46.0, 34.0, 34.0, 200.0, 34.0, 58.0, 110.0, 200.0, 27.0, 38.0, 66.0, 87.0, 64.0, 67.0, 71.0, 68.0, 149.0, 42.0, 96.0, 53.0, 120.0, 60.0, 53.0, 59.0, 78.0, 200.0, 200.0, 64.0, 70.0, 36.0, 63.0, 52.0, 200.0, 62.0, 82.0, 47.0, 51.0, 47.0, 148.0, 68.0, 58.0, 46.0, 55.0, 37.0, 33.0, 200.0, 92.0, 49.0, 43.0, 38.0, 70.0, 33.0, 41.0, 200.0, 40.0, 27.0, 37.0, 59.0, 53.0, 91.0, 88.0, 93.0, 58.0, 58.0, 55.0, 48.0, 48.0, 70.0, 200.0, 72.0, 69.0, 200.0, 102.0, 87.0, 161.0, 148.0, 104.0, 65.0, 78.0, 108.0, 92.0, 129.0, 61.0, 104.0, 113.0, 138.0, 142.0, 200.0, 161.0, 150.0, 200.0, 200.0, 200.0, 200.0, 200.0, 158.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 98.0, 32.0, 34.0, 128.0, 158.0, 200.0, 200.0, 200.0, 200.0, 200.0, 76.0, 155.0, 200.0, 36.0, 36.0, 69.0, 159.0, 200.0, 161.0, 200.0, 166.0, 200.0, 200.0, 155.0, 155.0, 200.0, 200.0, 195.0, 200.0, 200.0, 200.0, 68.0, 200.0, 195.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 135.0, 200.0, 200.0, 66.0, 99.0, 21.0, 39.0, 85.0, 200.0, 94.0, 200.0, 42.0, 57.0, 200.0, 143.0, 46.0, 65.0, 200.0, 176.0, 200.0, 135.0, 144.0, 200.0, 75.0, 63.0, 200.0, 200.0, 200.0, 41.0, 68.0, 200.0, 138.0, 200.0, 68.0, 169.0, 200.0, 54.0, 200.0, 200.0, 200.0, 200.0, 200.0, 165.0, 175.0, 200.0, 158.0, 200.0, 146.0, 136.0, 200.0, 178.0, 200.0, 200.0, 194.0, 114.0, 192.0, 200.0, 200.0, 200.0, 200.0, 64.0, 38.0, 200.0, 200.0, 157.0, 156.0, 200.0, 200.0, 200.0, 200.0, 181.0, 200.0, 200.0, 172.0, 200.0, 200.0, 200.0, 93.0, 195.0, 137.0, 166.0, 149.0, 149.0, 173.0, 164.0, 100.0, 200.0, 200.0, 200.0, 200.0, 110.0, 145.0, 112.0, 125.0, 200.0, 104.0, 119.0, 131.0, 200.0, 71.0, 139.0, 102.0, 126.0, 200.0, 116.0, 140.0, 78.0, 200.0, 147.0, 116.0, 80.0, 157.0, 200.0, 200.0, 174.0, 36.0, 87.0, 89.0, 140.0, 113.0, 128.0, 162.0, 45.0, 200.0, 185.0, 176.0, 141.0, 144.0, 161.0, 144.0, 200.0, 200.0, 127.0, 187.0, 112.0, 200.0, 175.0, 200.0, 181.0, 139.0, 162.0, 200.0, 187.0, 200.0, 143.0, 114.0, 154.0, 200.0, 193.0, 181.0, 176.0, 107.0, 168.0, 180.0, 177.0, 93.0, 131.0, 156.0, 114.0, 151.0, 200.0, 87.0, 175.0, 200.0, 200.0, 171.0, 87.0, 200.0, 70.0, 200.0, 143.0, 118.0, 124.0, 142.0, 200.0, 200.0, 114.0, 101.0, 128.0, 112.0, 123.0, 125.0, 152.0, 200.0, 166.0, 127.0, 129.0, 113.0, 120.0, 115.0, 86.0, 95.0, 119.0, 104.0, 120.0, 114.0, 122.0, 106.0, 106.0, 128.0, 141.0, 123.0, 105.0, 110.0, 95.0, 160.0, 171.0, 127.0, 128.0, 86.0, 88.0, 92.0, 77.0, 100.0, 107.0, 150.0, 57.0, 106.0, 101.0, 112.0, 156.0, 114.0, 111.0, 115.0, 84.0, 108.0, 92.0, 93.0, 124.0, 121.0, 131.0, 157.0, 102.0, 110.0, 117.0, 164.0, 166.0, 152.0, 138.0, 140.0, 154.0, 136.0, 153.0, 133.0, 132.0, 161.0, 114.0, 168.0, 136.0, 154.0, 123.0, 104.0, 200.0, 188.0, 166.0, 174.0, 200.0, 155.0, 179.0, 188.0, 156.0, 153.0, 179.0, 91.0, 163.0, 200.0, 174.0, 172.0, 192.0, 200.0, 200.0, 134.0, 71.0, 118.0, 199.0, 96.0, 158.0, 200.0, 28.0, 200.0, 175.0, 158.0, 169.0, 163.0, 119.0, 151.0, 176.0, 180.0, 163.0, 200.0, 156.0, 196.0, 178.0, 177.0, 183.0, 174.0, 114.0, 160.0, 190.0, 189.0, 177.0, 200.0, 174.0, 176.0, 190.0, 187.0, 183.0, 200.0, 200.0, 184.0, 164.0, 168.0, 183.0, 118.0, 200.0, 200.0, 200.0, 200.0, 186.0, 182.0, 184.0, 200.0, 198.0, 200.0, 189.0, 200.0, 191.0, 39.0, 194.0, 179.0, 194.0, 200.0, 200.0, 200.0, 196.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 199.0, 200.0, 200.0, 200.0, 135.0, 198.0, 200.0, 133.0, 200.0, 190.0, 196.0, 182.0, 119.0, 75.0, 200.0, 146.0, 200.0, 179.0, 197.0, 200.0, 163.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 137.0, 200.0, 200.0, 200.0, 182.0, 200.0, 200.0, 200.0, 198.0, 44.0, 163.0, 153.0, 163.0, 161.0, 175.0, 179.0, 172.0, 192.0, 162.0, 144.0, 171.0, 154.0, 154.0, 156.0, 157.0, 160.0, 167.0, 151.0, 161.0, 173.0, 155.0, 165.0, 169.0, 163.0, 155.0, 175.0, 160.0, 175.0, 175.0, 174.0, 190.0, 176.0, 181.0, 200.0, 181.0, 61.0, 77.0, 110.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 12.0, 200.0, 200.0, 116.0, 136.0, 200.0, 192.0, 179.0, 169.0, 200.0, 192.0, 181.0, 185.0, 187.0, 200.0, 179.0, 160.0, 169.0, 119.0, 154.0, 171.0, 159.0, 176.0, 176.0, 180.0, 200.0, 126.0, 137.0, 149.0, 140.0, 136.0, 174.0, 152.0, 141.0, 150.0, 140.0, 134.0, 142.0, 162.0, 171.0, 149.0, 165.0, 158.0, 158.0, 143.0, 177.0, 159.0, 155.0, 159.0, 153.0, 171.0, 170.0, 169.0, 190.0, 173.0, 177.0, 191.0, 184.0, 164.0, 189.0, 184.0, 200.0, 173.0, 192.0, 183.0, 194.0, 200.0, 195.0, 200.0, 195.0, 193.0, 200.0, 200.0, 200.0, 188.0, 200.0, 183.0, 145.0, 200.0, 162.0, 170.0, 174.0, 180.0, 169.0, 174.0, 171.0, 195.0, 179.0, 180.0, 192.0, 184.0, 174.0, 193.0, 181.0, 200.0, 200.0, 200.0, 145.0, 187.0, 155.0, 190.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 197.0, 200.0, 200.0, 173.0, 195.0, 183.0, 200.0, 200.0, 200.0, 199.0, 200.0, 200.0, 182.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 165.0, 152.0, 200.0, 191.0, 195.0, 196.0, 173.0, 196.0, 185.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 145.0, 199.0, 200.0, 200.0, 200.0, 200.0, 184.0, 162.0, 170.0, 165.0, 166.0, 170.0, 165.0, 167.0, 174.0, 164.0, 176.0, 148.0, 180.0, 177.0, 160.0, 190.0, 165.0, 168.0, 185.0, 177.0, 174.0, 185.0, 171.0, 171.0, 195.0, 200.0, 176.0, 177.0, 181.0, 185.0, 200.0, 152.0, 178.0, 198.0, 192.0, 189.0, 200.0, 191.0, 178.0, 175.0, 194.0, 187.0, 181.0, 189.0, 197.0, 194.0, 190.0, 180.0, 181.0, 200.0, 200.0, 200.0, 200.0, 120.0, 200.0, 200.0, 99.0, 200.0, 200.0, 200.0, 163.0, 192.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 179.0, 200.0, 164.0, 200.0, 105.0, 136.0, 200.0, 200.0, 200.0, 171.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 188.0, 144.0, 200.0, 14.0, 104.0, 133.0, 200.0, 92.0, 12.0, 88.0, 91.0, 114.0, 80.0, 15.0, 73.0, 91.0, 125.0, 14.0, 88.0, 113.0, 106.0, 13.0, 13.0, 84.0, 112.0, 14.0, 54.0, 70.0, 200.0, 200.0, 200.0, 200.0, 197.0, 84.0, 138.0, 81.0, 127.0, 200.0, 144.0, 113.0, 200.0, 187.0, 177.0, 200.0, 83.0, 180.0, 82.0, 128.0, 200.0, 200.0, 200.0, 200.0, 135.0, 200.0, 200.0, 109.0, 103.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 167.0, 200.0, 133.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 11.0, 188.0, 10.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 125.0, 190.0, 200.0, 200.0, 152.0, 191.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 134.0, 158.0, 200.0, 123.0, 13.0, 123.0, 200.0, 193.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 89.0, 95.0, 109.0, 91.0, 79.0, 110.0, 181.0, 105.0, 142.0, 171.0, 169.0, 90.0, 122.0, 125.0, 111.0, 119.0, 18.0, 81.0, 191.0, 200.0, 177.0, 146.0, 180.0, 200.0, 156.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 122.0, 125.0, 175.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 107.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 16.0, 16.0, 139.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 181.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 190.0, 200.0, 200.0, 200.0, 200.0, 183.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 185.0, 179.0, 164.0, 159.0, 142.0, 156.0, 157.0, 165.0, 12.0, 108.0, 123.0, 200.0, 200.0, 145.0, 157.0, 158.0, 142.0, 155.0, 145.0, 145.0, 13.0, 128.0, 136.0, 155.0, 183.0, 172.0, 200.0, 98.0, 105.0, 143.0, 131.0, 118.0, 139.0, 145.0, 135.0, 140.0, 126.0, 143.0, 112.0, 13.0, 17.0, 109.0, 131.0, 141.0, 171.0, 187.0, 200.0, 200.0, 200.0, 198.0, 125.0, 142.0, 200.0, 179.0, 153.0, 82.0, 109.0, 200.0, 200.0, 135.0, 200.0, 200.0, 149.0, 136.0, 148.0, 149.0, 200.0, 77.0, 85.0, 86.0, 56.0, 72.0, 90.0, 144.0, 99.0, 124.0, 75.0, 93.0, 66.0, 82.0, 200.0, 200.0, 127.0, 200.0, 75.0, 95.0, 200.0, 200.0, 139.0, 200.0, 149.0, 177.0, 117.0, 97.0, 124.0, 59.0, 69.0, 200.0, 145.0, 200.0, 200.0, 200.0, 95.0, 200.0, 200.0, 172.0, 102.0, 200.0, 185.0, 124.0, 12.0, 62.0, 115.0, 71.0, 177.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 69.0, 10.0, 69.0, 200.0, 200.0, 200.0, 187.0, 200.0, 89.0, 200.0, 200.0, 142.0, 171.0, 200.0, 198.0, 200.0, 177.0, 199.0, 190.0, 200.0, 152.0, 170.0, 196.0, 200.0, 81.0, 200.0, 163.0, 180.0, 200.0, 200.0, 200.0, 184.0, 200.0, 200.0, 200.0, 200.0, 75.0, 200.0, 200.0, 60.0, 128.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 158.0, 46.0, 105.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 71.0, 75.0, 176.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 165.0, 196.0, 200.0, 200.0, 127.0, 167.0, 200.0, 166.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 103.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 167.0, 200.0, 171.0, 200.0, 200.0, 138.0, 200.0, 200.0, 200.0, 200.0, 145.0, 200.0, 200.0, 177.0, 196.0, 200.0, 200.0, 39.0, 17.0, 54.0, 79.0, 169.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 69.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 101.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 135.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 161.0, 200.0, 200.0, 200.0, 200.0, 200.0, 131.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 173.0, 200.0, 164.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 111.0, 200.0, 200.0, 200.0, 87.0, 119.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
running_average_reward = [20.0, 18.666666666666668, 16.75, 17.8, 17.333333333333332, 17.857142857142858, 17.5, 17.11111111111111, 16.7, 16.272727272727273, 16.916666666666668, 17.0, 16.571428571428573, 16.066666666666666, 15.75, 15.352941176470589, 15.11111111111111, 14.947368421052632, 14.75, 14.428571428571429, 14.181818181818182, 13.91304347826087, 13.791666666666666, 13.64, 13.6, 12.92, 12.68, 12.68, 12.16, 11.96, 11.52, 11.32, 11.2, 11.08, 11.04, 10.52, 10.12, 10.08, 10.24, 10.36, 11.12, 12.4, 12.52, 12.88, 14.04, 14.52, 15.04, 15.36, 16.12, 16.32, 16.28, 16.28, 16.2, 16.16, 17.04, 17.04, 17.12, 17.08, 17.24, 18.6, 18.6, 19.44, 19.72, 21.48, 22.68, 23.04, 22.44, 23.04, 23.6, 22.92, 23.48, 23.52, 23.16, 23.48, 23.68, 24.0, 24.56, 24.72, 25.0, 24.76, 26.12, 26.88, 27.88, 29.52, 30.28, 33.6, 35.04, 35.56, 34.04, 33.52, 33.44, 32.92, 32.48, 32.08, 32.44, 32.4, 32.8, 32.84, 32.2, 32.28, 32.72, 32.6, 34.84, 35.44, 35.2, 34.48, 34.2, 33.6, 31.96, 30.2, 27.2, 25.36, 26.2, 26.8, 26.04, 25.12, 25.28, 25.4, 25.6, 25.44, 25.64, 26.52, 27.84, 29.92, 30.88, 30.8, 31.08, 29.12, 29.4, 29.6, 29.28, 28.96, 28.6, 28.6, 28.64, 28.44, 28.4, 26.84, 26.0, 26.12, 27.84, 28.32, 27.92, 29.0, 28.92, 28.0, 26.52, 25.36, 24.32, 23.04, 22.64, 21.88, 21.44, 20.28, 19.68, 19.36, 19.12, 19.08, 18.92, 18.44, 18.28, 17.88, 17.88, 17.76, 18.16, 16.68, 15.84, 15.88, 14.36, 13.8, 13.56, 13.2, 12.96, 11.48, 11.16, 10.92, 10.96, 11.0, 11.04, 11.04, 11.04, 11.08, 11.48, 11.92, 12.24, 12.44, 13.64, 13.72, 13.8, 13.6, 14.12, 14.52, 15.04, 16.4, 17.64, 18.56, 18.84, 19.04, 19.64, 19.76, 20.24, 20.96, 21.16, 23.24, 23.6, 23.76, 23.76, 23.4, 23.12, 22.96, 22.76, 21.64, 21.48, 21.36, 20.92, 20.4, 20.72, 21.76, 21.44, 21.36, 21.2, 22.76, 23.68, 24.72, 25.04, 24.92, 25.28, 26.16, 24.88, 25.6, 28.08, 28.36, 28.28, 28.16, 28.08, 28.08, 28.04, 28.08, 28.44, 30.72, 30.48, 29.96, 28.6, 27.72, 28.6, 28.44, 27.56, 26.88, 25.76, 25.72, 25.8, 25.12, 25.12, 25.52, 24.96, 25.76, 26.68, 27.64, 28.4, 28.92, 29.16, 29.76, 30.72, 30.96, 29.68, 31.0, 32.6, 33.68, 34.8, 33.4, 34.92, 36.48, 37.72, 37.96, 38.36, 38.52, 40.44, 41.72, 42.68, 44.24, 41.68, 42.52, 43.08, 45.36, 45.2, 45.56, 46.76, 46.4, 47.32, 47.56, 53.76, 53.04, 54.84, 54.92, 60.32, 59.92, 58.64, 57.76, 58.04, 58.92, 60.32, 58.8, 57.2, 57.08, 56.52, 57.24, 61.64, 67.76, 66.44, 67.72, 68.16, 68.6, 69.72, 70.68, 75.64, 69.2, 70.32, 68.24, 72.4, 67.48, 66.56, 66.56, 66.72, 66.28, 65.96, 65.04, 66.2, 66.8, 72.24, 71.6, 71.04, 66.44, 62.12, 63.4, 62.12, 61.76, 60.08, 59.32, 57.64, 53.32, 53.24, 51.72, 51.76, 47.4, 47.4, 54.04, 53.6, 54.44, 57.44, 57.48, 57.72, 58.52, 59.08, 52.96, 54.4, 56.0, 54.76, 55.84, 54.0, 54.92, 56.0, 56.76, 56.92, 56.76, 55.64, 55.56, 56.6, 57.36, 57.56, 59.36, 52.72, 52.92, 55.04, 51.92, 52.36, 58.68, 61.24, 66.92, 67.6, 67.6, 66.92, 67.2, 65.24, 65.36, 64.96, 65.28, 65.68, 65.12, 65.28, 65.12, 64.8, 63.28, 61.64, 60.96, 59.92, 66.72, 70.12, 70.0, 71.48, 73.12, 67.52, 63.32, 57.52, 58.4, 57.6, 58.2, 64.88, 63.92, 67.16, 67.52, 66.92, 67.64, 69.4, 71.72, 72.92, 73.84, 75.88, 77.6, 78.36, 77.64, 76.2, 73.6, 71.6, 70.76, 68.92, 69.08, 68.84, 68.56, 66.44, 65.96, 68.72, 62.76, 68.92, 65.44, 65.32, 65.6, 66.56, 65.4, 63.88, 63.8, 66.92, 66.08, 66.44, 66.68, 73.16, 74.6, 73.84, 73.0, 73.68, 72.68, 73.2, 74.48, 74.4, 74.6, 74.52, 76.8, 76.12, 70.44, 73.36, 79.8, 78.72, 76.48, 77.52, 79.04, 79.52, 77.08, 78.12, 78.08, 81.64, 75.32, 71.16, 71.92, 75.4, 75.6, 76.72, 76.0, 76.56, 82.72, 89.36, 90.56, 85.36, 85.44, 85.64, 83.32, 83.32, 84.72, 86.48, 85.72, 84.28, 83.6, 86.84, 86.72, 86.32, 82.2, 82.72, 80.36, 79.56, 82.76, 84.04, 83.88, 83.24, 81.64, 76.44, 69.76, 68.84, 74.04, 74.2, 72.76, 72.16, 66.52, 66.16, 66.52, 68.16, 69.84, 70.28, 66.68, 66.16, 65.76, 65.84, 66.44, 72.96, 74.52, 69.28, 73.6, 75.72, 77.48, 82.4, 85.52, 88.36, 89.32, 84.44, 87.16, 89.76, 93.44, 93.52, 95.56, 96.44, 98.44, 100.4, 106.08, 110.2, 114.0, 120.08, 126.16, 131.36, 131.36, 136.48, 140.04, 140.04, 143.96, 148.48, 150.04, 152.12, 155.96, 157.28, 155.44, 152.48, 153.92, 155.08, 160.64, 164.48, 167.96, 170.44, 172.76, 167.8, 167.56, 169.56, 163.0, 156.44, 151.2, 149.56, 149.56, 149.68, 149.68, 148.32, 148.32, 148.32, 146.52, 144.72, 148.8, 155.52, 161.96, 164.84, 166.52, 166.52, 161.24, 161.24, 161.04, 161.04, 166.0, 167.8, 167.8, 174.36, 180.92, 186.16, 187.8, 187.8, 186.76, 186.76, 188.12, 182.76, 178.72, 173.36, 168.72, 164.12, 164.12, 160.08, 160.08, 153.76, 148.04, 153.32, 151.04, 145.08, 139.68, 139.68, 138.72, 138.72, 136.12, 133.88, 133.88, 128.88, 123.4, 126.0, 126.0, 126.0, 125.0, 123.76, 130.92, 134.88, 139.48, 134.2, 137.2, 137.2, 137.68, 143.4, 143.4, 145.68, 151.84, 157.24, 155.84, 155.8, 155.8, 156.72, 158.96, 156.8, 159.24, 164.72, 163.84, 163.84, 163.84, 169.96, 171.8, 171.48, 173.96, 173.96, 179.24, 180.48, 175.04, 174.4, 174.4, 174.4, 172.68, 170.92, 170.92, 172.32, 173.32, 173.32, 174.24, 174.24, 176.4, 177.84, 177.84, 178.72, 178.72, 174.44, 174.48, 175.4, 174.36, 172.32, 170.28, 169.2, 167.76, 169.2, 175.68, 175.68, 175.68, 177.4, 175.56, 173.36, 169.84, 166.84, 166.84, 163.76, 160.52, 157.76, 158.88, 153.72, 151.28, 147.36, 148.68, 148.88, 148.04, 147.0, 144.16, 146.2, 145.16, 143.24, 142.44, 140.72, 140.72, 140.72, 139.68, 136.72, 134.4, 133.48, 134.08, 130.6, 131.56, 133.28, 129.84, 129.84, 134.4, 135.88, 137.44, 138.16, 136.6, 137.72, 140.12, 145.0, 142.08, 143.68, 143.52, 148.32, 149.04, 149.04, 148.28, 146.88, 151.92, 156.44, 160.36, 162.76, 163.96, 163.4, 163.08, 169.28, 169.0, 168.84, 168.84, 167.48, 168.44, 169.2, 170.52, 166.24, 163.48, 164.64, 161.72, 163.28, 163.28, 159.76, 158.76, 159.52, 161.96, 162.32, 157.8, 158.32, 153.12, 155.4, 156.56, 155.12, 152.08, 150.04, 150.8, 151.76, 152.04, 149.36, 147.28, 144.68, 145.88, 145.64, 145.48, 148.92, 149.52, 146.6, 148.28, 145.8, 142.6, 139.2, 135.8, 136.12, 132.88, 134.24, 131.04, 129.88, 130.04, 129.32, 127.88, 125.0, 122.64, 123.0, 123.16, 122.44, 121.76, 123.24, 125.08, 124.08, 121.2, 118.0, 116.44, 114.96, 113.52, 112.72, 112.4, 114.96, 113.44, 112.92, 112.8, 112.48, 114.16, 113.84, 114.04, 114.4, 112.64, 111.32, 110.08, 109.6, 110.16, 111.2, 110.04, 109.48, 108.48, 107.76, 109.0, 112.04, 115.0, 118.0, 119.52, 120.84, 121.0, 124.16, 126.04, 127.32, 128.12, 128.32, 128.32, 130.6, 131.44, 134.24, 134.84, 135.32, 139.6, 142.16, 143.96, 145.68, 147.4, 149.52, 152.28, 155.12, 154.8, 154.28, 155.36, 153.48, 154.4, 156.24, 157.76, 158.52, 160.88, 163.6, 165.16, 165.96, 162.08, 161.36, 163.16, 162.08, 164.24, 164.24, 157.84, 159.2, 159.24, 157.56, 158.12, 157.48, 154.72, 154.52, 155.44, 155.48, 158.36, 159.84, 158.08, 158.96, 159.2, 158.6, 157.92, 156.88, 156.08, 159.64, 162.52, 162.12, 165.36, 167.04, 166.0, 171.92, 171.52, 172.0, 173.0, 174.24, 175.72, 178.32, 178.84, 178.52, 178.64, 176.84, 176.84, 178.6, 178.76, 179.64, 180.0, 179.96, 180.36, 183.8, 185.32, 185.72, 185.72, 186.64, 186.28, 180.88, 181.6, 181.16, 181.44, 182.12, 182.12, 182.12, 182.6, 184.04, 185.32, 186.0, 189.28, 189.28, 189.28, 189.24, 189.24, 189.8, 190.52, 188.56, 188.48, 188.56, 185.88, 186.32, 185.92, 186.12, 191.84, 188.84, 184.68, 184.92, 182.76, 182.76, 181.92, 181.96, 181.96, 180.48, 180.48, 180.48, 180.48, 180.48, 180.52, 180.52, 180.52, 180.52, 183.12, 183.2, 183.2, 185.88, 185.88, 186.28, 186.44, 187.16, 190.4, 195.4, 195.4, 197.56, 197.56, 198.4, 196.0, 196.0, 197.48, 197.48, 196.76, 196.76, 196.76, 196.76, 196.68, 190.44, 188.96, 187.08, 185.6, 184.04, 183.04, 182.2, 181.08, 180.76, 179.24, 177.0, 175.84, 174.0, 172.16, 170.4, 168.68, 169.6, 168.28, 166.32, 164.76, 164.4, 162.6, 161.2, 159.96, 158.56, 163.0, 163.48, 163.76, 164.24, 164.8, 164.76, 165.2, 165.36, 164.92, 166.44, 167.92, 163.52, 160.44, 158.68, 160.44, 162.16, 163.76, 165.08, 167.04, 168.6, 169.68, 171.48, 172.88, 174.12, 175.6, 169.88, 170.88, 172.48, 170.12, 168.56, 169.6, 169.68, 169.8, 169.32, 169.32, 169.76, 174.56, 178.88, 181.96, 181.96, 181.12, 179.52, 178.28, 175.04, 173.2, 172.04, 170.4, 169.44, 168.48, 167.68, 175.2, 172.24, 169.72, 171.04, 171.2, 168.64, 167.92, 166.84, 165.72, 163.72, 161.64, 159.76, 158.04, 157.04, 155.88, 154.68, 154.88, 154.44, 156.0, 155.56, 155.8, 155.8, 154.96, 154.28, 153.2, 152.04, 153.8, 155.08, 156.72, 158.04, 159.68, 160.36, 161.64, 162.56, 164.12, 165.88, 168.52, 169.76, 170.96, 171.44, 173.24, 174.64, 176.12, 177.8, 179.88, 180.52, 182.16, 183.96, 185.6, 187.0, 188.16, 188.68, 187.72, 188.12, 187.68, 187.4, 186.72, 186.56, 186.76, 186.16, 185.64, 185.44, 185.68, 185.2, 185.56, 185.16, 184.12, 184.04, 183.28, 183.48, 183.76, 183.76, 181.56, 181.04, 179.72, 179.32, 180.0, 182.2, 182.2, 183.72, 184.92, 185.96, 186.76, 188.0, 189.04, 190.08, 190.28, 191.12, 190.84, 190.96, 190.92, 191.96, 192.24, 193.0, 192.96, 192.96, 192.96, 194.44, 194.96, 196.76, 197.16, 197.16, 197.16, 197.16, 197.16, 197.16, 195.76, 193.84, 193.84, 193.48, 193.4, 193.24, 192.16, 193.08, 192.68, 193.36, 193.36, 193.36, 193.36, 193.4, 193.4, 193.4, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 195.52, 197.44, 197.44, 197.8, 195.8, 195.92, 197.0, 197.16, 197.76, 197.76, 197.12, 195.6, 194.4, 193.0, 191.64, 190.44, 189.04, 187.72, 186.68, 185.24, 184.28, 182.2, 181.4, 180.48, 178.88, 178.48, 177.08, 175.8, 175.2, 176.48, 175.48, 174.88, 173.72, 172.56, 172.36, 173.0, 173.56, 173.84, 174.48, 175.24, 176.44, 175.92, 176.36, 177.32, 178.44, 178.96, 181.04, 181.48, 181.52, 182.12, 182.28, 183.16, 183.68, 183.84, 184.64, 185.44, 185.64, 186.0, 186.4, 186.6, 186.6, 187.56, 188.48, 186.04, 186.64, 186.64, 184.52, 185.4, 185.48, 185.8, 184.76, 184.44, 184.8, 185.68, 186.68, 186.92, 187.44, 188.2, 188.64, 188.76, 189.0, 189.4, 190.2, 190.96, 190.96, 190.96, 190.96, 190.96, 194.16, 194.16, 193.32, 197.36, 195.92, 195.92, 192.12, 191.04, 191.36, 191.36, 191.36, 190.2, 190.2, 190.2, 190.2, 190.2, 190.2, 190.2, 190.2, 190.2, 190.2, 189.72, 187.48, 187.48, 180.04, 176.2, 173.52, 174.36, 170.04, 163.96, 159.48, 158.92, 158.04, 153.24, 145.84, 140.76, 137.56, 134.56, 127.12, 122.64, 119.16, 115.4, 107.92, 100.44, 95.8, 92.28, 85.32, 81.72, 76.52, 83.96, 87.8, 90.48, 90.48, 94.68, 97.56, 99.56, 99.16, 99.68, 104.48, 109.64, 111.24, 115.6, 118.08, 124.6, 129.08, 127.88, 130.84, 133.6, 138.2, 142.84, 146.36, 153.8, 159.64, 162.24, 162.24, 162.24, 158.6, 154.72, 154.84, 159.48, 161.96, 166.72, 169.64, 169.64, 171.88, 175.36, 175.36, 175.88, 176.8, 176.8, 181.48, 182.28, 187.0, 189.88, 189.88, 189.88, 189.88, 189.88, 192.48, 192.48, 191.16, 194.8, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 197.32, 189.76, 191.96, 184.36, 184.36, 184.36, 184.36, 184.36, 184.36, 184.36, 184.36, 181.36, 180.96, 180.96, 180.96, 179.04, 178.68, 178.68, 178.68, 178.68, 178.68, 178.68, 178.68, 178.68, 176.04, 174.36, 181.92, 179.32, 179.44, 176.36, 176.36, 176.08, 176.08, 176.08, 176.08, 176.08, 179.08, 179.48, 179.48, 179.48, 181.4, 181.76, 181.76, 181.76, 181.76, 177.32, 173.12, 169.48, 165.12, 162.92, 161.0, 160.24, 159.52, 164.68, 166.6, 165.36, 161.24, 158.12, 155.12, 151.56, 148.32, 141.04, 136.28, 135.92, 135.92, 135.0, 132.84, 132.04, 132.04, 130.28, 134.72, 138.92, 142.56, 146.92, 151.76, 155.36, 156.12, 159.92, 162.24, 163.4, 164.64, 169.04, 172.16, 175.16, 178.72, 181.96, 189.24, 194.0, 194.36, 194.36, 195.28, 197.44, 198.24, 198.24, 200.0, 200.0, 196.88, 193.88, 192.88, 192.88, 192.88, 192.88, 192.88, 192.88, 192.88, 192.88, 192.88, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 189.16, 192.28, 195.28, 196.28, 196.28, 196.28, 196.28, 196.28, 196.28, 188.92, 181.56, 179.12, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.84, 182.08, 182.08, 182.08, 182.08, 182.08, 182.08, 182.08, 182.08, 181.68, 181.68, 189.04, 196.4, 198.84, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 198.16, 197.56, 197.48, 196.04, 194.4, 192.08, 190.32, 188.6, 187.2, 179.68, 176.4, 173.32, 173.32, 173.32, 171.12, 170.08, 168.4, 166.08, 164.28, 162.08, 159.88, 152.4, 149.52, 146.96, 145.16, 144.48, 143.96, 144.8, 142.16, 140.0, 140.04, 139.04, 137.48, 136.44, 141.76, 142.84, 143.52, 140.56, 138.28, 136.96, 131.2, 125.56, 124.24, 123.28, 123.12, 124.16, 131.12, 134.0, 136.56, 138.36, 138.96, 137.08, 134.76, 138.84, 141.8, 142.2, 140.24, 139.88, 142.32, 144.52, 144.52, 146.92, 149.88, 150.12, 151.08, 156.48, 161.76, 165.4, 163.24, 161.0, 157.6, 152.36, 147.24, 142.84, 140.6, 136.64, 136.6, 133.92, 129.64, 125.12, 122.28, 127.0, 130.64, 127.72, 127.72, 125.32, 121.12, 121.12, 123.16, 123.28, 125.36, 125.36, 124.44, 126.04, 126.52, 128.04, 128.16, 128.04, 132.44, 132.48, 136.52, 139.56, 144.56, 144.64, 150.0, 154.72, 153.6, 149.68, 152.6, 152.0, 153.96, 150.64, 145.12, 141.72, 139.0, 138.08, 140.12, 141.04, 144.36, 148.48, 151.52, 157.16, 157.16, 149.56, 146.52, 146.52, 146.52, 146.52, 150.2, 150.2, 145.76, 146.88, 150.8, 148.48, 147.92, 150.96, 158.4, 163.92, 166.4, 171.52, 172.04, 172.04, 170.12, 168.92, 168.76, 168.76, 164.0, 169.24, 175.36, 179.8, 179.8, 179.8, 179.8, 179.68, 179.68, 184.12, 184.12, 184.12, 181.44, 182.6, 182.6, 177.08, 174.2, 175.12, 175.16, 175.56, 175.56, 177.48, 178.68, 178.84, 178.84, 183.6, 181.92, 177.24, 174.24, 174.24, 174.24, 174.24, 174.88, 174.88, 174.88, 174.88, 169.72, 169.72, 168.76, 168.76, 174.36, 177.24, 177.24, 177.24, 177.24, 175.84, 175.68, 175.68, 175.68, 172.76, 171.44, 173.12, 177.92, 181.72, 181.72, 181.72, 181.72, 181.72, 181.72, 181.72, 181.72, 183.0, 188.0, 188.96, 188.96, 188.96, 188.96, 188.96, 187.64, 187.64, 187.88, 188.04, 188.04, 185.56, 188.48, 189.8, 189.8, 191.16, 188.96, 188.96, 188.96, 188.04, 187.88, 187.88, 187.88, 181.44, 178.0, 172.16, 167.32, 166.08, 166.08, 166.08, 166.08, 167.4, 167.4, 168.56, 168.56, 168.56, 165.8, 165.8, 165.8, 165.8, 165.8, 168.0, 168.0, 168.0, 168.92, 165.12, 165.12, 165.12, 171.56, 178.88, 184.72, 189.56, 190.8, 190.8, 188.2, 188.2, 188.2, 188.2, 188.2, 188.2, 188.2, 193.44, 193.44, 191.88, 191.88, 191.88, 191.88, 191.88, 191.88, 189.12, 193.08, 193.08, 193.08, 193.08, 193.08, 193.08, 193.08, 192.0, 192.0, 193.16, 193.16, 193.16, 193.16, 193.16, 193.16, 193.16, 193.16, 193.16, 191.16, 191.16, 191.16, 191.16, 186.64, 183.4, 186.16, 186.16, 186.16, 186.16, 186.16, 186.16, 186.16, 186.16, 187.24, 187.24, 188.68, 188.68, 188.68, 188.68, 188.68, 188.68, 188.68, 188.68, 188.68, 192.24, 192.24, 192.24, 192.24, 196.76, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]


if __name__ == '__main__':
    plot_two_lines('Reward Per Episode', 'Episode', 'Reward', episodes, 'Reward', episode_rewards, 'Running Avg Reward', running_average_reward)





