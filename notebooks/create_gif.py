import imageio
import glob

images = []
filenames = glob.glob("../test_samples/*.jpg")
gif_name = '../z_backup/test.gif'

#sort by number
filenames = sorted(filenames, key=lambda x: int(x.split('_')[2].split('.')[0]))
print(filenames)

for filename in sorted(filenames):
    images.append(imageio.imread(filename))
imageio.mimsave(gif_name, images)
