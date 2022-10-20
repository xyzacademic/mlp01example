import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

Download = False

c1 = 3
c2 = 8
if Download:
    os.system(
        'wget -c https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip')
    os.system(
        'wget -c https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip')
    os.system('wget -c https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip')
    os.mkdir('data')
    os.makedir('data/train_binary')
    os.makedir('data/test_binary')
    os.system('unzip data/GTSRB_Final_Training_Images.zip -d data')
    os.system('unzip data/GTSRB_Final_Test_Images.zip -d data')
    os.system('unzip data/GTSRB_Final_Test_GT.zip -d data')
    os.system('mv GTSRB/Final_Test/Images/*.ppm data/test')
    os.system(
        'wget https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/master/signnames.csv -P data')
    # os.system('cp -r GTSRB/Final_Training/Images/0000%d train_binary/' % c1)
    # os.system('cp -r GTSRB/Final_Training/Images/0000%d train_binary/' % c2)

# train
for j in range(10):
    df = pd.read_csv('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/'
                     'Final_Training/Images/0000%d/GT-0000%d.csv' % (j, j),
                     sep=';')
    if not os.path.exists('/home/y/yx277/research/ImageDataset/GTSRB/data/train_mc/0000%d' % j):
        os.makedirs('/home/y/yx277/research/ImageDataset/GTSRB/data/train_mc/0000%d' % j)
    for i in df.index:
        file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
        img = plt.imread(
            os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/Final_Training/Images/0000%d' % j, file))
        img = img[y1:y2, x1:x2]
        plt.imshow(img)
        plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/train_mc/0000%d/%s.png' % (j, file[:-4]))
        plt.close()




# df = pd.read_csv('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/Final_Training/Images/00003/GT-00003.csv',
#                  sep=';')
#
# for i in df.index:
#     file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
#     img = plt.imread(
#         os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/Final_Training/Images/00003', file))
#     img = img[y1:y2, x1:x2]
#     plt.imshow(img)
#     plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/train_binary/00003/%s.png' % file[:-4])
#     plt.close()
#
# df = pd.read_csv('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/Final_Training/Images/00008/GT-00008.csv',
#                  sep=';')
#
# for i in df.index:
#     file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
#     img = plt.imread(
#         os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/GTSRB/Final_Training/Images/00008', file))
#     img = img[y1:y2, x1:x2]
#     plt.imshow(img)
#     plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/train_binary/00008/%s.png' % file[:-4])
#     plt.close()

# test


df = pd.read_csv('/home/y/yx277/research/ImageDataset/GTSRB/data/GT-final_test.csv', sep=';')

for c1 in range(10):
    df1 = df[df['ClassId'] == c1]
    if not os.path.exists('/home/y/yx277/research/ImageDataset/GTSRB/data/test_mc/0000%d' % c1):
        os.makedirs('/home/y/yx277/research/ImageDataset/GTSRB/data/test_mc/0000%d' % c1)
    for i in df1.index:
        file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
        img = plt.imread(os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/test', file))
        img = img[y1:y2, x1:x2]
        plt.imshow(img)
        plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/test_mc/0000%d/%s.png' % (c1, file[:-4]))
        plt.close()


# df1 = df[df['ClassId'] == c1]
# df2 = df[df['ClassId'] == c2]
#
# for i in df1.index:
#     file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
#     img = plt.imread(os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/test', file))
#     img = img[y1:y2, x1:x2]
#     plt.imshow(img)
#     plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/test_binary/00003/%s.png' % file[:-4])
#     plt.close()
#
# for i in df2.index:
#     file, width, height, x1, y1, x2, y2, label = df.iloc[i].values
#     img = plt.imread(os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/test', file))
#     img = img[y1:y2, x1:x2]
#     plt.imshow(img)
#     plt.savefig('/home/y/yx277/research/ImageDataset/GTSRB/data/test_binary/00008/%s.png' % file[:-4])
#     plt.close()
#
# df1_files = df1['Filename'].values.tolist()
# df2_files = df2['Filename'].values.tolist()
#
# for file in df1_files:
#     shutil.copy(os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/test', file),
#                 '/home/y/yx277/research/ImageDataset/GTSRB/data/test_binary/0000%d/%s' % (c1, file))
# for file in df2_files:
#     shutil.copy(os.path.join('/home/y/yx277/research/ImageDataset/GTSRB/data/test', file),
#                 '/home/y/yx277/research/ImageDataset/GTSRB/data/test_binary/0000%d/%s' % (c2, file))