from tabnanny import filename_only
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import layers
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from PIL import Image
#from hashlib import sha256
#import time
from glob import glob
import os
import pathlib


class Augment():
    '''Generates a data augmentation of specified class

    Args:
        class_augment: string of the label to augment.
        N: int target of the class. It will generate an extra N-n samples,
            where n is the current amount of samples of the original class.
        share_test: float percentage of class samples to save for testing.
        share_test_original: bool to denote if the test share considers only
            original samples quantity or also augmented ones.
        output_dir: string path to the output directory

    Returns:
        N-n, the values of generated samples
       print("metadata not generated yet") Also, writes in output folder the augmented images, the metadata of
            the augmented images and a test.csv with the path to the original
            files to be used for testing (and whose were not seen for augmentation)
            The test.csv file only saves original images, no mater of
            share_test_original value.
    '''
    def __init__(self, class_augment='mel', N=1120, share_test=0.10, share_test_original=True,
                 output_dir=os.path.join('..', '..', 'raw_data', 'augment'),
                 path_to_metadata=os.path.join('..', '..', 'raw_data', 'HAM10000_metadata.csv')
                 ):
        self.class_augment = class_augment
        self.N = N
        self.share_test = share_test
        self.share_test_original = share_test_original
        self.output_dir = output_dir
        self.path_to_metadata = path_to_metadata
        #load data and corresponding image paths
        df = pd.read_csv(self.path_to_metadata)

        df_class = df[df.dx == class_augment]
        count_test = int(share_test*len(df_class)) if share_test_original else int(share_test*N)

        #this hard-coded path should be avoided. It's ok for now given that
        #data augmentation will be made offline from a local environment
        base_skin_dir = os.path.join('..', '..', 'raw_data')
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x[6:]
                            for x in glob(os.path.join(base_skin_dir,'**', '*.jpg'),recursive=True)}
        df_class['path'] = df_class['image_id'].map(imageid_path_dict.get)


        #group cases by same lesion_id and count for repetition
        df_class_grouped = df_class.groupby('lesion_id')[['image_id']].count()
        #just keep those samples that are unique
        df_class_grouped_unique = df_class_grouped[df_class_grouped['image_id'] == 1]
        #and select the necessary of those uniques in order to keep safe for test partition
        self.data_to_test = df_class[df_class.lesion_id.isin(df_class_grouped_unique.sample(count_test).reset_index()['lesion_id'].to_list())]
        self.data_to_augment = df_class[~df_class.index.isin(self.data_to_test.index)]
        #self.data_to_test = df_class.sample(count_test) #to delete if upper code works

        #create in output_dir/<class_augment_test.csv>
        self._generate_test_csv()

        #create in output_dir/<class_augment_metadata.csv>
        self._generate_augment_metadata_csv()
        #create in output_dir/<class_augment>/imgs_

    def _generate_test_csv(self):
        try:
            pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            generated_test_csv = os.path.join(self.output_dir, self.class_augment + "_test.csv")
            self.data_to_test.to_csv(generated_test_csv, index=False)
            print(f"Generated test csv at: {generated_test_csv}")
        except OSError:
            print ("Creation of the directory %s failed" % self.output_dir)


    def _generate_augment_metadata_csv(self):
        size_available_data = len(self.data_to_augment)
        #how many rows should I generate?
        #take them from 'the rest' of the dataset w/o test samples
        #generate: N - test - the_rest
        to_generate = self.N - len(self.data_to_test) - size_available_data

        list_of_rows = []
        for i in range(0, to_generate):
            if i%100==0:
                print(f"generatig {i}th image")
            i_loc = i % size_available_data
            i_count = i // size_available_data
            row_i = self.data_to_augment.iloc[i_loc]
            row_i['image_id'] = row_i['image_id'] + "_aug_" + str(i_count)
            head, _ = os.path.split(row_i['path'])
            head, ham_part = os.path.split(head)
            filename_jpg = row_i['image_id'] + ".jpg"
            original_path = row_i['path']
            row_i['path'] = os.path.join(head, 'augment', filename_jpg)
            list_of_rows.append(row_i)
            PIL_image = self._generate_augmented_image(path_to_image=os.path.join('..','..', original_path))
            PIL_image.save(os.path.join('..', '..', row_i['path']))

        pd.DataFrame(list_of_rows).to_csv(os.path.join(self.output_dir, self.class_augment + "_augmented.csv"), index=False)
        print(f"generated csv and {to_generate} images. Done.")

    def _generate_augmented_image(self, path_to_image):
        datagen = ImageDataGenerator(
                            rotation_range=15,
                            horizontal_flip=True,
                            vertical_flip=True,
                            shear_range=0.0,
                            fill_mode = 'reflect',
                            width_shift_range = [0, 0.05],
                            height_shift_range = [0, 0.05]
        )

        # generate samples and plot
        img_ker = load_img(path_to_image)
        img_np = img_to_array(img_ker)
        sample = np.expand_dims(img_np, 0)

        it = datagen.flow(sample, batch_size=1)
        im = Image.fromarray(img_np.astype('uint8'))
        batch = it.next()
        # convert to unsigned integers for viewing
        image_np = batch[0].astype('uint8')
        #plt.imshow(image)
        PIL_image = Image.fromarray(image_np.astype('uint8'), 'RGB')
        return PIL_image


if __name__ == '__main__':
    augment = Augment(N=6705)
    #augment = Augment(N=1300)
