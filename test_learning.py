import os
import shutil
import random

import tensorflow as tf
from tensorflow.python.framework.ops import eager_run
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dense

from SiameseNetwork import SiameseNetwork,ContrastiveLoss,SiameseAccuracy

def CreateDataset(data_num=4000,dir_path=os.path.join("dogs-vs-cats","train","train")):
    #Make dataset directory which is train, validation, test
    dataset_path={
        "train"       : os.path.join("Temp","train"),
        "validation"  : os.path.join("Temp","validation"),
        "test"        : os.path.join("Temp","test")
    }

    for key in dataset_path.keys():
        os.makedirs(os.path.join(dataset_path[key],"dogs"),exist_ok=True)
        os.makedirs(os.path.join(dataset_path[key],"cats"),exist_ok=True)

    #Allocating images
    cats_files=["cat.{}.jpg".format(i) for i in range(data_num)]
    dogs_files=["dog.{}.jpg".format(i) for i in range(data_num)]

    train_num       = int(data_num  * 0.5)
    validation_num  = int(data_num  * 0.25)
    test_num        = int(data_num  - (train_num+validation_num))

    data={
        "train"       : {"cats":cats_files[:train_num], "dogs":dogs_files[:train_num]},
        "validation"  : {"cats":cats_files[train_num:train_num+validation_num], "dogs":dogs_files[train_num:train_num+validation_num]},
        "test"        : {"cats":cats_files[train_num+validation_num:], "dogs":dogs_files[train_num+validation_num:]}
    }

    for dataset_key in data.keys():
        for class_key in data[dataset_key].keys():
            for filename in data[dataset_key][class_key]:
                src=os.path.join(dir_path,filename)
                dst=os.path.join(dataset_path[dataset_key],class_key,filename)

                shutil.copyfile(src,dst)

    return dataset_path,data

def CreatePairs(dataset_path,data,create_pair_num=4000):
    temp={}

    for type,choice_num in zip(("train","validation","test"),(int(create_pair_num*0.5),int(create_pair_num*0.25),int(create_pair_num*0.25))):
        temp[type]=[]

        files=[
            (file_name,class_name) for file_name,class_name in zip(
                random.choices(data[type]["cats"],k=choice_num)+random.choices(data[type]["dogs"],k=choice_num),
                ["cats" for i in range(choice_num)]+["dogs" for _ in range(choice_num)]
            )
        ]

        temp[type]=[[],[],[]]
        for _ in range(choice_num):
            indices=random.sample(list(range(len(files))),2)
            
            temp[type][0].append(os.path.join(dataset_path[type],files[indices[0]][1],files[indices[0]][0]))
            temp[type][1].append(os.path.join(dataset_path[type],files[indices[1]][1],files[indices[1]][0]))
            temp[type][2].append(1. if files[indices[0]][1]==files[indices[1]][1] else 0.)
            
            del files[indices[0]]

            if indices[0]<indices[1]:
                del files[indices[1]-1]
            else:
                del files[indices[1]]

    return temp["train"],temp["validation"],temp["test"]

def CreateTFDataset(train_pairs,validation_pairs,test_pairs):
    def Generator(imageA_paths,imageB_paths,labels):
        for imageA_path,imageB_path,label in zip(imageA_paths,imageB_paths,labels):

            imageA=tf.io.read_file(imageA_path)
            imageB=tf.io.read_file(imageB_path)

            imageA=tf.io.decode_image(imageA)
            imageB=tf.io.decode_image(imageB)

            imageA=tf.image.resize(imageA,(224,224))
            imageB=tf.image.resize(imageB,(224,224))

            imageA=tf.cast(imageA,tf.float32)
            imageB=tf.cast(imageB,tf.float32)
            label=tf.cast(label,tf.float32)

            imageA/=225.
            imageB/=225.

            yield tf.stack([imageA, imageB]), tf.expand_dims(label,0)

    train=tf.data.Dataset.from_generator(
        Generator,
        args=train_pairs,
        output_signature=(
            tf.TensorSpec(shape=(2,224,224,3),dtype=tf.float32),
            tf.TensorSpec(shape=(1,),dtype=tf.float32)
        )
    )

    validation=tf.data.Dataset.from_generator(
        Generator,
        args=validation_pairs,
        output_signature=(
            tf.TensorSpec(shape=(2,224,224,3),dtype=tf.float32),
            tf.TensorSpec(shape=(1,),dtype=tf.float32)
        )
    )

    test=tf.data.Dataset.from_generator(
        Generator,
        args=test_pairs,
        output_signature=(
            tf.TensorSpec(shape=(2,224,224,3),dtype=tf.float32),
            tf.TensorSpec(shape=(1,),dtype=tf.float32)
        )
    )

    train=train.shuffle(512).batch(8)
    validation=validation.shuffle(512).batch(8)
    test=test.shuffle(512).batch(8)

    return train,validation,test

if __name__=="__main__":
    # ================================================ この部分は自身で実装する ================================================
    # train_pairs,validation_pairs,test_pairsは同じ形式のリスト
    # 形状は[[N],[N],[N]] (Nは任意の自然数)
    # 内容は[[imageAのpathを格納したリスト], [imageBのpathを格納したリスト], [imageAとimageBのペアのラベル(同じか否か)]]
    
    train_pairs,validation_pairs,test_pairs=CreatePairs(*CreateDataset())

    # ========================================================================================================================

    train,validation,test=CreateTFDataset(train_pairs,validation_pairs,test_pairs)

    margin=1.

    # EuclideanDistance + ContrastiveLoss を用いた場合
    sn=SiameseNetwork()
    sn.compile(optimizer="adagrad",loss=ContrastiveLoss(margin=margin),metrics=SiameseAccuracy(threshold=margin))

    print("<< Feauture Extraction >>")
    sn.fit(train,validation_data=validation,epochs=10)

    print("<< Finetuning >>")
    sn.trainable=True
    sn.fit(train,validation_data=validation,epochs=30)

    print("<< Evaluate >>")
    sn.evaluate(test)

    # FCで直接類似度を算出する場合
    cnn_model=tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet")
    cnn_model.trainable=False

    sn=SiameseNetwork(
        feauture_extraction_layers=[cnn_model],
        difference_layer=tf.keras.layers.Subtract(),
        calculation_similar_layers=[
            tf.keras.layers.Dense(1024,activation="relu"),
            tf.keras.layers.Dense(1,activation="sigmoid")
        ]
    )
    sn.compile(optimizer="adagrad",loss=ContrastiveLoss(margin=margin),metrics=SiameseAccuracy(threshold=margin))

    print("<< Feauture Extraction >>")
    sn.fit(train,validation_data=validation,epochs=10)

    print("<< Finetuning >>")
    sn.trainable=True
    sn.fit(train,validation_data=validation,epochs=30)

    print("<< Evaluate >>")
    sn.evaluate(test)
