import csv
import tensorflow as tf
import transformers as tf_transformers
import tensorflow_datasets as tfds
import os
data = []
path = "MEDIQA-Chat-Training-ValidationSets-Feb-10-2023"
Task = "TaskA"
File_train=Task+'-TrainingSet.csv'
File_val = Task+'-ValidationSet.csv'
path_task = os.path.join(path,Task)
train_file = os.path.join(path_task,File_train)
val_file = os.path.join(path_task,File_val)
# Load the medical dialogue text summarization dataset

# TODO
train_dataset, train_dataset_info = dataset("File_train", split="train", data_dir=train_file,with_info=True)
val_dataset, val_dataset_info=dataset("File_train", split="train", data_dir=train_file,with_info=True)

tokenizer = tf_transformers.PegasusTokenizer.from_pretrained("google/pegasus-cased")
# Preprocess the data
def preprocess_fn(example,tokenizer):
    # Tokenize the input text and target summary
    # tokenizer = tfds.features.text.Tokenizer()
    input_text = tokenizer.tokenize(example["dialogue"])
    target_summary = tokenizer.tokenize(example["note"])

    # Truncate the input text and target summary to a maximum length
    MAX_LENGTH = 1024
    input_text = input_text[:MAX_LENGTH]
    target_summary = target_summary[:MAX_LENGTH]

    # Pad the input text and target summary to the maximum length
    input_text = tf.keras.preprocessing.sequence.pad_sequences(input_text, maxlen=MAX_LENGTH, padding="post")
    target_summary = tf.keras.preprocessing.sequence.pad_sequences(target_summary, maxlen=MAX_LENGTH, padding="post")
    #TODO
    # 后面要处理长度问题！！
    # Return the preprocessed data
    return input_text, target_summary

train_dataset = train_dataset.map(preprocess_fn)
val_dataset =val_dataset.map(preprocess_fn)


model =tf.keras.models.load_model("pegasus_model.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save the trained model
model.save("pegasus_model_fine_tuned.h5")