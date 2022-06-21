import tensorflow as tf
import os

# LOAD A DATASET #

# We'll use the CoNLL-2003 dataset. From my research, it seems to be quite popular with NER problems, and has the tags
# that are relavent to us (names, organization, locations)
from datasets import load_dataset_builder
ds_builder = load_dataset_builder("conll2003")
#print(ds_builder.info.description)
#print(ds_builder.info.features)

from datasets import load_dataset
dataset_train = load_dataset("conll2003", split="train")
dataset_vali = load_dataset("conll2003", split="validation")
dataset_test = load_dataset("conll2003", split="test")

# PREPCROCESSING #
# We note that the dataset is very nice and already tokenized/tagged
# In the dataset, we're mostly focused on ner_tags.
# The tags are represented as numbers (0 to 8), and correspond to tags as follows:
# {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
# Since we wish to pick out the names, organization, and locations, we'd focus on tags with the suffix PER, ORG, and LOC

# From my research, it seems that a LSTM RNN would work well for this problem, as we'd want a temporal element to
# our model (i.e. words from earlier in the sentence should affect whether this current word is predicted to be
# a name, organization or a location).

# To use this model, we would need to convert our textual data to integers.
# We would need to map each character to an integer (similar to how the tags are mapped to integers).

# The following function would do some pre-processing by splitting the words into characters, while keeping the tags
def split_tokens(dataset_row):
    # The function considers one row of the dataset
    # and outputs a tuple of inputs to the model and outputs
    
    # Gathering the relavent data from the dataset
    tokens = dataset_row["tokens"]
    pos_tags = dataset_row["pos_tags"]
    chunk_tags = dataset_row["chunk_tags"]
    ner_tags = dataset_row["ner_tags"]
    
    # Initializing lists
    output_tokens = []
    output_pos_tags = []
    output_chunk_tags = []
    output_ner_tags = []
    
    # Iterating over the data
    for token,pos_tags,chunk_tags,ner_tags in zip(tokens, pos_tags, chunk_tags, ner_tags):
        # Splitting the tokens into characters
        token_split = [char for char in token]
        
        # Multiplying the tags to match them with the characters
        pos_tag_split = [[pos_tags] * len(token)]
        chunk_tag_split = [[chunk_tags] * len(token)]
        ner_tag_split = [[ner_tags] * len(token)]
        
        # Adding it to the lists
        output_tokens.extend(token_split)
        output_pos_tags.extend(pos_tag_split)
        output_chunk_tags.extend(chunk_tag_split)
        output_ner_tags.extend(ner_tag_split)

    # flattens output
    output_ner_tags = sum(output_ner_tags, [])

    # converts strings to ASCII integers
    output_tokens = list(map(ord, output_tokens))
    
    # Return the relavent tags. I've taken out the chunk/pos information as we aren't using it right now
    return((output_tokens, output_ner_tags))#output_pos_tags, output_chunk_tags, output_ner_tags])

# Format the data using the split_tokens function
formatted_train = [split_tokens(row) for row in dataset_train]
formatted_vali = [split_tokens(row) for row in dataset_vali]
formatted_test = [split_tokens(row) for row in dataset_test]

# It would be useful to know the number of unique words we have for later
all_words = []
for i in range(0, len(dataset_train)):
    all_words.append(dataset_train[i]["tokens"])
for i in range(0, len(dataset_vali)):
    all_words.append(dataset_vali[i]["tokens"])
for i in range(0, len(dataset_test)):
    all_words.append(dataset_test[i]["tokens"])

all_words = sum(all_words, [])
vocab_size = len(list(set(all_words)))

# Creating a generator to get data the way TensorFlow likes it
def gen_train():
    for row in formatted_train:
        yield row[0], row[1]
def gen_vali():
    for row in formatted_vali:
        yield row[0], row[1]
def gen_test():
    for row in formatted_test:
        yield row[0], row[1]

train_data = tf.data.Dataset.from_generator(gen_train,output_types=(tf.int32, tf.int32),output_shapes = ((None, None)))
vali_data = tf.data.Dataset.from_generator(gen_vali,output_types=(tf.int32, tf.int32),output_shapes = ((None, None)))
test_data = tf.data.Dataset.from_generator(gen_test,output_types=(tf.int32, tf.int32),output_shapes = ((None, None)))

# I'm not sure how padding the number this way (i.e. with the number "9") will affect the model.
# The default is to pad it with "0", but since 0 is a meaningful value in our dataset (it corresponds to NULL in ASCII,
# and corresponds to an O tag), I thought it would be a problem if I padded it with 0s.
batch_size = 128
train_data_batch = train_data.padded_batch(batch_size, padding_values= 9,padded_shapes=([None], [None]), drop_remainder= True)
vali_data_batch = vali_data.padded_batch(batch_size, padding_values= 9,padded_shapes=([None], [None]), drop_remainder= True)
test_data_batch = test_data.padded_batch(batch_size, padding_values= 9,padded_shapes=([None], [None]), drop_remainder= True)

# BUILDING THE MODEL #

# From my looking things up, it seems like the architecture is going to look like:
# 1) an embedding layer
# 2) a hidden layer of RNN nodes
# 3) output layer
# It seems that the embedding layer is important because it reduces the dimensions of the input
# However my understanding is probably a bit spotty here

def model_builder(nodes, input_dim, output_dim, emb_output_dim, batch_size):
    # note: it seems like the input_dim for embedding layer is the vocabulary size
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim, emb_output_dim, batch_input_shape=[batch_size, None],mask_zero=True),
        tf.keras.layers.LSTM(nodes,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(output_dim)
    ])
    return model

model = model_builder(nodes= 1024, input_dim = (vocab_size + 1), output_dim= 10, emb_output_dim=256, batch_size= batch_size)

# We have more than 2 classes, and targets that aren't one-hot encoded. As such we're going to use
# sparse categorical crossentropy as the loss. This is mostly from experience.

model.compile(optimizer='adam',
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

# Directory where the checkpoints will be saved
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# MODEL TRAINING #

# The number of epoch is set to 1 for run time purposes for now, but it could easily be increased if required.
epochs_num = 1

model.fit(train_data_batch, epochs= epochs_num, validation_data= vali_data_batch, callbacks= cp_callback)

# LOADING MODEL (for after I've trained it)#
#model.compile(optimizer='adam',
#              loss = tf.keras.losses.sparse_categorical_crossentropy,
#              metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])
#
#checkpoint_path = "training_1/cp.ckpt"
#model.load_weights(checkpoint_path)
#pred = model.predict(test_data_batch)

# PREDICTIONS #

# I've unfortunately sort of ran into a problem where the dimensions of my test batch and my training/vali batches don't
# match, and I can't seem to pinpoint how to fix it.

# However, from looking at the dataset and the model training metrics (i.e. the validation accuracy), we can probably
# infer that its performance is not stellar (validation accuracy in the last epoch was at 0.2294).
# There may also be a problem where our classes are not evenly distributed, as there are a lot more class 1's (O's) than
# there are of other classes. Thus our model may be good at predicting class 1's, but not as good when trying to
# predict the rest of the classes, of which we are more interested in.


