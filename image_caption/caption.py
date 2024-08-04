import tensorflow as tf
import einops

from PIL import Image, ImageDraw, ImageFont

from django.conf import settings

import os
import io
import shutil
from pathlib import Path

import numpy as np

import pickle
import textwrap
import base64


def standardize(s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
        s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
        return s


def load_image(img):
    global IMAGE_SHAPE

    buffer = np.frombuffer(img, dtype=np.uint8)
    img = Image.open(io.BytesIO(buffer)).convert('RGB')
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    return img


def create_model(tokenizer, mobilenet, output_layer, weights_path):
    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
    
    model.build(input_shape=[(None, 224, 224, 3), (None, None)])
    model.load_weights(str(weights_path)).expect_partial()

    return model


def add_caption(caption, image_path, original_size):
    caption = caption[0].upper() + caption[1:] + '.'

    buffer = np.frombuffer(image_path, dtype=np.uint8)
    img = Image.open(io.BytesIO(buffer)).resize((640, 480))
    width, height = img.size

    font = ImageFont.truetype('arial.ttf', 16)
    _, _, w, h = font.getbbox(caption)
    
    wrapper = textwrap.TextWrapper(width=int(width*0.15))
    word_list = wrapper.wrap(text=caption)

    new_img = Image.new('RGB', (width+10, height+((height//10))), 'black')
    new_img.paste(img, (5, 5, width+5, height+5))

    draw = ImageDraw.Draw(new_img)
    draw.text(((width-w)//2, height+((height//10)-h)//2), '\n'.join(word_list), font=font, fill='white')

    original_img = new_img.resize((int(original_size[0]), int(original_size[1])))

    return caption, new_img, original_img


def encoded_image(img):
    buffer_img = io.BytesIO()
    img.save(buffer_img, format='JPEG')
    buffer_img = buffer_img.getvalue()
    buffer = base64.b64encode(buffer_img).decode('utf-8')

    mime = 'image/jpg'
    mime = mime + ';' if mime else ';'

    encoded = f'data:{mime}base64,{buffer}'

    return encoded


def get_caption(image, original_size):
    global model

    if not model:
        main()

    result = model.simple_gen(load_image(image))
    result, img, original_img = add_caption(result, image, original_size)

    img, original_img = encoded_image(img), encoded_image(original_img)

    return result, img, original_img


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


# Building the model
class SeqEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)
        
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=depth,
            mask_zero=True)
        
        self.add = tf.keras.layers.Add()
    

    def call(self, seq):
        seq = self.token_embedding(seq) # (batch, seq, depth)
        
        x = tf.range(tf.shape(seq)[1])  # (seq)
        x = x[tf.newaxis, :]  # (1, seq)
        x = self.pos_embedding(x)  # (1, seq, depth)
        
        return self.add([seq, x])


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn = self.mha(query=x, value=x,
                        use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)


class CrossAttention(BaseAttention):
    def call(self, x, y, **kwargs):
        attn, attention_scores = self.mha(
                 query=x, value=y,
                 return_attention_scores=True)
        
        self.last_attention_scores = attention_scores
        
        x = self.add([x, attn])
        return self.layernorm(x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=2*units, activation='relu'),
            tf.keras.layers.Dense(units=units),
            tf.keras.layers.Dropout(rate=dropout_rate),
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()
    

    def call(self, x):
        x = self.add([x, self.seq(x)])
        return self.layernorm(x)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        
        self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                                  key_dim=units,
                                                  dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads,
                                              key_dim=units,
                                              dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)
    
    
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs
        
        # Text input
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        out_seq = self.ff(out_seq)
        return out_seq


class Captioner(tf.keras.Model):
    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=5,
               units=512, max_length=50, num_heads=3, dropout_rate=0.2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary())
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True) 
        
        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length)
        
        self.decoder_layers = [
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for n in range(num_layers)]
        
        self.output_layer = output_layer
    

    def simple_gen(self, image):
        initial = self.word_to_index([['[START]']]) # (batch, sequence)
        img_features = self.feature_extractor(image[tf.newaxis, ...])
        
        tokens = initial # (batch, sequence)
        for n in range(50): # 50 words
            preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
            preds = preds[:,-1, :]  #(batch, vocab)

            next = tf.argmax(preds, axis=-1)[:, tf.newaxis]
            tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 
            
            if next[0] == self.word_to_index('[END]'):
                break
        words = self.index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()
    

    def call(self, inputs):
        image, txt = inputs

        if image.shape[-1] == 3:
        # Apply the feature-extractor, if you get an RGB image.
            image = self.feature_extractor(image)
        
        # Flatten the feature map
        image = einops.rearrange(image, 'b h w c -> b (h w) c')
        
        
        if txt.dtype == tf.string:
        # Apply the tokenizer if you get string inputs.
            txt = tokenizer(txt)
        
        txt = self.seq_embedding(txt)
        
        # Look at the image
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))
        
        txt = self.output_layer(txt)
        
        return txt


class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()
        
        self.dense = tf.keras.layers.Dense(
            units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.bias = CustomUnpickler(open(str(settings.BASE_DIR / 'image_caption/Model/bias.pkl'), "rb")).load()
    

    def call(self, x):
        x = self.dense(x)
        return x + self.bias


def main():
    global model, model_data_path, weights_path

    mobilenet = tf.keras.applications.MobileNetV3Large(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True)
    mobilenet.trainable=False

    from_disk = CustomUnpickler(open(str(model_data_path / 'tokenizer.pkl'), "rb")).load()
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=from_disk['config']['max_tokens'],
        standardize=standardize,
        ragged=True)
    tokenizer.set_weights(from_disk['weights'])

    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
    model = create_model(tokenizer, mobilenet, output_layer, weights_path)


model_data_path = settings.BASE_DIR / 'image_caption/Model'
weights_path = model_data_path / 'weights/model.tf'

IMAGE_SHAPE=(224, 224, 3)

model = None