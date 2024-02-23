#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2

import numpy as np
from ppocr_onnx.ppocr_onnx import PaddleOcrONNX

from textblob import TextBlob
from spellchecker import SpellChecker







# Function to calculate percentage of corrected words
# def calculate_corrected_percentage(text_array):
#     total_words = 0
#     corrected_words = 0
#     for text_tuple in text_array:
#         text = text_tuple[0]  # Extracting the sentence from the tuple
#         total_words += len(text.split())
#         corrected_text = str(TextBlob(text).correct())
#         corrected_words += sum(1 for a, b in zip(corrected_text.split(), text.split()) if a.lower() == b.lower())
#     return (corrected_words / total_words) * 100 if total_words > 0 else 0





# def calculate_corrected_and_return_corrected_percentage(text_array):
#     total_words = 0
#     corrected_words = 0
#     corrected_word_list = []  # List to store tuples of (original_word, corrected_word)
#     for text_tuple in text_array:
#         text = text_tuple[0]  # Extracting the sentence from the tuple
#         total_words += len(text.split())
#         corrected_text_blob = TextBlob(text).correct()
#         corrected_text = str(corrected_text_blob)
#         corrected_word_list.extend([(original.lower(), corrected.lower()) for original, corrected in zip(text.split(), corrected_text.split()) if original.lower() != corrected.lower()])
#         corrected_words += sum(1 for a, b in zip(corrected_text.split(), text.split()) if a.lower() == b.lower())
    
#     percentage_corrected = (corrected_words / total_words) * 100 if total_words > 0 else 0
#     return percentage_corrected, corrected_word_list


# def ccalculate_corrected_and_return_corrected_percentage(texts):
#     spell = SpellChecker()
#     total_words = 0
#     corrected_words = 0
#     corrected_word_list = []  # List to store tuples of (original_word, corrected_word)
#     for text_tuple in texts:
#         text = text_tuple[0]  # Extracting the sentence from the tuple
#         words = text.split()
#         total_words += len(words)
#         corrected_text = []
#         for word in words:
#             corrected_word = spell.correction(word)
#             if corrected_word.lower() != word.lower():
#                 corrected_word_list.append((word, corrected_word))
#             corrected_text.append(corrected_word)
#         corrected_text = ' '.join(corrected_text)
#         corrected_words += sum(1 for a, b in zip(corrected_text.split(), text.split()) if a.lower() == b.lower())
    
#     percentage_corrected = (corrected_words / total_words) * 100 if total_words > 0 else 0
#     return percentage_corrected, corrected_word_list





import string

def preprocess_text(text):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def calculate_corrected_and_return_corrected_percentage(text_tuples):
    spell = SpellChecker()
    total_words = 0
    corrected_words = 0
    corrected_word_list = []  # List to store tuples of (original_word, corrected_word)
    
    for text_tuple in text_tuples:
        text = text_tuple[0]  # Extracting the sentence from the tuple
        text = preprocess_text(text)
        total_words += len(text.split())
        
        corrected_text = []
        for word in text.split():
            corrected_word = spell.correction(word)
            if corrected_word is not None and corrected_word.lower() != word.lower():
                corrected_word_list.append((word.lower(), corrected_word.lower()))
                corrected_words += 1
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(word)
        
        corrected_text = ' '.join(corrected_text)
    
    percentage_corrected = (corrected_words / total_words) * 100 if total_words > 0 else 0
    return percentage_corrected, corrected_word_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='sample.jpg')

    parser.add_argument(
        "--det_model",
        type=str,
        default='./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx',
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default='./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx',
    )
    parser.add_argument(
        "--rec_char_dict",
        type=str,
        default='./ppocr_onnx/ppocr/utils/dict/en_dict.txt',
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default=
        './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class DictDotNotation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_paddleocr_parameter():
    paddleocr_parameter = DictDotNotation()

    # params for prediction engine
    paddleocr_parameter.use_gpu = False

    # params for text detector
    paddleocr_parameter.det_algorithm = 'DB'
    paddleocr_parameter.det_model_dir = './ppocr_onnx/model/det_model/ch_PP-OCRv3_det_infer.onnx'
    paddleocr_parameter.det_limit_side_len = 960
    paddleocr_parameter.det_limit_type = 'max'
    paddleocr_parameter.det_box_type = 'quad'

    # DB parmas
    paddleocr_parameter.det_db_thresh = 0.3
    paddleocr_parameter.det_db_box_thresh = 0.6
    paddleocr_parameter.det_db_unclip_ratio = 1.5
    paddleocr_parameter.max_batch_size = 10
    paddleocr_parameter.use_dilation = False
    paddleocr_parameter.det_db_score_mode = 'fast'

    # params for text recognizer
    paddleocr_parameter.rec_algorithm = 'SVTR_LCNet'
    paddleocr_parameter.rec_model_dir = './ppocr_onnx/model/rec_model/japan_PP-OCRv3_rec_infer.onnx'
    paddleocr_parameter.rec_image_shape = '3, 48, 320'
    paddleocr_parameter.rec_batch_num = 6
    paddleocr_parameter.rec_char_dict_path = './ppocr_onnx/ppocr/utils/dict/japan_dict.txt'
    paddleocr_parameter.use_space_char = True
    paddleocr_parameter.drop_score = 0.5

    # params for text classifier
    paddleocr_parameter.use_angle_cls = False
    paddleocr_parameter.cls_model_dir = './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    paddleocr_parameter.cls_image_shape = '3, 48, 192'
    paddleocr_parameter.label_list = ['0', '180']
    paddleocr_parameter.cls_batch_num = 6
    paddleocr_parameter.cls_thresh = 0.9

    paddleocr_parameter.save_crop_res = False

    return paddleocr_parameter


def main():

    args = get_args()
    #  --image=1.png
    # enable for input from cmd line
    # image_path = args.image
    image_path="2.png"
    image = cv2.imread(image_path)

    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)


    image = cv2.imread(image_path)


    dt_boxes, rec_res, time_dict = paddle_ocr_onnx(image)




    percentage_corrected, corrected_word_list = calculate_corrected_and_return_corrected_percentage(rec_res)
    # Print the percentage
    print("----------------------------------------------------------------")
    print("Percentage of corrected words: {:.2f}%".format(percentage_corrected))
    print("----------------------------------------------------------------")
    # Print words that were corrected
    print("\nWords that were corrected:")
    for original_word, corrected_word in corrected_word_list:
        print("Original: {:15s} Corrected: {:15s}".format(original_word, corrected_word))






    print("----------------------------------------------------------------")
    print("----------------------------------------------------------------")










    # print(time_dict)
    # print(dt_boxes)
    for dt_box, rec in zip(dt_boxes, rec_res):
        # print(dt_box)
        # print(bbox)
        # print(rec)
        # print(type(rec))
        print(rec[0])


if __name__ == '__main__':
    main()
