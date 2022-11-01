# Распознавание речи: wav2vec, system combination

[Слайды лекции](https://docs.google.com/presentation/d/1Bu3wWKJqA4f1C1vb4wDgCs85cn4yv4Hz_vHMP-DlwNI/edit?usp=sharing)

## Домашнее задание

Ансамбль CTC-/LAS-конформеров:
* скачать те же [данные](https://drive.google.com/file/d/1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd/view?usp=sharing), что использовались на [неделе 7](../week07/) и распоковать их в `./data`
* оценить Word Error Rate каждого из трех конформеров из [`data/*.ckpt`](./asr/data)
* сагрегировать предсказания трех моделей с помощью ROVER'а, оценить WER
* выбрать лучшую гипотезу из трех с помощью MBR-decoding, оценить WER
* может пригодиться [ноутбук](./asr/asr_ensemble.ipynb)
