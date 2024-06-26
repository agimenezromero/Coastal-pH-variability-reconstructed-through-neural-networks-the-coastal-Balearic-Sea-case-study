# Coastal pH variability reconstructed through neural networks: the coastal Balearic Sea case study
This repository contains the code and data used to produce the results published in [Scientific Reports](https://www.nature.com/articles/s41598-022-17253-5)

## Abstract

The decreasing seawater pH trend associated with increasing atmospheric carbon dioxide levels is an issue of concern due to possible negative consequences for marine organisms, especially calcifiers. Globally, coastal areas represent important transitional land-ocean zones with complex interactions between biological, physical and chemical processes. Here, we evaluated the pH variability at two sites in the coastal area of the Balearic Sea (Western Mediterranean). High resolution pH data along with temperature, salinity and dissolved oxygen were obtained with autonomous sensors from 2018-2021 in order to determine the temporal pH variability and the principal drivers involved. By using environmental datasets of temperature, salinity and dissolved oxygen, Recurrent Neural Networks were trained to predict pH and fill data gaps. Longer environmental time-series (2012-2021) were used to obtain the pH trend using reconstructed data. The best predictions show a rate of -0.0025 +/- 0.00053 pH units per year, which is in good agreement with other observations of pH rates in coastal areas. The methodology presented here opens the possibility to obtain pH trends when only limited pH observations are available, if other variables are accessible. Potentially, this could be a way to reliably fill the unavoidable gaps present in time-series data provided by sensors.

## Application example

We provide the trained models for Palma bay and Cabrera, `Bidirectional_LSTM_Palma_final.h5` and `Bidirectional_LSTM_Cabrera_final.h5`, respectively.

To apply those models (that in principle are just valid for those locations, but you can try with data from other zones!) just see the example in `Model_application.ipynb`.

# Requirements

The `Model_application.ipynb` notebook should run without problems if the following libraries are installed together with Python 3.xx (tested with 3.11)

- Tensorflow 2.12
- Pandas
- Matplotlib
